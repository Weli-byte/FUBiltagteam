"""Sample pack (patch_v0), slide-level splits and the baseline training loop (1.3 / 1.9)."""

from pathlib import Path

import pytest

pytest.importorskip("openslide")
pytest.importorskip("cv2")
pytest.importorskip("tifffile")
torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from torch.utils.data import DataLoader  # noqa: E402

from models.histo.patch_classifier.data import (  # noqa: E402
    assert_slide_disjoint,
    class_weights,
    label_counts,
    load_labelled_rows,
    make_loader,
    rows_for_split,
    split_rows_by_slide,
)
from models.histo.patch_classifier.models import build_model, layerwise_lr_groups  # noqa: E402
from models.histo.patch_classifier.train import (  # noqa: E402
    TrainConfig,
    load_best,
    predict,
    predict_tta,
    train_classifier,
)
from models.histo.patches.dataset import SamplePackDataset  # noqa: E402
from models.histo.patches.grid import PatchGridConfig  # noqa: E402
from models.histo.patches.sample_pack import (  # noqa: E402
    PackSlide,
    SamplePackConfig,
    build_sample_pack,
    read_rows,
)
from models.histo.stain.augment import AugmentConfig  # noqa: E402
from models.histo.wsi.tissue import TissueMaskConfig  # noqa: E402
from tests.histo_fixtures import make_slide, make_xml  # noqa: E402

GRID = PatchGridConfig(patch_px=64, target_mpp=0.5, min_tissue_fraction=0.5, thumbnail_px=512)
MASK = TissueMaskConfig(min_object_px=20, max_hole_px=200)


@pytest.fixture(scope="module")
def pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("pack")
    slides = []
    for i in range(4):
        path, _ = make_slide(root, f"tumor_00{i + 1}", seed=i)
        xml = make_xml(root, f"tumor_00{i + 1}")
        slides.append(PackSlide(f"tumor_00{i + 1}", path, "camelyon16", "train", "tumor", xml))
    path, _ = make_slide(root, "normal_001", tumor_rect=None, seed=9)
    slides.append(PackSlide("normal_001", path, "camelyon16", "train", "normal", None))
    path, _ = make_slide(root, "test_001", seed=5)
    slides.append(
        PackSlide("test_001", path, "camelyon16", "test", None, make_xml(root, "test_001"))
    )
    path, _ = make_slide(root, "tcga-uuid-0001", seed=6)
    slides.append(PackSlide("tcga-uuid-0001", path, "tcga", "unassigned", "tumor", None))
    out = root / "patch_v0"
    build_sample_pack(slides, out, SamplePackConfig(patches_per_slide=24, seed=1), GRID, MASK)
    return out


def test_pack_contents_and_labels(pack: Path) -> None:
    rows = read_rows(pack / "patches.csv")
    assert rows and (pack / "pack_meta.json").exists()
    assert all((pack / r["image"]).exists() for r in rows)
    by_slide = {s: [r for r in rows if r["slide_id"] == s] for s in {r["slide_id"] for r in rows}}
    tumour_rows = by_slide["tumor_001"]
    assert {r["label_method"] for r in tumour_rows} == {"camelyon_xml_mask"}
    assert {int(r["label"]) for r in tumour_rows} == {0, 1}  # balanced: both classes present
    assert all(float(r["tumor_fraction"]) >= 0.5 for r in tumour_rows if int(r["label"]) == 1)
    assert all(float(r["tumor_fraction"]) == 0.0 for r in tumour_rows if int(r["label"]) == 0)
    assert {int(r["label"]) for r in by_slide["normal_001"]} == {0}
    assert {r["label_method"] for r in by_slide["tcga-uuid-0001"]} == {"slide_level"}
    assert all(int(r["patch_level0_px"]) == 64 for r in rows)
    assert (pack / "overlays" / "tumor_001.png").exists()
    import cv2

    img = cv2.imread(str(pack / rows[0]["image"]))
    assert img is not None and img.shape == (64, 64, 3)


def test_pack_sampling_is_deterministic_across_processes(tmp_path: Path) -> None:
    """Same slides, same seed -> identical patches.csv, even in a fresh interpreter (hash salt)."""
    import subprocess
    import sys

    path, _ = make_slide(tmp_path, "tumor_009", seed=3)
    xml = make_xml(tmp_path, "tumor_009")
    code = "\n".join(
        [
            "import sys",
            "from pathlib import Path",
            "from models.histo.patches.grid import PatchGridConfig as G",
            "from models.histo.patches import sample_pack as sp",
            "from models.histo.wsi.tissue import TissueMaskConfig as M",
            "a = sys.argv",
            "PackSlide, SamplePackConfig = sp.PackSlide, sp.SamplePackConfig",
            "build_sample_pack = sp.build_sample_pack",
            "slide = PackSlide('tumor_009', Path(a[1]), 'c16', 'train', 'tumor', Path(a[2]))",
            "cfg = SamplePackConfig(patches_per_slide=20, seed=5)",
            "grid = G(patch_px=64, target_mpp=0.5, thumbnail_px=512)",
            "rows = build_sample_pack([slide], Path(a[3]), cfg, grid, M(min_object_px=20))",
            "print(sorted((r['x'], r['y']) for r in rows))",
        ]
    )
    outputs = []
    for run in ("a", "b"):
        env = {**__import__("os").environ, "PYTHONHASHSEED": "1" if run == "a" else "2"}
        result = subprocess.run(
            [sys.executable, "-c", code, str(path), str(xml), str(tmp_path / run)],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        outputs.append(result.stdout.strip().splitlines()[-1])
    assert outputs[0] == outputs[1]


def test_dataset_and_loader_from_pack(pack: Path) -> None:
    rows = load_labelled_rows(pack)
    dataset = SamplePackDataset(pack, rows[:8], image_px=32)
    item = dataset[0]
    assert item["image"].shape == (3, 32, 32) and item["label"] in (0, 1)
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    assert batch["image"].shape == (4, 3, 32, 32) and batch["label"].shape == (4,)


def test_slide_level_split_and_test_lock(pack: Path) -> None:
    rows = load_labelled_rows(pack)
    train_all = rows_for_split(rows, "train")
    train_rows, val_rows = split_rows_by_slide(train_all, val_fraction=0.25, seed=0)
    assert train_rows and val_rows
    assert_slide_disjoint(train_rows, val_rows)
    with pytest.raises(AssertionError):
        assert_slide_disjoint(train_rows, train_rows[:1])
    with pytest.raises(PermissionError):
        rows_for_split(rows, "test")
    assert rows_for_split(rows, "test", final_evaluation=True)
    counts = label_counts(train_rows)
    assert counts["normal"] > 0 and counts["tumor"] > 0
    weights = class_weights(train_rows)
    assert len(weights) == 2 and abs(sum(weights) / 2 - 1.0) < 1e-6


def test_layerwise_groups_decay_towards_the_stem() -> None:
    timm = pytest.importorskip("timm")
    vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=2)
    groups = layerwise_lr_groups(vit, base_lr=1e-3, decay=0.8, weight_decay=0.05)
    lrs = {g["layer_id"]: g["lr"] for g in groups}
    depth = len(vit.blocks)
    assert lrs[depth + 1] == pytest.approx(1e-3) and lrs[0] == pytest.approx(
        1e-3 * 0.8 ** (depth + 1)
    )
    assert all(lrs[i] < lrs[i + 1] for i in range(depth + 1))
    assert any(g["weight_decay"] == 0.0 for g in groups) and any(
        g["weight_decay"] == 0.05 for g in groups
    )
    flat = layerwise_lr_groups(build_model("tiny_cnn"), 1e-3, 0.8, 0.05)
    assert {g["lr"] for g in flat} == {1e-3}


def test_training_loop_runs_checkpoints_and_predicts(pack: Path, tmp_path: Path) -> None:
    rows = load_labelled_rows(pack)
    train_rows, val_rows = split_rows_by_slide(rows_for_split(rows, "train"), 0.25, seed=0)
    cfg = TrainConfig(
        model="tiny_cnn",
        pretrained=False,
        epochs=2,
        batch_size=8,
        lr=1e-2,
        amp=False,
        image_px=32,
        max_steps_per_epoch=3,
        early_stop_patience=5,
        device="cpu",
        seed=1,
        augment=AugmentConfig(blur_p=0.0, noise_p=0.0),
    )
    logged: list[tuple[str, float, int]] = []
    outcome = train_classifier(
        cfg, pack, train_rows, val_rows, tmp_path / "run", lambda k, v, s: logged.append((k, v, s))
    )
    assert len(outcome.history) == 2 and outcome.best_epoch in (0, 1)
    assert (tmp_path / "run" / "last.pt").exists() and (tmp_path / "run" / "best.pt").exists()
    assert len(outcome.val_y) == len(val_rows) == len(outcome.val_p)
    assert {k for k, _, _ in logged} == {"train_loss", "val_loss", "val_auroc", "val_auprc"}

    model = load_best(cfg, tmp_path / "run", device="cpu")
    loader = make_loader(pack, val_rows, 8, False, 0, image_px=32)
    y, p, slides, _ = predict(model, loader, "cpu", amp=False)
    assert len(y) == len(val_rows) and set(slides) <= {r["slide_id"] for r in val_rows}
    assert ((p >= 0) & (p <= 1)).all()
    y2, p2, _ = predict_tta(model, loader, "cpu", amp=False)
    assert np.array_equal(y, y2) and p2.shape == p.shape


def test_same_seed_same_losses(pack: Path, tmp_path: Path) -> None:
    rows = load_labelled_rows(pack)
    train_rows, val_rows = split_rows_by_slide(rows_for_split(rows, "train"), 0.25, seed=0)
    cfg = TrainConfig(
        model="tiny_cnn",
        pretrained=False,
        epochs=1,
        batch_size=8,
        amp=False,
        image_px=32,
        max_steps_per_epoch=2,
        device="cpu",
        seed=3,
    )
    a = train_classifier(cfg, pack, train_rows, val_rows, tmp_path / "a").history[0]
    b = train_classifier(cfg, pack, train_rows, val_rows, tmp_path / "b").history[0]
    assert a.train_loss == pytest.approx(b.train_loss) and a.val_auroc == pytest.approx(b.val_auroc)
