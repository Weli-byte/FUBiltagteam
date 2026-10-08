from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from onkos.common.checkpoint import CheckpointManager  # noqa: E402


def _model_opt() -> tuple[Any, Any]:
    model = torch.nn.Linear(3, 1)
    return model, torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


def test_save_load_roundtrip_restores_weights_and_optimizer(tmp_path: Path) -> None:
    model, opt = _model_opt()
    model(torch.ones(2, 3)).sum().backward()
    opt.step()
    mgr = CheckpointManager(tmp_path)
    mgr.save(model, opt, step=5, metric=1.0, config={"a": 1})

    model2, opt2 = _model_opt()
    meta = mgr.load(model2, opt2)
    assert meta["step"] == 5 and meta["config"] == {"a": 1}
    assert torch.equal(model.weight, model2.weight)
    assert opt2.state_dict()["state"]  # momentum buffers restored


def test_best_only_updates_on_improvement(tmp_path: Path) -> None:
    model, opt = _model_opt()
    mgr = CheckpointManager(tmp_path, mode="min")
    assert mgr.save(model, opt, step=1, metric=0.5)
    assert not mgr.save(model, opt, step=2, metric=0.7)
    assert mgr.save(model, opt, step=3, metric=0.2)
    assert mgr.best_metric() == pytest.approx(0.2)
    assert mgr.load(model, which="last")["step"] == 3
