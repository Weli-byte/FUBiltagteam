"""Print 5 random labeled patients (seed 1) for manual checking against cBioPortal (read-only)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    path = ROOT / "data" / "processed" / "labels" / "labels.parquet"
    labels = pd.read_parquet(path)
    genes = [
        c
        for c in labels.columns
        if c not in {"patient_barcode", "project_id", "label_status", "label_sample_barcode"}
    ]
    labeled = labels[labels["label_status"] == "labeled"]
    sample = labeled.sample(min(5, len(labeled)), random_state=1)
    columns = ["patient_barcode", "project_id", "label_sample_barcode", *genes]
    print(sample[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
