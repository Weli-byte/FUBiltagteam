"""Raw/derived data must never be tracked by git (it lives in DVC)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = re.compile(r"^data/(raw|interim|processed)/(.*/)?(\.gitkeep|\.gitignore|[^/]+\.dvc)$")


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_no_data_files_are_tracked() -> None:
    out = subprocess.run(
        ["git", "ls-files", "data/raw", "data/interim", "data/processed"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    offenders = [path for path in out if not ALLOWED.match(path)]
    assert not offenders, f"Veri dosyaları git'e girmiş (DVC kullanın): {offenders[:5]}"
