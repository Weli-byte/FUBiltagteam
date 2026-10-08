"""Git metadata helper used to tie experiments and checkpoints to code versions."""

import subprocess
from pathlib import Path


def git_commit_hash() -> str | None:
    """Return the commit hash of the ONKOS checkout (not the caller's cwd), or ``None``."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None
