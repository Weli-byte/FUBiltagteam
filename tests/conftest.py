"""Test-wide settings."""

import os

# Small synthetic problems: one OpenMP/BLAS thread is faster and avoids oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
