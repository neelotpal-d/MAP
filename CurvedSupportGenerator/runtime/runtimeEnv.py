"""Runtime environment guards for local research scripts."""

import os


def configure_openmp_runtime() -> None:
    """Allow execution when multiple OpenMP runtimes are loaded on Windows.

    This is a workaround for environments where packages such as PyTorch,
    NumPy, libigl, or MKL-linked libraries bring in conflicting OpenMP
    runtimes (`libomp.dll` vs `libiomp5md.dll`).

    It is not the ideal long-term fix, but it keeps the scripts runnable in
    the current local environment.
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
