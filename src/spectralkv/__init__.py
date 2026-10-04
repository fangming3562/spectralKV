"""SpectralKV: redundancy-aware, byte-budgeted KV compression."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from .config import Config

__version__ = "0.1.0"
__all__ = ["Config", "generate"]


def generate(*args, **kwargs):
    from .api import generate as implementation

    return implementation(*args, **kwargs)
