"""Build and load the two small CPU kernels; no model or GPU allocation."""

import ctypes
import fcntl
import hashlib
import os
import platform
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np

_libraries = {}
_lock = threading.RLock()
_blas = None
_dot = None


def dot_pointer():
    """Use NumPy's tested ILP64 OpenBLAS reduction, including floating ties."""
    global _blas, _dot
    with _lock:
        if _dot is None:
            candidates = sorted((Path(np.__file__).parent.parent / "numpy.libs").glob("*openblas64*"))
            if not candidates:
                raise RuntimeError(
                    "SpectralKV requires a NumPy ILP64 OpenBLAS wheel; install the tested requirements.txt."
                )
            _blas = ctypes.CDLL(str(candidates[0]))
            _dot = ctypes.cast(_blas.scipy_cblas_ddot64_, ctypes.c_void_p)
        return _dot


def _load(name):
    with _lock:
        if name in _libraries:
            return _libraries[name]
        source = Path(__file__).parent / "csrc" / f"{name}.cpp"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        cache = (
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "spectralkv" / platform.machine()
        )
        cache.mkdir(parents=True, exist_ok=True)
        binary = cache / f"{name}_{digest}.so"
        with (cache / "build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not binary.exists():
                compiler = shutil.which(os.environ.get("CXX", "c++"))
                if compiler is None:
                    raise RuntimeError("A C++17 compiler is required on first use. Install g++ or set CXX.")
                temporary = binary.with_suffix(f".{os.getpid()}.tmp.so")
                try:
                    subprocess.run(
                        [
                            compiler,
                            "-O3",
                            "-std=c++17",
                            "-ffp-contract=off",
                            "-shared",
                            "-fPIC",
                            str(source),
                            "-o",
                            str(temporary),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    temporary.replace(binary)
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(f"SpectralKV C++ compilation failed:\n{error.stderr}") from error
                finally:
                    temporary.unlink(missing_ok=True)
        library = ctypes.CDLL(str(binary))
        if name == "sparse":
            library.prepare.restype = ctypes.c_int64
            library.prepare.argtypes = [ctypes.c_int64] * 2 + [ctypes.c_void_p] * 9
        else:
            library.solve.restype = None
            library.solve.argtypes = [ctypes.c_int64] * 3 + [ctypes.c_void_p] * 11
            dot_pointer()
        _libraries[name] = library
        return library


def sparse():
    return _load("sparse")


def greedy():
    return _load("greedy")
