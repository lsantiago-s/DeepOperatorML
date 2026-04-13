from __future__ import annotations

import os
import platform
from functools import lru_cache
from pathlib import Path
from ctypes import CDLL, POINTER, byref, c_double, c_long


def _resolve_library_name() -> str:
    system = platform.system()
    if system == "Windows":
        return "axsgrsce.dll"
    if system == "Darwin":
        return "axsgrsce.dylib"
    if system == "Linux":
        return "axsgrsce.so"
    raise OSError(f"Unsupported operating system: {system}")


def _candidate_library_paths(lib_name: str) -> list[Path]:
    current_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    env_path = os.environ.get("RAJAPAKSE_AXSGRSCE_LIB")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(current_dir / "libs" / lib_name)
    candidates.append(current_dir.parent / "rajapakse_homogeneous" / "libs" / lib_name)
    return candidates


def _configure_axsanisgreen_signature(lib: CDLL) -> None:
    lib.axsanisgreen.argtypes = [
        POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
        POINTER(c_double), POINTER(c_double),
        POINTER(c_double), POINTER(c_double),
        POINTER(c_double), POINTER(c_double), POINTER(c_double),
        POINTER(c_double),
        POINTER(c_long), POINTER(c_long), POINTER(c_long),
        POINTER(c_double), POINTER(c_double),
    ]
    lib.axsanisgreen.restype = None


def _build_load_error_message(lib_name: str, errors: list[tuple[Path, OSError]]) -> str:
    tried = "\n".join([f"- {path}: {err}" for path, err in errors])
    has_glibc_mismatch = any("GLIBC_" in str(err) for _, err in errors)
    if has_glibc_mismatch and platform.system() == "Linux":
        return (
            f"Failed to load '{lib_name}' due to Linux runtime compatibility.\n"
            "Detected a glibc symbol mismatch (for example GLIBC_2.34 not found).\n"
            "Rebuild axsgrsce.so on a Linux environment compatible with this runtime, "
            "or provide a compatible shared library via RAJAPAKSE_AXSGRSCE_LIB.\n"
            f"Tried:\n{tried}"
        )
    return f"Failed to load '{lib_name}'. Tried:\n{tried}"


@lru_cache(maxsize=1)
def _load_library() -> CDLL:
    lib_name = _resolve_library_name()
    candidates = _candidate_library_paths(lib_name=lib_name)
    existing_paths = [path for path in candidates if path.exists()]
    if not existing_paths:
        all_paths = "\n".join([f"- {path}" for path in candidates])
        raise FileNotFoundError(
            f"Library '{lib_name}' was not found. Checked:\n{all_paths}\n"
            "Set RAJAPAKSE_AXSGRSCE_LIB to a valid shared-library path if needed."
        )

    errors: list[tuple[Path, OSError]] = []
    for lib_path in existing_paths:
        try:
            lib = CDLL(str(lib_path))
            _configure_axsanisgreen_signature(lib)
            return lib
        except OSError as exc:
            errors.append((lib_path, exc))

    raise RuntimeError(_build_load_error_message(lib_name=lib_name, errors=errors))


def load_native_library() -> CDLL:
    """Backward-compatible public loader."""
    return _load_library()


def _ensure_library_available() -> None:
    _load_library()


def influence(c11_val, c12_val, c13_val, c33_val, c44_val,
               dens_val, damp_val,
               r_campo_val, z_campo_val,
               z_fonte_val, r_fonte_val, l_fonte_val,
               freq_val,
               bvptype_val, loadtype_val, component_val):
    lib = _load_library()

    c11 = c_double(c11_val)
    c12 = c_double(c12_val)
    c13 = c_double(c13_val)
    c33 = c_double(c33_val)
    c44 = c_double(c44_val)
    dens = c_double(dens_val)
    damp = c_double(damp_val)
    r = c_double(r_campo_val)
    z = c_double(z_campo_val)
    h = c_double(z_fonte_val)
    loadr = c_double(r_fonte_val)
    loadh = c_double(l_fonte_val)
    omega = c_double(freq_val)
    bvptype = c_long(bvptype_val)
    loadtype = c_long(loadtype_val)
    component = c_long(component_val)

    resultr = c_double()
    resulti = c_double()

    lib.axsanisgreen(
        byref(c11), byref(c12), byref(c13), byref(c33), byref(c44),
        byref(dens), byref(damp),
        byref(r), byref(z),
        byref(h), byref(loadr), byref(loadh),
        byref(omega),
        byref(bvptype), byref(loadtype), byref(component),
        byref(resultr), byref(resulti)
    )

    wd = resultr.value + 1j * resulti.value
    return wd
