"""Minifier registry: importing this package registers every family module."""
from .base import REGISTRY, detect, minify, preserves, register, strip_marks  # noqa: F401
import importlib as _il
for _mod in ("python_family", "brace", "config_docs", "outputs"):
    try:
        _il.import_module(f"{__name__}.{_mod}")
    except ModuleNotFoundError as _e:          # a family not written yet
        if _e.name != f"{__name__}.{_mod}":
            raise
