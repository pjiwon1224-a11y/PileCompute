# function/pilelib/__init__.py
from importlib import import_module
from pkgutil import iter_modules
import warnings as _warn

__all__ = []

def _export_symbols(mod):
    names = getattr(mod, "__all__", None)
    if names is None:
        names = [k for k in mod.__dict__ if not k.startswith("_")]
    for k in names:
        globals()[k] = getattr(mod, k)
    return list(names)

for info in iter_modules(__path__):
    if info.ispkg:
        continue
    name = info.name
    try:
        m = import_module(f"{__name__}.{name}")
        __all__ += _export_symbols(m)
    except Exception as e:
        _warn.warn(f"[pilelib] fail import module '{name}': {e}")
