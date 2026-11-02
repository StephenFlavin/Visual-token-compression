"""Scheme registry. ``get_scheme(name, **kwargs)``; modules are imported lazily."""
from __future__ import annotations

from vbpe.schemes.base import Scheme


def get_scheme(name: str, **kw) -> Scheme:
    from vbpe import config

    if name == "latin":
        from vbpe.schemes.latin import LatinScheme
        return LatinScheme(config.LATIN_FONT, "latin", **kw)
    if name == "latin_condensed":
        from vbpe.schemes.latin import LatinScheme
        return LatinScheme(config.CONDENSED_FONT, "latin_condensed", **kw)
    if name.startswith("subst"):       # e.g. subst300, subst1000, subst300_legend (vbpe/schemes/substitution.py)
        from vbpe.schemes.substitution import make_substitution_scheme
        return make_substitution_scheme(name, **kw)
    if name.startswith("quad"):        # e.g. quad2, quad3, quad4   (vbpe/schemes/quadrant.py)
        from vbpe.schemes.quadrant import make_quadrant_scheme
        return make_quadrant_scheme(name, **kw)
    raise KeyError(f"unknown scheme {name!r}")


SCHEMES = ["latin", "latin_condensed", "subst300", "subst1000", "subst300_legend", "quad2", "quad3", "quad4"]
