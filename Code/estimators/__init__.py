"""Registry aller Estimators -- hier wird ein neues Verfahren eingetragen.

Ein Eintrag besteht aus einer Factory (ohne Argumente bzw. mit Defaults) und
der Kategorie. Die Kategorie haengt *nicht* am Estimator-Modul, sondern wird
erst hier vergeben: ob ein Verfahren real umsetzbar ist, entscheidet das
Oracle, und dasselbe Verfahren kann mit anderem Oracle in die andere Kategorie
fallen. `build()` setzt das Label nach der Konstruktion auf die Instanz.

Die Random-Walk-Varianten werden als Kreuzprodukt erzeugt:
    Sackgassen-Strategie (restart | backtrack | history)
  x Thinning             (none | simple | shifted)

Schnittstelle:
    REGISTRY: dict[str, Entry]
    register(name, factory, category)
    build(name) -> Estimator            (setzt .name und .category)
    build_all(selected=None, category=None) -> list[Estimator]
    names(category=None) -> list[str]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from estimators.base import Category, Estimator
from estimators.methods import (capture_recapture, deg_weighted_independent,
                                random_walk_collision, short_walk_independent,
                                uniform_collision)
from sampling.dead_ends import DEAD_ENDS
from sampling.thinning import THINNINGS


@dataclass(frozen=True)
class Entry:
    factory: Callable[[], Estimator]
    category: Category


REGISTRY: dict[str, Entry] = {
    # -- Vergleich: gleichverteiltes Ziehen aus V ------------------------
    "uniform_collision": Entry(
        partial(uniform_collision.build, formula="uis-collision"), Category.COMPARISON),
    "uniform_collision_weighted": Entry(
        partial(uniform_collision.build, formula="wis-col-katzir"), Category.COMPARISON),
}

# -- Real umsetzbar: Random Walk x Sackgassen-Strategie x Thinning -------
for _dead_end in DEAD_ENDS:
    for _thinning in THINNINGS:
        REGISTRY[f"rw_plain__{_dead_end}__{_thinning}"] = Entry(
            partial(random_walk_collision.build,
                    dead_end=_dead_end, thinning=_thinning, formula="uis-collision"),
            Category.REALIZABLE,
        )
    REGISTRY[f"capture_recapture__{_dead_end}"] = Entry(
        partial(capture_recapture.build, dead_end=_dead_end), Category.REALIZABLE)

# -- WIS-Vergleichsreihe -------------------------------------------------
# Dieselbe Formel (Katzir) auf zwei Sampling-Verfahren mit
# derselben Verteilung pi(v) ~ deg(v):
#   *__indep : unabhaengige Ziehungen  -> nur Gradverzerrung
#   *__rw-*  : echter Random Walk      -> Gradverzerrung + Autokorrelation
# Die Differenz ist der Preis der Abhaengigkeit. Referenz fuer beide ist
# "uniform_collision" (gleichverteilt, ohne Gewicht).
REGISTRY["wis-katzir__indep"] = Entry(
    partial(deg_weighted_independent.build, formula="wis-col-katzir"),
    Category.COMPARISON)
for _de in DEAD_ENDS:
    REGISTRY[f"wis-katzir__rw-{_de}"] = Entry(
        partial(random_walk_collision.build,
                dead_end=_de, thinning="none", formula="wis-col-katzir"),
        Category.REALIZABLE,
    )
del _de

# -- Kurze unabhaengige Walks -------------------------------------------
# Endknoten eines 5-Schritt-Walks je Sample: Walk-Verzerrung ohne
# Autokorrelation. Beide Formeln auf denselben Samples, damit ablesbar ist, was
# die Gradgewichtung bewirkt -- auf `undirected` passt sie, auf `directed`
# nicht. Ein Sample kostet eine Query, die Schritte sind gratis.
for _f, _tag in (("wis-col-katzir", "wis-katzir"), ("uis-collision", "uis")):
    REGISTRY[f"{_tag}__walk5"] = Entry(
        partial(short_walk_independent.build, formula=_f, steps=5),
        Category.COMPARISON)
del _f, _tag

# Gradkorrigierte Referenz -- zeigt, was die Verzerrung des Walks ausmacht.
REGISTRY["rw_weighted__restart__none"] = Entry(
    partial(random_walk_collision.build,
            dead_end="restart", thinning="none", formula="wis-col-katzir"),
    Category.REALIZABLE,
)

del _dead_end, _thinning


def register(name: str, factory: Callable[[], Estimator], category: Category) -> None:
    REGISTRY[name] = Entry(factory, category)


def build(name: str) -> Estimator:
    entry = REGISTRY[name]
    est = entry.factory()
    est.name = name
    est.category = entry.category  # Label erst hier, nach der Konstruktion
    return est


def names(category: Category | None = None) -> list[str]:
    return [n for n, e in REGISTRY.items() if category is None or e.category == category]


def build_all(selected: list[str] | None = None, category: Category | None = None) -> list[Estimator]:
    return [build(n) for n in (selected or names(category)) if category is None
            or REGISTRY[n].category == category]
