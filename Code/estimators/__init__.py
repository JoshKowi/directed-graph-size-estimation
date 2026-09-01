"""Registry aller Estimators -- hier wird ein neues Verfahren eingetragen.

Ein Eintrag besteht aus einer Factory (ohne Argumente bzw. mit Defaults) und
der Kategorie. Die Kategorie haengt *nicht* am Estimator-Modul, sondern wird
erst hier vergeben: ob ein Verfahren real umsetzbar ist, entscheidet das
Oracle, und dasselbe Verfahren kann mit anderem Oracle in die andere Kategorie
fallen. `build()` setzt das Label nach der Konstruktion auf die Instanz.

Die Random-Walk-Varianten werden als Kreuzprodukt erzeugt:
    Sackgassen-Strategie (restart | backtrack | history)
  x Umgang mit Abhaengigkeit (none | simple | shifted | margin)

`margin` steht im selben Namensslot wie das Thinning, ist aber keines: es
verwirft keine Samples, sondern laesst bei der Kollisionszaehlung Paare aus,
die im Walk weniger als m+1 Schritte auseinanderliegen (estimators.formulas).
Die Groesse kommt aus config.SAFETY_MARGIN und laesst sich im Namen
ueberschreiben: `rw-plain__restart__margin20` benutzt m = 20. Dasselbe gilt
fuer die Zahl der Faenge bei Schnabel (`capture-recapture__restart__schnabel8`).
Solche Namen stehen nicht in der REGISTRY, `build()` loest sie zur Laufzeit
auf -- `names()` und `build_all()` listen deshalb nur die Default-Variante.

Schnittstelle:
    REGISTRY: dict[str, Entry]
    register(name, factory, category)
    build(name) -> Estimator            (setzt .name und .category)
    build_all(selected=None, category=None) -> list[Estimator]
    names(category=None) -> list[str]
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import config
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
    "uniform-collision": Entry(
        partial(uniform_collision.build, formula="uis-collision"), Category.COMPARISON),
    "uniform-collision__weighted": Entry(
        partial(uniform_collision.build, formula="wis-col-katzir"), Category.COMPARISON),
}

# -- Real umsetzbar: Random Walk x Sackgassen-Strategie x Thinning -------
for _dead_end in DEAD_ENDS:
    for _thinning in THINNINGS:
        REGISTRY[f"rw-plain__{_dead_end}__{_thinning}"] = Entry(
            partial(random_walk_collision.build,
                    dead_end=_dead_end, thinning=_thinning, formula="uis-collision"),
            Category.REALIZABLE,
        )
    # Safety Margin: vierter Wert im Thinning-Slot, aber kein Thinning (s.o.).
    # Immer mit thinning="none" -- beide zusammen waeren doppelt gemoppelt.
    REGISTRY[f"rw-plain__{_dead_end}__margin"] = Entry(
        partial(random_walk_collision.build, dead_end=_dead_end, thinning="none",
                margin=config.SAFETY_MARGIN, formula="uis-collision"),
        Category.REALIZABLE,
    )
    # Capture-Recapture: dieselben Faenge, drei Formeln darueber. Der Name ohne
    # Zusatz bleibt Lincoln-Petersen, damit vorhandene Aufrufe weiter gelten.
    REGISTRY[f"capture-recapture__{_dead_end}"] = Entry(
        partial(capture_recapture.build, dead_end=_dead_end), Category.REALIZABLE)
    REGISTRY[f"capture-recapture__{_dead_end}__chapman"] = Entry(
        partial(capture_recapture.build, dead_end=_dead_end, formula="chapman"),
        Category.REALIZABLE)
    REGISTRY[f"capture-recapture__{_dead_end}__schnabel"] = Entry(
        partial(capture_recapture.build, dead_end=_dead_end, formula="schnabel",
                n_captures=config.DEFAULT_CAPTURES),
        Category.REALIZABLE)
    # Kollisionen zwischen den Faengen -- die einzige Capture-Recapture-Form,
    # die sich gradkorrigieren laesst. Beide Gewichtungen auf denselben Faengen.
    for _cf in ("cross", "cross-wis"):
        REGISTRY[f"capture-recapture__{_dead_end}__{_cf}"] = Entry(
            partial(capture_recapture.build, dead_end=_dead_end, formula=_cf),
            Category.REALIZABLE)

# -- WIS-Vergleichsreihe -------------------------------------------------
# Dieselbe Formel (Katzir) auf zwei Sampling-Verfahren mit
# derselben Verteilung pi(v) ~ deg(v):
#   *__indep : unabhaengige Ziehungen  -> nur Gradverzerrung
#   *__rw-*  : echter Random Walk      -> Gradverzerrung + Autokorrelation
# Die Differenz ist der Preis der Abhaengigkeit. Referenz fuer beide ist
# "uniform-collision" (gleichverteilt, ohne Gewicht).
REGISTRY["wis-katzir__indep"] = Entry(
    partial(deg_weighted_independent.build, formula="wis-col-katzir"),
    Category.COMPARISON)
for _de in DEAD_ENDS:
    REGISTRY[f"wis-katzir__rw-{_de}"] = Entry(
        partial(random_walk_collision.build,
                dead_end=_de, thinning="none", formula="wis-col-katzir"),
        Category.REALIZABLE,
    )
    # dieselbe Formel mit Safety Margin -- auf `undirected` die interessantere
    # Reihe, weil dort pi ~ deg stimmt und nur die Abhaengigkeit stoert
    REGISTRY[f"wis-katzir__rw-{_de}__margin"] = Entry(
        partial(random_walk_collision.build, dead_end=_de, thinning="none",
                margin=config.SAFETY_MARGIN, formula="wis-col-katzir"),
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
REGISTRY["rw-weighted__restart__none"] = Entry(
    partial(random_walk_collision.build,
            dead_end="restart", thinning="none", formula="wis-col-katzir"),
    Category.REALIZABLE,
)

del _dead_end, _thinning, _cf


def register(name: str, factory: Callable[[], Estimator], category: Category) -> None:
    REGISTRY[name] = Entry(factory, category)


# Namen mit angehaengter Zahl werden zur Laufzeit aufgeloest:
#   "...__margin20"   -> Eintrag "...__margin"   mit margin=20
#   "...__schnabel8"  -> Eintrag "...__schnabel" mit n_captures=8
_NUMBERED = {"margin": "margin", "schnabel": "n_captures"}
_NUMBERED_RE = re.compile(r"^(?P<base>.+__(?P<kind>" + "|".join(_NUMBERED)
                          + r"))(?P<n>\d+)$")


def build(name: str) -> Estimator:
    entry, kwargs = REGISTRY.get(name), {}
    if entry is None:
        m = _NUMBERED_RE.match(name)
        if m:
            entry = REGISTRY.get(m.group("base"))
            kwargs = {_NUMBERED[m.group("kind")]: int(m.group("n"))}
    if entry is None:
        raise KeyError(
            f"{name!r} ist kein bekannter Estimator. Bekannt sind: "
            f"{', '.join(sorted(REGISTRY))} (dazu '...__margin<N>' fuer einen "
            "abweichenden Safety Margin)."
        )
    # partial-Keywords werden von Aufruf-Keywords ueberschrieben
    est = entry.factory(**kwargs)
    est.name = name
    est.category = entry.category  # Label erst hier, nach der Konstruktion
    return est


def names(category: Category | None = None) -> list[str]:
    return [n for n, e in REGISTRY.items() if category is None or e.category == category]


def build_all(selected: list[str] | None = None, category: Category | None = None) -> list[Estimator]:
    # REGISTRY.get statt [], weil dynamische Margin-Namen dort nicht stehen
    return [build(n) for n in (selected or names(category))
            if category is None or getattr(REGISTRY.get(n), "category", None) == category]
