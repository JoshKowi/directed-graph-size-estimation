"""DURW + Thinning + Collision Counting.

Das Gegenstueck zu estimators/methods/random_walk_collision.py: derselbe
Aufbau, nur laeuft statt des einfachen Random Walks der DURW aus
sampling.durw. Er baut sich waehrend des Laufs einen ungerichteten Graphen G_u
und springt mit Wahrscheinlichkeit w/(w + deg_Gu(v)) -- dadurch ist seine
Stationaerverteilung auch auf den *gerichteten* Views geschlossen bekannt,
was fuer den einfachen Random Walk gerade nicht gilt.

Die austauschbaren Achsen:

    jump        -- "uniform" (sampling.jumps); an die Stelle von `dead_end`
                   getreten. DURW braucht keine Sackgassen-Strategie, weil eine
                   Sackgasse nie absorbierend wird: ueber eine Kante erreicht,
                   traegt sie diese Kante als Rueckweg in ihrem G_u-Grad; per
                   Sprung erreicht, hat sie deg_Gu = 0 und springt zwingend
                   weiter (w/(w+0) = 1).
    jump_weight -- w der Sprungregel (config.DURW_JUMP_WEIGHT)
    thinning    -- "none" | "simple" | "shifted"    (sampling.thinning)
    margin      -- Safety Margin wie dort (estimators.formulas). 0 = aus.
    formula     -- "uis-collision" | "wis-col-katzir"

"wis-col-katzir" gehoert mit DurwWeighting zusammen: der Walk zieht
proportional zu (w + deg_Gu), das Gewicht korrigiert genau das. Nicht mit
InverseDegreeWeighting verwechseln -- die passt zum einfachen Random Walk,
nicht zu DURW. "uis-collision" ignoriert die Gewichte und unterstellt
gleichverteilte Ziehungen; die Differenz ist der Preis der Verzerrung, also
das, was DURW ueberhaupt korrigiert.

Welches Oracle ein Lauf braucht, haengt an der Sprungart -- JUMP_ORACLES haelt
die Zuordnung. Die Kategorie (real umsetzbar oder nicht) folgt daraus, wird
aber wie im Repo ueblich erst in estimators/__init__.py vergeben.

Schnittstelle:
    JUMP_ORACLES: dict[str, type]
    build(jump, thinning, step, margin, formula, jump_weight, ...)
        -> PipelineEstimator
"""

from __future__ import annotations

from functools import partial

import numpy as np

import config
import namelists
from estimators.formulas import FORMULAS
from estimators.pipeline import PipelineEstimator
from oracles.local_access import JumpCrawlOracle
from oracles.name_list import NameListOracle
from sampling.durw import DurwSampler
from sampling.jumps import JUMPS
from sampling.thinning import THINNINGS
from weighting.schemes import DurwWeighting, UniformWeighting

# Jede Sprungart braucht ein Oracle, das sie bedienen kann. Eine spaeter
# hinzukommende Sprungart, die ihr Ziel aus externen Daten simuliert, traegt
# hier ihr eigenes Oracle ein -- am Sampler aendert das nichts.
JUMP_ORACLES: dict[str, object] = {
    "uniform": JumpCrawlOracle,
}
# Die Listenquellen ziehen ueber NameListOracle. Sie brauchen keine Kenntnis
# von V -- deshalb sind sie in estimators/__init__.py REALIZABLE, waehrend
# "uniform" COMPARISON bleibt.
for _src in sorted(namelists.SOURCES):
    JUMP_ORACLES[_src] = partial(NameListOracle, source=_src)
del _src


def build(
    jump: str = "uniform",
    thinning: str = "none",
    step: int = 5,
    margin: int = 0,
    formula: str = "uis-collision",
    jump_weight: float = config.DURW_JUMP_WEIGHT,
    n_seeds: int = 1,
    burn_in: int = 0,
    draw_burn_in: int = config.DEFAULT_DRAW_BURN_IN,
    cost_miss: float = config.COST_DRAW_MISS,
    aggregate=np.median,
) -> PipelineEstimator:
    # `burn_in` verwirft die ersten Schritte des Walks, `draw_burn_in` die
    # ersten Schritte *nach jedem Sprung* -- zwei verschiedene Dinge, die nicht
    # verwechselt werden duerfen. Letzteres kennt nur das NameListOracle.
    oracle_cls = JUMP_ORACLES[jump]
    if jump != "uniform":
        oracle_cls = partial(oracle_cls, burn_in=draw_burn_in, cost_miss=cost_miss)
    thin_cls = THINNINGS[thinning]
    thin = thin_cls() if thinning == "none" else thin_cls(step=step)
    weighting = (DurwWeighting(jump_weight) if FORMULAS[formula].weighted
                 else UniformWeighting())

    return PipelineEstimator(
        # w nur dann im Namen, wenn es vom Default abweicht -- sonst hiessen
        # die Registry-Eintraege ohne w-Angabe anders als bisher. Fuer Laeufe
        # ueber die Registry ist der Name ohnehin kosmetisch (estimators.build()
        # ueberschreibt ihn), fuer Direktaufrufe aus einem Notebook nicht.
        name=f"durw__{formula}__{jump}__{thinning}"
             + (f"__w{jump_weight:g}" if jump_weight != config.DURW_JUMP_WEIGHT else "")
             + (f"__m{margin}" if margin else ""),
        oracle_cls=oracle_cls,
        sampler=DurwSampler(jump=JUMPS[jump](), jump_weight=jump_weight,
                            n_seeds=n_seeds, burn_in=burn_in),
        weighting=weighting,
        formula=FORMULAS[formula](margin=margin),
        thinning=thin,
        aggregate=aggregate,
    )
