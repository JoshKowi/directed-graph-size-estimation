"""DURW + Thinning + Collision Counting.

Das Gegenstueck zu estimators/methods/random_walk_collision.py: derselbe
Aufbau, nur laeuft statt des einfachen Random Walks der DURW aus
sampling.durw. Er baut sich waehrend des Laufs einen ungerichteten Graphen G_u
und springt mit Wahrscheinlichkeit w/(w + deg_Gu(v)) -- dadurch ist seine
Stationaerverteilung auch auf den *gerichteten* Views geschlossen bekannt,
was fuer den einfachen Random Walk gerade nicht gilt.

Die austauschbaren Achsen:

    jump        -- "uniform" (sampling.jumps); an die Stelle von `dead_end`
                   getreten. DURW braucht keine Sackgassen-Strategie: bei
                   deg_Gu = 0 ist die Sprungwahrscheinlichkeit 1.
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

import numpy as np

import config
from estimators.formulas import FORMULAS
from estimators.pipeline import PipelineEstimator
from oracles.local_access import JumpCrawlOracle
from sampling.durw import DurwSampler
from sampling.jumps import JUMPS
from sampling.thinning import THINNINGS
from weighting.schemes import DurwWeighting, UniformWeighting

# Jede Sprungart braucht ein Oracle, das sie bedienen kann. Eine spaeter
# hinzukommende Sprungart, die ihr Ziel aus externen Daten simuliert, traegt
# hier ihr eigenes Oracle ein -- am Sampler aendert das nichts.
JUMP_ORACLES: dict[str, type] = {
    "uniform": JumpCrawlOracle,
}


def build(
    jump: str = "uniform",
    thinning: str = "none",
    step: int = 5,
    margin: int = 0,
    formula: str = "uis-collision",
    jump_weight: float = config.DURW_JUMP_WEIGHT,
    n_seeds: int = 1,
    burn_in: int = 0,
    aggregate=np.median,
) -> PipelineEstimator:
    thin_cls = THINNINGS[thinning]
    thin = thin_cls() if thinning == "none" else thin_cls(step=step)
    weighting = (DurwWeighting(jump_weight) if FORMULAS[formula].weighted
                 else UniformWeighting())

    return PipelineEstimator(
        name=f"durw__{formula}__{jump}__{thinning}"
             + (f"__m{margin}" if margin else ""),
        oracle_cls=JUMP_ORACLES[jump],
        sampler=DurwSampler(jump=JUMPS[jump](), jump_weight=jump_weight,
                            n_seeds=n_seeds, burn_in=burn_in),
        weighting=weighting,
        formula=FORMULAS[formula](margin=margin),
        thinning=thin,
        aggregate=aggregate,
    )
