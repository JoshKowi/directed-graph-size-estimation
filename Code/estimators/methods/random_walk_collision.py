"""Random Walk + Thinning + Collision Counting.

Braucht mit CrawlOracle nur Nachbarschaftsabfragen ab einem Seed. Die drei
austauschbaren Achsen:

    dead_end -- "restart" | "backtrack" | "history"   (sampling.dead_ends)
    thinning -- "none" | "simple" | "shifted"         (sampling.thinning)
    margin   -- Safety Margin: Mindestabstand im Walk, ab dem ein Paar als
                Kollision zaehlt (estimators.formulas). 0 = aus.
    formula    -- "uis-collision" (C(k,2)/n_col) | "wis-col-katzir"
                  (beide gradkorrigiert)

"wis-col-katzir" gehoert mit InverseDegreeWeighting
zusammen: der Walk zieht proportional zum Grad, das Gewicht korrigiert das.
"uis-collision" ignoriert die Gewichte und unterstellt gleichverteilte
Ziehungen -- die Differenz ist gerade der Preis der Gradverzerrung.

Thinning und Margin sind zwei Antworten auf dasselbe Problem (Autokorrelation)
und gehoeren nicht kombiniert: bei `thinning="simple"` mit Schritt s liegen
benachbarte Samples des Sets bereits s Schritte auseinander, ein Margin m
verlangt dann s*m -- fast immer zu viel. Die Registry setzt den Margin
deshalb nur zusammen mit `thinning="none"`.

Schnittstelle:
    build(dead_end, thinning, step, formula, ...) -> PipelineEstimator
"""

from __future__ import annotations

import numpy as np

import config
from estimators.formulas import FORMULAS
from estimators.pipeline import PipelineEstimator
from oracles.local_access import CrawlOracle
from sampling.dead_ends import DEAD_ENDS
from sampling.samplers import RandomWalkSampler
from sampling.thinning import THINNINGS
from weighting.schemes import InverseDegreeWeighting, UniformWeighting


def build(
    dead_end: str = "restart",
    thinning: str = "none",
    step: int = 5,
    margin: int = 0,
    formula: str = "uis-collision",
    n_seeds: int = 1,
    burn_in: int = 0,
    aggregate=np.median,
) -> PipelineEstimator:
    thin_cls = THINNINGS[thinning]
    thin = thin_cls() if thinning == "none" else thin_cls(step=step)
    weighting = (InverseDegreeWeighting() if FORMULAS[formula].weighted
                 else UniformWeighting())

    return PipelineEstimator(
        name=f"rw_{formula}__{dead_end}__{thinning}"
             + (f"__m{margin}" if margin else ""),
        oracle_cls=CrawlOracle,
        sampler=RandomWalkSampler(dead_end=DEAD_ENDS[dead_end](), n_seeds=n_seeds,
                                  burn_in=burn_in),
        weighting=weighting,
        formula=FORMULAS[formula](margin=margin),
        thinning=thin,
        aggregate=aggregate,
    )
