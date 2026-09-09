"""Unabhaengige Ziehung aus einer externen Namensliste + Collision Counting.

Das real umsetzbare Gegenstueck zu estimators/methods/uniform_collision.py:
dort wird gleichverteilt aus V gezogen (setzt Kenntnis der Knotenmenge voraus),
hier aus einer Liste plausibler Namen (oracles.name_list, namelists). Dieselbe
Auswertung, anderer Zugriff -- die Differenz der beiden Reihen ist genau der
Preis dafuer, V nicht zu kennen.

    source        -- "enwiki" | "dewiki" | "top-q"   (namelists.SOURCES)
    draw_burn_in  -- Schritte nach jedem Treffer, bevor der Knoten zaehlt
    cost_miss     -- Preis eines Namens, den der Graph nicht kennt
    formula       -- "uis-collision" | "wis-col-katzir"

Zur Gewichtung: "uis-collision" unterstellt gleichverteilte Ziehungen. Das ist
hier **nicht** erfuellt -- gezogen wird gleichverteilt auf der Trefferteilmenge,
und die ist gradverzerrt (Details in oracles/name_list.py). Der Schaetzer misst
also die Groesse dieser Teilmenge, nicht |V|; genau das soll er sichtbar machen.
"wis-col-katzir" mit InverseDegreeWeighting korrigiert eine *Gradverzerrung*,
nicht die Auswahl der Liste, und ist deshalb keine Reparatur, sondern eine
zweite Messung derselben Stichprobe.

Schnittstelle:
    build(source, draw_burn_in, cost_miss, formula, ...) -> PipelineEstimator
"""

from __future__ import annotations

from functools import partial

import numpy as np

import config
from estimators.formulas import FORMULAS
from estimators.pipeline import PipelineEstimator
from oracles.name_list import NameListOracle
from sampling.samplers import UniformSampler
from weighting.schemes import InverseDegreeWeighting, UniformWeighting


def build(
    source: str = "enwiki",
    draw_burn_in: int = config.DEFAULT_DRAW_BURN_IN,
    draw_limit: int | None = None,
    cost_miss: float = config.COST_DRAW_MISS,
    formula: str = "uis-collision",
    margin: int = 0,
    aggregate=np.median,
) -> PipelineEstimator:
    weighting = (InverseDegreeWeighting() if FORMULAS[formula].weighted
                 else UniformWeighting())
    return PipelineEstimator(
        name=f"namelist-{source}"
             + (f"__n{draw_limit}" if draw_limit else "")
             + f"__b{draw_burn_in}__{formula}"
             + (f"__m{margin}" if margin else ""),
        # PipelineEstimator ruft oracle_cls(graph, rng, budget, metric) auf --
        # Quelle, Burn-in und Fehlschlagpreis kommen ueber partial dazu, und
        # pipeline._oracle_key loest das fuer den Walk-Schluessel wieder auf.
        oracle_cls=partial(NameListOracle, source=source, limit=draw_limit,
                           burn_in=draw_burn_in, cost_miss=cost_miss),
        sampler=UniformSampler(with_degree=weighting.needs_degree),
        weighting=weighting,
        formula=FORMULAS[formula](margin=margin),
        aggregate=aggregate,
    )
