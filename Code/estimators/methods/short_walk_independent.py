"""Collision Counting auf Endknoten kurzer, unabhaengiger Random Walks.

Baut auf ShortWalkIndependentOracle auf: jedes Sample ist der Endknoten eines
eigenen `steps`-Schritt-Walks von einem gleichverteilten Startknoten aus. Damit
ist die Abhaengigkeit zwischen aufeinanderfolgenden Samples weg, die
Walk-typische Verzerrung aber da.

Der Punkt: mit `formula="wis-col-katzir"` wird mit w_i = 1/deg(v) korrigiert.
Auf ungerichteten Graphen ist das (naeherungsweise) richtig, weil die
Walk-Verteilung gegen pi(v) ~ deg(v) laeuft. Auf gerichteten Graphen ist es
falsch -- und weil hier keine Autokorrelation mehr im Spiel ist, laesst sich
der Effekt der Gewichtung isoliert ablesen. Der Vergleich mit
formula="uis-collision" (ohne Gewichte) auf denselben Samples zeigt, was die
Korrektur ueberhaupt bewirkt.

Setzt gleichverteiltes Ziehen des Startknotens voraus und ist damit nicht real
umsetzbar -- die Kategorie steht in der REGISTRY.

Schnittstelle:
    build(formula="wis-col-katzir", steps=5) -> PipelineEstimator
"""

from __future__ import annotations

from functools import partial

from estimators.formulas import FORMULAS
from estimators.pipeline import PipelineEstimator
from oracles.global_access import ShortWalkIndependentOracle
from sampling.samplers import UniformSampler
from weighting.schemes import InverseDegreeWeighting, UniformWeighting


def build(formula: str = "wis-col-katzir", steps: int = 5) -> PipelineEstimator:
    weighting = (InverseDegreeWeighting() if FORMULAS[formula].weighted
                 else UniformWeighting())

    return PipelineEstimator(
        name=f"walk{steps}_{formula}",
        # PipelineEstimator ruft oracle_cls(graph, rng, budget, metric) auf --
        # die Walk-Laenge kommt ueber partial dazu.
        oracle_cls=partial(ShortWalkIndependentOracle, steps=steps),
        sampler=UniformSampler(),
        weighting=weighting,
        formula=FORMULAS[formula](),
    )
