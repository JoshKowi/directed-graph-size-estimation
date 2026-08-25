"""WIS Collision Counting auf gradgewichteten, aber *unabhaengigen* Samples.

Baut auf DegWeightedIndependentOracle auf: jedes Sample hat wie beim Random
Walk die Verteilung pi(v) ~ deg(v), aber die Samples sind unabhaengig
voneinander. Damit trennt dieser Estimator die beiden Effekte, die beim echten
Walk zusammenfallen:

    Gradverzerrung      -- hier vorhanden, vom WIS-Gewicht korrigiert
    Autokorrelation     -- hier nicht vorhanden

Der Vergleich mit dem echten Random Walk (methods/random_walk_collision.py)
zeigt also, was allein die Abhaengigkeit aufeinanderfolgender Samples kostet.
Das Ziehen setzt Kenntnis von V voraus und ist damit nicht real umsetzbar --
die Kategorie steht in der REGISTRY.

Schnittstelle:
    build(formula="wis-col-katzir") -> PipelineEstimator
"""

from __future__ import annotations

from estimators.formulas import FORMULAS
from estimators.pipeline import PipelineEstimator
from oracles.global_access import DegWeightedIndependentOracle
from sampling.samplers import UniformSampler
from weighting.schemes import InverseDegreeWeighting, UniformWeighting


def build(formula: str = "wis-col-katzir") -> PipelineEstimator:
    # UniformSampler heisst nach seinem Zugriffsmuster (unabhaengige Ziehungen
    # ueber oracle.random_node()), nicht nach der Verteilung -- die legt hier
    # das Oracle auf pi(v) ~ deg(v) fest.
    weighting = (InverseDegreeWeighting() if FORMULAS[formula].weighted
                 else UniformWeighting())

    return PipelineEstimator(
        name=f"dwi_{formula}",
        oracle_cls=DegWeightedIndependentOracle,
        sampler=UniformSampler(),
        weighting=weighting,
        formula=FORMULAS[formula](),
    )
