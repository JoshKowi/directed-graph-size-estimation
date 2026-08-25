"""Collision Counting auf gleichverteilten Knoten.

Setzt mit UniformNodeOracle gleichverteiltes Ziehen aus V voraus und ist damit
bei uns nicht real umsetzbar -- die Kategorie steht in der REGISTRY. Dient als
Referenz dafuer, was mit einem gegebenen Budget ueberhaupt erreichbar waere.

Schnittstelle:
    build(formula="uis-collision") -> PipelineEstimator
"""

from __future__ import annotations

from estimators.formulas import FORMULAS
from estimators.pipeline import PipelineEstimator
from oracles.global_access import UniformNodeOracle
from sampling.samplers import UniformSampler
from weighting.schemes import UniformWeighting


def build(formula: str = "uis-collision") -> PipelineEstimator:
    return PipelineEstimator(
        name="uniform_collision",
        oracle_cls=UniformNodeOracle,
        sampler=UniformSampler(),
        weighting=UniformWeighting(),
        formula=FORMULAS[formula](),
    )
