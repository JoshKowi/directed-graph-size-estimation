"""Zusammengesetzter Estimator: Oracle + Sampler + Thinning + Weighting +
Formula + Aggregation.

Deckt den Regelfall ab. Estimators, die alles in einem Schritt machen, erben
stattdessen direkt von Estimator (siehe estimators/methods/capture_recapture.py).
Die Kategorie wird hier bewusst nicht gesetzt -- sie haengt am Oracle und wird
in der REGISTRY vergeben.

Ablauf: der Sampler liefert die volle Trajektorie, das Thinning macht daraus
ein oder mehrere Sample-Sets, je Set wird gewichtet und geschaetzt, und die
Einzelschaetzungen werden aggregiert (Default: Median). NaN-Schaetzungen --
Sets ohne beobachtete Kollision -- werden vorher aussortiert.

Schnittstelle:
    class PipelineEstimator(Estimator)
"""

from __future__ import annotations

import random

import numpy as np

import config
from estimators.base import EstimateResult, Estimator
from graphs.graph import Graph
from sampling.thinning import NoThinning


class PipelineEstimator(Estimator):
    def __init__(
        self,
        name: str,
        oracle_cls,
        sampler,
        weighting,
        formula,
        thinning=None,
        aggregate=np.median,
        budget_metric: str = config.DEFAULT_BUDGET_METRIC,
    ) -> None:
        self.name = name
        self.oracle_cls = oracle_cls
        self.sampler = sampler
        self.weighting = weighting
        self.formula = formula
        self.thinning = thinning or NoThinning()
        self.aggregate = aggregate
        self.budget_metric = budget_metric

    def estimate(self, graph: Graph, budget: int, rng: random.Random) -> EstimateResult:
        oracle = self.oracle_cls(graph, rng, budget, self.budget_metric)
        trace = self.sampler.sample(oracle)

        subsets = self.thinning.apply(trace)
        values = np.array(
            [self.formula.compute(s, self.weighting.weights(s)) for s in subsets],
            dtype=float,
        )
        valid = values[~np.isnan(values)]
        value = float(self.aggregate(valid)) if valid.size else float("nan")

        return EstimateResult(
            value=value,
            cost=oracle.cost(),
            visits=oracle.visits,
            extra={
                "n_samples": len(trace),
                "n_subsets": len(subsets),
                "n_valid": int(valid.size),
                # Streuung der Einzelschaetzungen *innerhalb* eines Walks --
                # im Vergleich zur Streuung ueber die Laeufe zeigt sie, ob das
                # Thinning die Abhaengigkeit wirklich reduziert.
                "subset_std": float(valid.std(ddof=1)) if valid.size > 1 else float("nan"),
                "subset_spread": float(valid.max() - valid.min()) if valid.size > 1 else 0.0,
            },
        )
