"""Capture-Recapture (Lincoln-Petersen) ueber zwei unabhaengige Random Walks.

    n_hat = |S1| * |S2| / |S1 geschnitten S2|

mit |S1|, |S2| als Anzahl *verschiedener* besuchter Knoten je Walk. Ohne
Ueberschneidung ist keine Schaetzung moeglich -> NaN.

Estimator, der nicht in Oracle/Weighting/Formula zerfaellt und deshalb direkt von
Estimator erbt. Das Budget wird haelftig auf die beiden Walks aufgeteilt.

Die beiden Crawler teilen sich einen Cache: es ist derselbe Client, der zweimal
losgeschickt wird, und was er beim ersten Walk geholt hat, muss er beim zweiten
nicht erneut anfragen. Der zweite Walk kommt mit seiner Budget-Haelfte dadurch
deutlich weiter, sobald er bekanntes Gebiet betritt. Die Schaetzung selbst ist
davon unberuehrt -- gezaehlt werden Knoten, nicht Anfragen.

Schnittstelle:
    class CaptureRecaptureEstimator(Estimator)
    build(dead_end="restart", ...) -> CaptureRecaptureEstimator
"""

from __future__ import annotations

import random
from collections import Counter

from estimators.base import EstimateResult, Estimator
from graphs.graph import Graph
from oracles.local_access import CrawlOracle
from sampling.dead_ends import DEAD_ENDS
from sampling.samplers import RandomWalkSampler


class CaptureRecaptureEstimator(Estimator):
    name = "capture_recapture"

    def __init__(self, sampler: RandomWalkSampler, budget_metric: str = "queries") -> None:
        self.sampler = sampler
        self.budget_metric = budget_metric

    def estimate(self, graph: Graph, budget: int, rng: random.Random) -> EstimateResult:
        half = max(budget // 2, 1)

        cache: set = set()  # gemeinsam: derselbe Crawler, zwei Laeufe
        oracles = [CrawlOracle(graph, rng, half, self.budget_metric, cache=cache)
                   for _ in range(2)]
        sets = [{s.node for s in self.sampler.sample(o)} for o in oracles]

        overlap = len(sets[0] & sets[1])
        value = float("nan") if overlap == 0 else len(sets[0]) * len(sets[1]) / overlap

        visits: Counter = Counter()
        for o in oracles:
            visits.update(o.visits)
        cost = {k: sum(o.cost()[k] for o in oracles) for k in oracles[0].cost()}
        cost["unique_nodes"] = len(visits)  # ueber beide Walks zusammen

        return EstimateResult(
            value=value,
            cost=cost,
            visits=visits,
            extra={"n_unique_s1": len(sets[0]), "n_unique_s2": len(sets[1]),
                   "overlap": overlap},
        )


def build(dead_end: str = "restart", n_seeds: int = 1, burn_in: int = 0
          ) -> CaptureRecaptureEstimator:
    sampler = RandomWalkSampler(dead_end=DEAD_ENDS[dead_end](), n_seeds=n_seeds,
                                burn_in=burn_in)
    return CaptureRecaptureEstimator(sampler)
