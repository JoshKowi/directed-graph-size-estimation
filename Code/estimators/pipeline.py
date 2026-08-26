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

Genestete Budgets (`estimate_nested`): statt je Budget einen eigenen Lauf zu
rechnen, laeuft *ein* Lauf mit dem groessten Budget und haelt unterwegs fest,
wo die kleineren geendet haetten. Die Stichprobe wird dort abgeschnitten und
ganz normal durch Thinning, Weighting und Formel geschickt.

Das ist exakt, nicht genaehert: kein Sampler kennt sein Budget, es steuert nur
den Abbruch. Der abgeschnittene Lauf ist deshalb bitgleich mit einem
eigenstaendigen Lauf desselben Zufallsstroms bei diesem Budget (siehe
oracles/base.py). Was sich aendert, ist nicht die Verteilung je Budget, sondern
die *Abhaengigkeit zwischen* den Budgets: die Punkte einer Laufnummer sind
danach genestet, nicht unabhaengig. Deshalb steht das in der Ergebnis-CSV
(Spalte `nested`).

Schnittstelle:
    class PipelineEstimator(Estimator)
        .estimate(graph, budget, rng) -> EstimateResult
        .estimate_nested(graph, budgets, rng) -> dict[int, EstimateResult]
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
        return self._evaluate(trace, oracle.cost(), oracle.visits)

    def estimate_nested(self, graph: Graph, budgets, rng: random.Random
                        ) -> dict[int, EstimateResult]:
        """Alle Budgets aus einem einzigen Lauf -- Ergebnis je absolutem Budget.

        Die Besuchszaehler gibt es nur fuer das groesste Budget: sie sind
        kumulativ, ein Zwischenstand muesste den ganzen Counter kopieren. Fuer
        die kleineren Budgets steht deshalb `visits=None`.
        """
        budgets = sorted({int(b) for b in budgets})
        top = budgets[-1]
        oracle = self.oracle_cls(graph, rng, top, self.budget_metric,
                                 checkpoints=budgets[:-1])
        trace = self.sampler.sample(oracle)
        oracle.finalize_checkpoints()

        out = {}
        for snap in oracle.snapshots:
            snap = dict(snap)
            b, k = snap.pop("budget_abs"), snap.pop("n_samples")
            out[b] = self._evaluate(trace[:k], snap, None)
        out[top] = self._evaluate(trace, oracle.cost(), oracle.visits)
        return out

    def _evaluate(self, trace, cost: dict, visits) -> EstimateResult:
        """Aus einer (ggf. abgeschnittenen) Trajektorie eine Schaetzung machen."""
        subsets = self.thinning.apply(trace)
        values = np.array(
            [self.formula.compute(s, self.weighting.weights(s)) for s in subsets],
            dtype=float,
        )
        valid = values[~np.isnan(values)]
        value = float(self.aggregate(valid)) if valid.size else float("nan")

        return EstimateResult(
            value=value,
            cost=cost,
            visits=visits,
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
