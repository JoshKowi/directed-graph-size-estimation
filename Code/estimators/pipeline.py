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

Geteilte Walks (`estimate_group`): Thinning, Weighting und Formel kommen alle
*nach* dem Sampler -- sie sind reine Nachbearbeitung einer Trajektorie und
aendern nichts am Walk. Estimators mit gleichem `walk_key` (Oracle + Sampler +
Budget-Metrik) koennen sich deshalb einen einzigen Walk teilen und nur die
Auswertung variieren. Das spart nicht nur Rechenzeit: der Vergleich zwischen
den Varianten wird dadurch *gepaart*, enthaelt also kein RNG-Rauschen mehr.

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
        .walk_key -> str
        .run_walk(graph, budget, rng, checkpoints=()) -> (trace, oracle)
        .evaluate(trace, cost, visits) -> EstimateResult
        .estimate(graph, budget, rng) -> EstimateResult
        .estimate_nested(graph, budgets, rng) -> dict[int, EstimateResult]
    estimate_group(estimators, graph, budgets, rng)
        -> dict[(name, budget), EstimateResult]
"""

from __future__ import annotations

import random
from functools import partial

import numpy as np

import config
from estimators.base import EstimateResult, Estimator
from estimators.formulas import SetsFormula
from graphs.graph import Graph
from sampling.thinning import NoThinning


def _oracle_key(oracle_cls) -> str:
    """Identitaet des Oracles, inklusive gebundener Parameter.

    `short_walk_independent.build` uebergibt das Oracle als
    `partial(ShortWalkIndependentOracle, steps=5)` -- ohne Aufloesen des
    partials faenden steps=5 und steps=7 faelschlich in derselben Walk-Gruppe
    zusammen und teilten sich einen Walk, den nur einer von beiden erzeugt
    haette.
    """
    if isinstance(oracle_cls, partial):
        args = ",".join(f"{k}={v}" for k, v in sorted(oracle_cls.keywords.items()))
        return f"{oracle_cls.func.__name__}({args})"
    return oracle_cls.__name__


class PipelineEstimator(Estimator):
    # Duerfen mehrere Budgets aus einem Lauf abgelesen werden? Nur falsch, wenn
    # die Ziehung selbst vom Budget abhaengt -- siehe
    # estimators/methods/capture_recapture.py.
    supports_nested = True

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

    @property
    def walk_key(self) -> str:
        """Zwei Estimators mit demselben Schluessel erzeugen dieselbe
        Trajektorie -- alles danach ist Nachbearbeitung (s. Modul-Docstring)."""
        return (f"{_oracle_key(self.oracle_cls)}|{self.sampler.key()}"
                f"|{self.budget_metric}")

    def run_walk(self, graph: Graph, budget: int, rng: random.Random,
                 checkpoints=()) -> tuple[list, object]:
        """Nur ziehen, nicht auswerten. Die Trennung macht die geteilten Walks
        moeglich und haelt estimate/estimate_nested/estimate_group auf
        derselben Mechanik."""
        oracle = self.oracle_cls(graph, rng, budget, self.budget_metric,
                                 checkpoints=tuple(checkpoints))
        trace = self.sampler.sample(oracle)
        oracle.finalize_checkpoints()
        return trace, oracle

    def estimate(self, graph: Graph, budget: int, rng: random.Random) -> EstimateResult:
        trace, oracle = self.run_walk(graph, budget, rng)
        return self.evaluate(trace, oracle.cost(), oracle.visits)

    def estimate_nested(self, graph: Graph, budgets, rng: random.Random
                        ) -> dict[int, EstimateResult]:
        """Alle Budgets aus einem einzigen Lauf -- Ergebnis je absolutem Budget.

        Die Besuchszaehler gibt es nur fuer das groesste Budget: sie sind
        kumulativ, ein Zwischenstand muesste den ganzen Counter kopieren. Fuer
        die kleineren Budgets steht deshalb `visits=None`.
        """
        budgets = sorted({int(b) for b in budgets})
        top = budgets[-1]
        trace, oracle = self.run_walk(graph, top, rng, checkpoints=budgets[:-1])

        out = {}
        for snap in oracle.snapshots:
            snap = dict(snap)
            b, k = snap.pop("budget_abs"), snap.pop("n_samples")
            out[b] = self.evaluate(trace[:k], snap, None)
        out[top] = self.evaluate(trace, oracle.cost(), oracle.visits)
        return out

    def evaluate(self, trace, cost: dict, visits) -> EstimateResult:
        """Aus einer (ggf. abgeschnittenen) Trajektorie eine Schaetzung machen."""
        subsets = self.thinning.apply(trace)
        weights = [self.weighting.weights(s) for s in subsets]

        if isinstance(self.formula, SetsFormula):
            # Formel ueber alle Sets gemeinsam (Capture-Recapture): hier gibt es
            # nichts zu aggregieren, die Sets sind Teile *einer* Schaetzung.
            values = np.array([self.formula.compute_sets(subsets, weights)],
                              dtype=float)
        else:
            values = np.array(
                [self.formula.compute(s, w) for s, w in zip(subsets, weights)],
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
                **self.formula.extras(subsets, weights),
                # Streuung der Einzelschaetzungen *innerhalb* eines Walks --
                # im Vergleich zur Streuung ueber die Laeufe zeigt sie, ob das
                # Thinning die Abhaengigkeit wirklich reduziert.
                "subset_std": float(valid.std(ddof=1)) if valid.size > 1 else float("nan"),
                "subset_spread": float(valid.max() - valid.min()) if valid.size > 1 else 0.0,
            },
        )


def estimate_group(estimators, graph: Graph, budgets, rng: random.Random
                   ) -> dict[tuple[str, int], EstimateResult]:
    """Ein Walk, alle Varianten -- Ergebnis je (Estimator-Name, Budget).

    Alle uebergebenen Estimators muessen denselben `walk_key` haben; gezogen
    wird genau einmal, mit dem ersten von ihnen. Danach bekommt jeder dieselbe
    Trajektorie (bzw. deren Praefix je Budget) durch sein eigenes Thinning,
    Weighting und seine eigene Formel geschickt.

    Die Besuchszaehler haengen am Walk, nicht an der Variante: sie stehen nur
    beim ersten Estimator und nur beim groessten Budget, sonst wuerde derselbe
    Counter mehrfach gezaehlt.
    """
    keys = {e.walk_key for e in estimators}
    if len(keys) != 1:
        raise ValueError(
            f"estimate_group braucht denselben Walk fuer alle Estimators, "
            f"bekam aber {sorted(keys)}"
        )

    budgets = sorted({int(b) for b in budgets})
    if len(budgets) > 1 and not all(getattr(e, "supports_nested", False)
                                    for e in estimators):
        raise ValueError(
            "Mehrere Budgets aus einem Lauf gehen nur, wenn die Ziehung nicht "
            "vom Budget abhaengt. Hier nicht der Fall (capture_recapture "
            "schaltet bei der Haelfte des Gesamtbudgets um) -- je Budget einzeln "
            "aufrufen."
        )
    top = budgets[-1]
    owner = estimators[0]
    trace, oracle = owner.run_walk(graph, top, rng, checkpoints=budgets[:-1])

    slices = [(dict(s), s["budget_abs"], s["n_samples"]) for s in oracle.snapshots]
    slices.append((oracle.cost(), top, len(trace)))

    out = {}
    for cost, budget, k in slices:
        cost = {c: v for c, v in cost.items()
                if c not in ("budget_abs", "n_samples")}
        part = trace[:k] if k < len(trace) else trace
        for i, est in enumerate(estimators):
            visits = oracle.visits if (i == 0 and budget == top) else None
            out[(est.name, budget)] = est.evaluate(part, cost, visits)
    return out
