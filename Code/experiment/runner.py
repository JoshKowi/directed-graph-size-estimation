"""Experiment-Runner: View x Estimator x Budget x Wiederholung.

Ablauf: Graph einmal laden, daraus je gewaehlter View (directed / undirected /
reverse) eine Kantensicht bauen und darauf fuer jedes relative Budget b das
absolute Budget = round(b * |V|) bestimmen. Jeder Estimator laeuft n-mal.

Der Seed haengt bewusst *nicht* von der View ab: Lauf i benutzt in jeder View
denselben Zufallsstrom. Damit ist der Vergleich gepaart und Unterschiede gehen
auf die Kantensicht zurueck, nicht auf RNG-Rauschen.

Schnittstelle:
    run_graph(graph, estimators, budgets, n_runs, seed, views, collect_visits)
        -> (results_df, visits_df | None)
"""

from __future__ import annotations

import gc
import random
import time
from collections import Counter

import pandas as pd

import config
from estimators.base import Estimator
from graphs.graph import Graph
from graphs.views import build_view


def run_graph(
    graph: Graph,
    estimators: list[Estimator],
    budgets=config.DEFAULT_BUDGETS,
    n_runs: int = config.DEFAULT_N_RUNS,
    seed: int = config.DEFAULT_SEED,
    views=config.DEFAULT_VIEWS,
    collect_visits: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    rows: list[dict] = []
    visit_rows: list[dict] = []

    for view_name in views:
        view = build_view(graph, view_name)
        true_size = view.n_nodes

        for b in budgets:
            budget = max(int(round(b * true_size)), 2)
            for est in estimators:
                visits: Counter = Counter()
                for run in range(n_runs):
                    # ohne view_name -> gepaarte Laeufe ueber alle Views
                    rng = random.Random(f"{seed}|{est.name}|{b}|{run}")
                    t0 = time.perf_counter()
                    res = est.estimate(view, budget, rng)
                    rows.append(
                        {
                            "graph": graph.name,
                            "view": view_name,
                            "estimator": est.name,
                            "category": str(est.category),
                            "budget_rel": b,
                            "budget_abs": budget,
                            "run": run,
                            "estimate": res.value,
                            "true_size": true_size,
                            "rel_error": res.value / true_size - 1.0,
                            "queries_used": res.cost.get("queries"),
                            "unique_nodes_used": res.cost.get("unique_nodes"),
                            "cached_queries": res.cost.get("cached_queries"),
                            "n_random_node": res.cost.get("n_random_node"),
                            "n_neighbors": res.cost.get("n_neighbors"),
                            "stopped_by": res.cost.get("stopped_by"),
                            "seconds": time.perf_counter() - t0,
                            **{f"extra_{k}": v for k, v in res.extra.items()},
                        }
                    )
                    if collect_visits:
                        visits.update(res.visits)

                if collect_visits:
                    visit_rows.extend(
                        {
                            "graph": graph.name,
                            "view": view_name,
                            "estimator": est.name,
                            "budget_rel": b,
                            "node": node,
                            "visits": count,
                        }
                        for node, count in visits.items()
                    )

        del view
        gc.collect()

    results = pd.DataFrame(rows)
    visits_df = pd.DataFrame(visit_rows) if collect_visits else None
    return results, visits_df
