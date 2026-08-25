"""Experiment-Runner: View x Estimator x Budget x Wiederholung.

Ablauf: Graph einmal laden, daraus je gewaehlter View (directed / undirected /
reverse) eine Kantensicht bauen und darauf fuer jedes relative Budget b das
absolute Budget = round(b * |V|) bestimmen. Jeder Estimator laeuft n-mal.

Der Seed haengt bewusst *nicht* von der View ab: Lauf i benutzt in jeder View
denselben Zufallsstrom. Damit ist der Vergleich gepaart und Unterschiede gehen
auf die Kantensicht zurueck, nicht auf RNG-Rauschen.

Parallelisierung: die (Budget, Estimator, Lauf)-Tripel einer View sind
vollstaendig unabhaengig -- gleicher Graph, nur lesend, eigener Seed. Sie
laufen ueber einen Pool mit `fork`. Der Graph wird dabei *nicht* kopiert: er
liegt als CSR in drei numpy-Arrays (siehe graphs.graph), deren Datenpuffer
Copy-on-Write ueberleben, weil CPython sie nicht elementweise
referenzzaehlt. `gc.freeze()` vor dem Fork haelt zusaetzlich die
Garbage-Collection davon ab, die uebrigen Objekte anzufassen.

Die Ergebnisse sind vom Grad der Parallelisierung unabhaengig: der Seed haengt
nur an (Estimator, Budget, Lauf), nicht an der Ausfuehrungsreihenfolge. Die
Zeilen werden am Ende sortiert, damit auch die CSV reproduzierbar ist.

Schnittstelle:
    run_graph(graph, estimators, budgets, n_runs, seed, views, collect_visits,
              n_jobs, log) -> (results_df, visits_df | None)
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import random
import time
from collections import Counter

import pandas as pd

import config
import estimators as estimator_registry
from estimators.base import Estimator
from graphs.graph import Graph
from graphs.views import build_view

# Wird vor dem Fork gesetzt und von den Kindprozessen geerbt -- so muss der
# Graph nie durch einen Pickle-Kanal.
_VIEW: Graph | None = None
_COLLECT_VISITS = False


def _estimate_one(task):
    """Ein (Budget, Estimator, Lauf)-Tripel. Laeuft im Kindprozess."""
    b, budget, est_name, category, run, seed = task
    est = estimator_registry.build(est_name)
    rng = random.Random(f"{seed}|{est_name}|{b}|{run}")
    t0 = time.perf_counter()
    res = est.estimate(_VIEW, budget, rng)
    row = {
        "estimator": est_name,
        "category": category,
        "budget_rel": b,
        "budget_abs": budget,
        "run": run,
        "estimate": res.value,
        "queries_used": res.cost.get("queries"),
        "unique_nodes_used": res.cost.get("unique_nodes"),
        "cached_queries": res.cost.get("cached_queries"),
        "n_random_node": res.cost.get("n_random_node"),
        "n_neighbors": res.cost.get("n_neighbors"),
        "stopped_by": res.cost.get("stopped_by"),
        "seconds": time.perf_counter() - t0,
        **{f"extra_{k}": v for k, v in res.extra.items()},
    }
    return row, (res.visits if _COLLECT_VISITS else None)


def run_graph(
    graph: Graph,
    estimators: list[Estimator],
    budgets=config.DEFAULT_BUDGETS,
    n_runs: int = config.DEFAULT_N_RUNS,
    seed: int = config.DEFAULT_SEED,
    views=config.DEFAULT_VIEWS,
    collect_visits: bool = True,
    n_jobs: int = config.DEFAULT_N_JOBS,
    log=print,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    global _VIEW, _COLLECT_VISITS
    rows: list[dict] = []
    visit_rows: list[dict] = []
    t_start = time.perf_counter()

    for view_name in views:
        t0 = time.perf_counter()
        view = build_view(graph, view_name)
        true_size = view.n_nodes
        log(f"[{graph.name}/{view_name}] View gebaut in {time.perf_counter()-t0:.1f}s "
            f"-- |V|={true_size:,}, Kanten={view.n_edges:,}".replace(",", " "))

        tasks = [
            (b, max(int(round(b * true_size)), 2), est.name, str(est.category), run, seed)
            for b in budgets
            for est in estimators
            for run in range(n_runs)
        ]
        _VIEW, _COLLECT_VISITS = view, collect_visits
        visits: dict[tuple, Counter] = {}
        done = 0
        t_view = time.perf_counter()

        # Sammelt je (Budget, Estimator), damit erst geloggt wird, wenn eine
        # ganze Gruppe fertig ist -- sonst waeren es n_runs mal so viele Zeilen.
        pending: dict[tuple, list] = {}
        expected = len(estimators) * len(budgets)

        def handle(result):
            nonlocal done
            row, vis = result
            row.update(graph=graph.name, view=view_name, true_size=true_size,
                       rel_error=row["estimate"] / true_size - 1.0)
            rows.append(row)
            key = (row["budget_rel"], row["estimator"])
            pending.setdefault(key, []).append(row)
            if vis is not None:
                visits.setdefault(key, Counter()).update(vis)
            if len(pending[key]) == n_runs:
                done += 1
                grp = pending.pop(key)
                med = pd.Series([g["estimate"] for g in grp]).median()
                secs = sum(g["seconds"] for g in grp)
                steps = sum(g.get("extra_n_samples", 0) or 0 for g in grp)
                elapsed = time.perf_counter() - t_view
                eta = elapsed / done * (expected - done)
                log(f"  [{done:>3}/{expected}] {key[1]:<26} b={key[0]:<6g} "
                    f"est/|V|={med / true_size:8.4f}  Schritte={steps:>10,}  "
                    f"{secs:7.1f}s CPU  |  {elapsed/60:5.1f} min, Rest ~{eta/60:.0f} min"
                    .replace(",", " "))

        gc.freeze()          # bestehende Objekte aus der GC nehmen -> CoW bleibt heil
        if n_jobs > 1:
            with mp.get_context("fork").Pool(n_jobs) as pool:
                for result in pool.imap_unordered(_estimate_one, tasks, chunksize=1):
                    handle(result)
        else:
            for task in tasks:
                handle(_estimate_one(task))

        if collect_visits:
            for (b, est_name), counter in visits.items():
                visit_rows.extend(
                    {"graph": graph.name, "view": view_name, "estimator": est_name,
                     "budget_rel": b, "node": graph.name_of(node), "visits": count}
                    for node, count in counter.items()
                )

        _VIEW = None
        del view
        gc.collect()
        log(f"[{graph.name}/{view_name}] fertig in "
            f"{(time.perf_counter()-t_view)/60:.1f} min")

    log(f"[{graph.name}] gesamt {(time.perf_counter()-t_start)/60:.1f} min")
    results = pd.DataFrame(rows).sort_values(
        ["view", "budget_rel", "estimator", "run"]).reset_index(drop=True)
    visits_df = pd.DataFrame(visit_rows) if collect_visits else None
    return results, visits_df
