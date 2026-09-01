"""Experiment-Runner: View x Estimator x Budget x Wiederholung.

Ablauf: Graph einmal laden, daraus je gewaehlter View (directed / undirected /
reverse) eine Kantensicht bauen und darauf fuer jedes relative Budget b das
absolute Budget = round(b * |V|) bestimmen. Jeder Estimator laeuft n-mal.

Der Seed haengt bewusst *nicht* von der View ab: Lauf i benutzt in jeder View
denselben Zufallsstrom. Damit ist der Vergleich gepaart und Unterschiede gehen
auf die Kantensicht zurueck, nicht auf RNG-Rauschen.

Der uebergebene `seed` ist der Startpunkt aller dieser Stroeme. Mit einem
anderen Seed bekommt man einen komplett anderen, gleichberechtigten Durchlauf
desselben Experiments -- deshalb steht er in jeder Ergebniszeile.

`nested_budgets=True` rechnet je (Estimator, Lauf) nur noch *einen* Lauf mit
dem groessten Budget und liest die kleineren unterwegs ab (siehe
estimators/pipeline.py). Das spart Sigma(Budgets)/max(Budget) an Rechenzeit --
bei einer geometrischen Leiter also weniger als Faktor 2, weil das groesste
Budget die Summe dominiert. Zwei Dinge aendern sich dadurch:

  * Der abgeleitete Strom haengt dann nicht mehr am Budget (es gibt nur noch
    einen Lauf), sondern nur an (Seed, Estimator, Lauf).
  * Die Punkte einer Laufnummer sind ueber die Budgets *genestet*, nicht mehr
    unabhaengig: ein Walk, der bei 1 % feststeckt, steckt bei 10 % immer noch
    fest. Je Budget bleibt die Verteilung dieselbe -- die Zahl unabhaengiger
    Trajektorien im Experiment sinkt aber von Laeufe x Budgets auf Laeufe.
    Deshalb die Spalte `nested` in der Ergebnis-CSV.

Estimators ohne `estimate_nested` (capture_recapture teilt sein Budget vorab
auf zwei Walks auf, ein Praefix hat dort eine andere Struktur) laufen auch in
diesem Modus weiter je Budget einzeln.

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
              n_jobs, nested_budgets, start_nodes, log)
        -> (results_df, visits_df | None)
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


def _n(x) -> str:
    """Zahl mit schmalem Leerzeichen als Tausendertrenner."""
    return f"{int(x):,}".replace(",", "\u2009")

# Wird vor dem Fork gesetzt und von den Kindprozessen geerbt -- so muss der
# Graph nie durch einen Pickle-Kanal.
_VIEW: Graph | None = None
_COLLECT_VISITS = False


def _row(est_name, category, seed, start, b, budget, run, res, seconds, nested):
    return {
        "estimator": est_name,
        "seed": seed,
        "start_node": start,
        "category": category,
        "budget_rel": b,
        "budget_abs": budget,
        "run": run,
        "nested": nested,
        "estimate": res.value,
        "queries_used": res.cost.get("queries"),
        "unique_nodes_used": res.cost.get("unique_nodes"),
        "cached_queries": res.cost.get("cached_queries"),
        "n_random_node": res.cost.get("n_random_node"),
        "n_neighbors": res.cost.get("n_neighbors"),
        "stopped_by": res.cost.get("stopped_by"),
        "seconds": seconds,
        **{f"extra_{k}": v for k, v in res.extra.items()},
    }


def _estimate_one(task):
    """Ein (Budget(s), Estimator, Lauf)-Paket. Laeuft im Kindprozess.

    `budgets` ist entweder ein einzelnes (rel, abs)-Paar oder -- im genesteten
    Modus -- die ganze Leiter, die aus einem Lauf abgelesen wird.
    """
    budgets, est_name, category, run, seed, start = task
    est = estimator_registry.build(est_name)
    nested = len(budgets) > 1
    t0 = time.perf_counter()

    if nested:
        # Ein Strom je (Estimator, Lauf) -- das Budget kann hier nicht mehr
        # eingehen, es gibt nur noch einen Lauf fuer alle Budgets.
        rng = random.Random(f"{seed}|{est_name}|{run}")
        results = est.estimate_nested(_VIEW, [a for _, a in budgets], rng)
        seconds = time.perf_counter() - t0
        out = []
        for b, budget in budgets:
            res = results[budget]
            # Besuchszaehler gibt es nur fuer das groesste Budget (s. pipeline).
            vis = res.visits if (_COLLECT_VISITS and res.visits is not None) else None
            out.append((_row(est_name, category, seed, start, b, budget, run, res,
                             seconds if budget == budgets[-1][1] else 0.0, True), vis))
        return out

    b, budget = budgets[0]
    rng = random.Random(f"{seed}|{est_name}|{b}|{run}")
    res = est.estimate(_VIEW, budget, rng)
    return [(_row(est_name, category, seed, start, b, budget, run, res,
                  time.perf_counter() - t0, False),
             res.visits if _COLLECT_VISITS else None)]


def run_graph(
    graph: Graph,
    estimators: list[Estimator],
    budgets=config.DEFAULT_BUDGETS,
    n_runs: int = config.DEFAULT_N_RUNS,
    seed: int = config.DEFAULT_SEED,
    views=config.DEFAULT_VIEWS,
    collect_visits: bool = True,
    n_jobs: int = config.DEFAULT_N_JOBS,
    nested_budgets: bool = False,
    start_nodes=None,
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
        # Feste Einstiegsknoten *vor* dem Fork aufloesen: der Durchlauf durch
        # names ist bei 18 Mio. Knoten sonst in jedem Kindprozess erneut faellig
        # (siehe graphs.graph.seed_ids).
        view.seed_ids()
        log(f"[{graph.name}/{view_name}] View gebaut in {time.perf_counter()-t0:.1f}s "
            f"-- |V|={_n(true_size)}, Kanten={_n(view.n_edges)}")

        # (rel, abs) je Budget. Zwei relative Budgets koennen auf einem kleinen
        # Graphen auf dieselbe absolute Zahl fallen -- genestet waere das eine
        # Kollision im Ergebnis-dict, deshalb aufsteigend und eindeutig.
        ladder = sorted({max(int(round(b * true_size)), 2): b
                         for b in budgets}.items(), key=lambda p: p[0])
        ladder = [(rel, abs_) for abs_, rel in ladder]

        # Je Einstiegsknoten ein eigener Durchlauf: der Einstieg ist eine
        # eigene Bedingung, keine Zufallsquelle (siehe README, Punkt 5).
        t_all = time.perf_counter()
        starts = list(start_nodes) if start_nodes else [None]
        for start in starts:
            if start is not None:
                # vor dem Fork -- die Kindprozesse erben die Einschraenkung
                view.restrict_seeds([start])
            where = f" start={start}" if len(starts) > 1 else ""
            tasks = []
            for est in estimators:
                # capture_recapture teilt sein Budget vorab auf zwei Walks auf --
                # ein Praefix hat dort eine andere Struktur, der Estimator bringt
                # deshalb kein estimate_nested mit und laeuft weiter je Budget.
                nest = nested_budgets and hasattr(est, "estimate_nested")
                groups = [ladder] if nest else [[pair] for pair in ladder]
                tasks += [(g, est.name, str(est.category), run, seed, start)
                          for g in groups
                          for run in range(n_runs)]
            if nested_budgets:
                single = sorted({e.name for e in estimators
                                 if not hasattr(e, "estimate_nested")})
                saving = sum(a for _, a in ladder) / ladder[-1][1]
                log(f"[{graph.name}/{view_name}] genestete Budgets: {len(tasks)} statt "
                    f"{len(estimators) * len(budgets) * n_runs} Laeufen "
                    f"(~{saving:.2f}x weniger Arbeit)"
                    + (f", ausgenommen: {', '.join(single)}" if single else ""))
            _VIEW, _COLLECT_VISITS = view, collect_visits
            visits: dict[tuple, Counter] = {}
            done = 0
            t_view = time.perf_counter()

            # Sammelt je (Budget, Estimator), damit erst geloggt wird, wenn eine
            # ganze Gruppe fertig ist -- sonst waeren es n_runs mal so viele Zeilen.
            pending: dict[tuple, list] = {}
            expected = len(estimators) * len(budgets)

            def handle(results):
                for row, vis in results:
                    handle_row(row, vis)

            def handle_row(row, vis):
                nonlocal done
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
                    log(f"  [{done:>3}/{expected}]{where} {key[1]:<26} b={key[0]:<6g} "
                        f"est/|V|={med / true_size:8.4f}  Schritte={_n(steps):>12}  "
                        f"{secs:7.1f}s CPU  |  {elapsed/60:5.1f} min, Rest ~{eta/60:.0f} min")

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
                         "seed": seed, "start_node": start, "budget_rel": b,
                         "node": graph.name_of(node), "visits": count}
                        for node, count in counter.items()
                    )

        _VIEW = None
        del view
        gc.collect()
        log(f"[{graph.name}/{view_name}] fertig in "
            f"{(time.perf_counter()-t_all)/60:.1f} min")

    log(f"[{graph.name}] gesamt {(time.perf_counter()-t_start)/60:.1f} min")
    results = pd.DataFrame(rows).sort_values(
        ["view", "start_node", "budget_rel", "estimator", "run"],
        na_position="first").reset_index(drop=True)
    visits_df = pd.DataFrame(visit_rows) if collect_visits else None
    return results, visits_df
