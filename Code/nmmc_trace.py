"""CLI: was NMMC waehrend eines Laufs innen tut.

Die Ergebnis-CSV von run_experiment.py sagt, *was* NMMC schaetzt, aber nicht,
*warum*: die Pipeline reicht vom Sampler nichts weiter als die Samples
(estimators.pipeline.evaluate baut `extra` allein aus der Formel, und
experiment.runner._row kopiert eine feste Schluesselliste aus den Kosten).
Genau die Groessen, an denen NMMC haengt, fehlen dort also.

Aus der bestehenden CSV lassen sich nur *Schranken* lesen: jede Umverteilung
landet auf einem schon geholten Knoten und erzeugt einen Cache-Treffer, jeder
Erstbesuch wurde ueber einen angenommenen Zug erreicht. Also ist
`cached_queries / extra_n_samples` eine obere Schranke des
Umverteilungsanteils und `unique_nodes_used / extra_n_samples` eine untere
Schranke der Annahmequote. Die Luecke dazwischen -- angenommene Zuege auf
bereits bekannte Knoten -- ist dort per Konstruktion unsichtbar. Dieses Skript
misst sie.

Zwei Zahlen tragen die Auswertung:

`frac_dinhat_1` -- Anteil der Vorschlaege, bei denen der Eingangsgrad im Nenner
von b noch 1 war, der Nenner also keine Information trug. Im Grenzfall 1 ist
b = d+(i) und die Kette P = A/c, deren QSD die Eigenvektorzentralitaet ist
(Korollar 3.3) statt der Zielverteilung. Gemessen liegt der Wert deutlich
darunter, weil ein verfangener Walk die Kanten seiner eigenen Region immer
wieder sieht -- die Schaetzung ist lokal gut und nur am Rand des Besuchten
gleich 1 (s. sampling.indegree).

`c_ratio` = c_t/c -- wie weit die gelernte Normierung an das wahre max b_ij
herangekommen ist. Solange sie darunter liegt, greift die Kappung gamma = 1 und
die QSD ist noch gar nicht die Zielverteilung; die Verzerrung haengt dann am
Budget statt am Verfahren.

`--indeg exact` ist zu beidem die Gegenprobe.

Gefahren wird das Kreuzprodukt aus `--indeg` und `--target`, je `--runs` Laeufe
mit demselben Budget. Ergebnisse landen als CSV unter
data/results/<graph>__[...]nmmc_trace.csv (angehaengt, nicht ueberschrieben --
wie run_experiment.py); --plot-only erzeugt den Plot ohne Neurechnung.

Beispiele:
    python nmmc_trace.py --graph slashdot --runs 10 --budget 0.05
    python nmmc_trace.py --graph gpt4o_io --indeg online --budget 0.01 --jobs 8
    python nmmc_trace.py --graph slashdot --plot-only
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import random

import numpy as np
import pandas as pd

import config
import provenance
from experiment import results as results_io
from graphs import loader
from graphs.views import VIEWS, build_view
from oracles.local_access import (CrawlOracle, CrossInDegreeCrawlOracle,
                                  InDegreeCrawlOracle)
from sampling.indegree import IN_DEGREES
from sampling.nmmc import TARGETS, NmmcSampler

KIND = "nmmc_trace"

# Dieselbe Zuordnung wie in estimators/methods/nmmc.py -- dort ist sie die
# Wahrheit, hier nur nachgezogen, damit die Diagnose ohne Registry auskommt.
INDEG_ORACLES = {"online": CrawlOracle, "exact": InDegreeCrawlOracle,
                 "cross-one": CrossInDegreeCrawlOracle,
                 "cross-online": CrossInDegreeCrawlOracle}


def true_c(view, target: str, indeg: str) -> float:
    """Das wahre c = max b_ij, gegen das c_t laeuft -- NaN, wo es keines gibt.

    Global gerechnet und damit nur als Diagnose zulaessig; der Sampler selbst
    kennt es nie, er lernt c_t unterwegs (Algorithmus 2 des Papers).

    Entscheidend: c haengt an den Eingangsgraden, die der Sampler *tatsaechlich
    benutzt*, nicht an den wahren. Fuer `exact` sind das die echten, fuer
    `cross-one` die des Partners mit Boden 1 -- beides steht vorab fest. Bei
    `online` und `cross-online` haengt der Nenner dagegen am bisherigen Lauf;
    ein festes c existiert dort gar nicht, und eine Quote dagegen waere frei
    erfunden. Deshalb NaN statt einer Zahl, die man versehentlich liest.
    """
    if indeg in ("online", "cross-online"):
        return float("nan")
    if indeg == "cross-one":
        from oracles.local_access import CrossInDegreeCrawlOracle
        idx = CrossInDegreeCrawlOracle._index(view)
        d_in = np.maximum(np.where(idx < 0, 1, idx), 1).astype(float)
    else:
        d_in = np.maximum(view.in_degrees, 1).astype(float)
    d_out = np.diff(view.indptr).astype(float)
    if target == "indeg":
        # b haengt nur an i; Knoten ohne Ausgangskanten schlagen nie vor.
        live = d_out > 0
        return float((d_out[live] / d_in[live]).max()) if live.any() else 1.0
    src = np.repeat(np.arange(view.n_nodes), np.diff(view.indptr))
    return float((d_out[src] / d_in[view.indices]).max()) if len(src) else 1.0


def run_trace(view, rng: random.Random, budget_abs: int, indeg: str,
              target: str, alpha: float, agents: int = 1,
              shared_c: bool = False) -> dict:
    """Ein Lauf mit eingeschalteter Diagnose."""
    oracle = INDEG_ORACLES[indeg](view, rng, budget_abs,
                                  config.DEFAULT_BUDGET_METRIC)
    sampler = NmmcSampler(target=target, indeg=IN_DEGREES[indeg](), alpha=alpha,
                          n_agents=agents, shared_c=shared_c)
    stats: dict = {}
    trace = sampler.sample(oracle, stats=stats)

    prop = max(stats["proposals"], 1)
    steps = max(stats["steps"], 1)
    d_hat = np.array([s.degree for s in trace], dtype=float) if trace else np.zeros(1)
    return {
        "budget_abs": budget_abs,
        # Wie viele Agenten wirklich liefen: bei kleinem Budget kappt der
        # Sampler die Zahl (s. sampling.nmmc).
        "agents_eff": stats["agents_eff"],
        "steps": stats["steps"],
        "n_samples": len(trace),
        "n_unique": oracle.unique_nodes,
        "queries_used": oracle.queries,
        "cached_queries": oracle.cached_queries,
        "accepted": stats["accepted"],
        "redistributed": stats["redistributed"],
        "dead_ends": stats["dead_ends"],
        "proposals": stats["proposals"],
        # Die Groessen, die es sonst nirgends gibt:
        "acc_rate": stats["accepted"] / prop,
        "redist_share": (stats["redistributed"] + stats["dead_ends"]) / steps,
        "mean_gamma": stats["gamma_sum"] / prop,
        "frac_dinhat_1": stats["dinhat_one"] / prop,
        "dinhat_mean": float(d_hat.mean()),
        "dinhat_p99": float(np.percentile(d_hat, 99)),
        "c_final": stats["c_final"],
        # Schranken, die auch die Haupt-CSV hergibt -- hier zum Abgleich.
        "acc_rate_lower": oracle.unique_nodes / steps,
        "redist_share_upper": oracle.cached_queries / steps,
        "stopped_by": oracle.stopped_by,
    }


# Wie in experiment.runner und walk_sinks.py: die View wird als Modul-Global
# vor dem Pool gesetzt und per copy-on-write vererbt, nie gepickelt.
_VIEW = None


def _run_one(task):
    run_idx, seed_str, budget_abs, indeg, target, alpha, agents, shared_c = task
    res = run_trace(_VIEW, random.Random(seed_str), budget_abs, indeg, target,
                    alpha, agents, shared_c)
    res.update(run=run_idx, indeg=indeg, target=target, alpha=alpha,
               agents=agents, shared_c=shared_c)
    return res


def _compute_graph(args, graph_name: str, code: str) -> pd.DataFrame:
    global _VIEW
    graph = loader.load_graph(graph_name)
    start = _resolve_start(graph.name, args.start_node)

    rows: list[dict] = []
    for view_name in args.views:
        view = build_view(graph, view_name)
        view.seed_ids()                     # vor dem Fork aufloesen
        for _ind in args.indeg:                 # ebenso (s. oracles.base.prepare)
            INDEG_ORACLES[_ind].prepare(view)
        if start is not None:
            view.restrict_seeds([start])
        budget_abs = max(int(round(args.budget * view.n_nodes)), 2)
        start_col = (str(start) if start is not None
                     else ("<mixed>" if config.seed_nodes(graph.name) else "<zufaellig>"))
        c_true = {(t, i): true_c(view, t, i)
                  for t in args.target for i in args.indeg}

        tasks = [
            (run_idx,
             f"{args.seed}|nmmc-trace|{indeg}|{target}|{args.alpha:g}|{k}"
             f"|{int(args.shared_c)}|{view_name}|{run_idx}",
             budget_abs, indeg, target, args.alpha, k, args.shared_c)
            for indeg in args.indeg for target in args.target
            for k in args.agents for run_idx in range(args.runs)
        ]

        _VIEW = view
        gc.freeze()
        if args.jobs > 1:
            with mp.get_context("fork").Pool(args.jobs) as pool:
                results = list(pool.imap_unordered(_run_one, tasks, chunksize=1))
        else:
            results = [_run_one(t) for t in tasks]

        for r in results:
            r["c_max"] = c_true[(r["target"], r["indeg"])]
            r["c_ratio"] = r["c_final"] / r["c_max"]
            rows.append({
                "graph": graph.name, "view": view_name, "seed": args.seed,
                "start_node": start_col,
                "estimator": f"nmmc-trace__{r['indeg']}__{r['target']}",
                "budget_rel": args.budget, "code": code, **r,
            })

        for indeg in args.indeg:
            for target in args.target:
              for k in args.agents:
                sel = [r for r in results if r["indeg"] == indeg
                       and r["target"] == target and r["agents"] == k]
                med = lambda k: float(np.median([r[k] for r in sel]))  # noqa: E731
                ratio = med("c_ratio")
                ratio_s = f"{ratio:6.3f}" if ratio == ratio else "     -"
                print(f"  [{graph.name}/{view_name}] {indeg:12s} {target:7s} "
                      f"K={k:<5d} "
                      f"Annahme {med('acc_rate'):6.3f}  d-=1 bei {med('frac_dinhat_1'):6.3f}  "
                      f"c_t/c {ratio_s}  Schritte {med('steps'):,.0f}"
                      .replace(",", " "))

    df = pd.DataFrame(rows).sort_values(
        ["view", "indeg", "target", "run"]).reset_index(drop=True)
    save = results_io.save_results if args.replace else results_io.append_results
    path = save(df, graph.name, kind=KIND, seed=args.seed, start=args.start_node)
    print("  ->", path)
    return df


def _resolve_start(graph_name: str, raw: str | None):
    """--start-node auf einen Eintrag aus config.SEED_NODES abbilden (wie
    walk_sinks._resolve_start)."""
    known = config.seed_nodes(graph_name)
    if raw is None:
        return None
    if not known:
        raise SystemExit(f"Fuer {graph_name} sind keine Einstiegsknoten hinterlegt "
                         "(config.SEED_NODES)")
    match = [k for k in known if str(k).lower() == raw.lower()]
    if not match:
        raise SystemExit(f"{raw!r} ist kein Einstiegsknoten von {graph_name}. "
                         f"Moeglich: {', '.join(map(str, known))}")
    return match[0]


def _compute(args) -> list[tuple[str, pd.DataFrame]]:
    code = provenance.code_fingerprint()
    out = [(config.resolve_graph(g), _compute_graph(args, g, code)) for g in args.graphs]
    provenance.write_readmes()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", "--graphs", dest="graphs", nargs="+", required=True,
                   metavar="GRAPH", help="ein oder mehrere Namen/Kuerzel")
    p.add_argument("--views", nargs="+", default=["directed"],
                   choices=sorted(VIEWS), help="Kantensichten (Default: directed)")
    p.add_argument("--indeg", nargs="+", default=sorted(IN_DEGREES),
                   choices=sorted(IN_DEGREES),
                   help="Herkunft des Eingangsgrades (Default: beide)")
    p.add_argument("--target", nargs="+", default=list(TARGETS), choices=list(TARGETS),
                   help="Zielverteilung der QSD (Default: beide)")
    p.add_argument("--agents", nargs="+", type=int, default=[1],
                   help="Zahl der Agenten je Lauf (Default: 1); mehrere Werte "
                        "ergeben ein Kreuzprodukt")
    p.add_argument("--shared-c", action="store_true",
                   help="c_t ueber alle Agenten stehen lassen statt je Agent "
                        "auf 1 zurueckzusetzen (Gegenprobe, s. sampling.nmmc)")
    p.add_argument("--alpha", type=float, default=config.NMMC_ALPHA,
                   help=f"Gedaechtnis der Umverteilung (Default: {config.NMMC_ALPHA:g})")
    p.add_argument("--budget", type=float, default=0.05,
                   help="Budget je Lauf, relativ zu |V| (Default: 0.05)")
    p.add_argument("--runs", type=int, default=10, help="Laeufe je Variante (Default: 10)")
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED,
                   help=f"Basis der Zufallsstroeme (Default: {config.DEFAULT_SEED})")
    p.add_argument("--start-node", default=None,
                   help="fester Einstiegsknoten (Default: je Lauf zufaellig aus "
                        "config.SEED_NODES)")
    p.add_argument("--jobs", type=int, default=1,
                   help="Parallele Prozesse ueber die Laeufe (Default: 1)")
    p.add_argument("--replace", action="store_true",
                   help="CSV ueberschreiben statt anhaengen")
    p.add_argument("--no-plot", action="store_true", help="nur CSV, kein Plot")
    p.add_argument("--plot-only", action="store_true",
                   help="nicht rechnen, Plot aus vorhandener CSV erzeugen")
    args = p.parse_args()

    if args.plot_only:
        pairs = []
        for g in args.graphs:
            graph_name = config.resolve_graph(g)
            df = results_io.load_results(graph_name, kind=KIND, seed=args.seed,
                                         start=args.start_node)
            df = df[df["indeg"].isin(args.indeg) & df["target"].isin(args.target)
                    & df["agents"].isin(args.agents)]
            if df.empty:
                raise SystemExit(f"Keine {KIND}-Zeilen fuer {graph_name} "
                                 f"(seed={args.seed})")
            pairs.append((graph_name, df))
    else:
        pairs = _compute(args)

    if not args.no_plot:
        from plotting.nmmc_diagnosis import plot_nmmc_trace
        for _, df in pairs:
            print("  ->", plot_nmmc_trace(df))


if __name__ == "__main__":
    main()
