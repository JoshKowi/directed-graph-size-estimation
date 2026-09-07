"""CLI: wie oft geraet ein einfacher Random Walk in ein "Sink"?

Ein Sink ist hier eine Region, aus der der Walk nicht mehr herausfindet: er
wiederholt nur noch bekannte Knoten. Operationalisiert als **Stagnation** --
fuer `z` aufeinanderfolgende Schritte (Default 1000) waechst die Zahl der
verschieden besuchten Knoten nicht mehr. Sobald das eintritt, gilt der Lauf als
versunken und endet.

Gemessen wird ueber viele Laeufe (Default 100), jeder mit demselben Budget
(Default 20 % von |V|). Jeder Lauf liefert genau eine Zahl: den Budget-Anteil,
bei dem er versunken ist -- oder nichts, wenn er das volle Budget ueberlebt hat.
Daraus entsteht eine Survival-Kurve: x = verbrauchter Anteil des Budgets,
y = wie viele Laeufe zu diesem Zeitpunkt noch "live" sind (noch neue Knoten
finden). Laeufe, die nie versinken, bleiben bis x = 1 in der Kurve
(rechts-zensiert).

Die Sackgassen-Strategie (`--dead-end`) ist frei waehlbar; die Auswahl kommt
aus sampling.dead_ends.DEAD_ENDS, kuenftige Strategien sind also automatisch
dabei. Sie bestimmt mit, was ein Sink ueberhaupt bedeutet: `history` bleibt in
der besuchten Region (Sink wirkt absorbierend), `restart` kann ueber den
Startknoten entkommen, `backtrack` sucht laenger, bevor es aufgibt.

Ergebnisse landen als CSV unter data/results/<graph>__[...]walk_sink.csv
(angehaengt, nicht ueberschrieben -- wie run_experiment.py). Der Plot laesst
sich mit --plot-only jederzeit ohne Neurechnung aus der CSV erzeugen.

Beispiele:
    python walk_sinks.py --graph Slashdot0811 --runs 5 --z 50 --budget 0.02 --no-plot
    python walk_sinks.py --graph gpt4_io --views directed --dead-end history --jobs 8
    python walk_sinks.py --graph gpt4_io --plot-only
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import random

import pandas as pd

import config
import provenance
from experiment import results as results_io
from graphs import loader
from graphs.views import VIEWS, build_view
from oracles.base import BudgetExceeded
from oracles.local_access import CrawlOracle
from sampling.dead_ends import DEAD_ENDS
from sampling.samplers import RandomWalkSampler

DEFAULT_STAGNATION = 1000   # Schritte ohne neuen Knoten, bis ein Lauf als Sink gilt
KIND = "walk_sink"


class WalkSunk(BudgetExceeded):
    """Der Walk stagniert -- z Schritte ohne neuen Knoten.

    Erbt von BudgetExceeded, damit RandomWalkSampler.sample() sie mit demselben
    `except` faengt wie das Budgetende und die bis dahin gelaufene Trajektorie
    sauber zurueckgibt.
    """


class _SinkOracle(CrawlOracle):
    """CrawlOracle, das die Stagnation mitzaehlt und den Lauf bei Sink beendet.

    `_fetch` laeuft genau einmal je Walk-Schritt (auch beim Wiederbesuch, s.
    oracles.base). Steigt `len(self.visits)` bei diesem Schritt, wurde ein neuer
    Knoten entdeckt und der Zaehler faellt zurueck auf 0; sonst waechst er.
    Erreicht er `stagnation`, ist der Lauf versunken: Zeitpunkt festhalten und
    WalkSunk werfen.
    """

    def __init__(self, *args, stagnation: int = DEFAULT_STAGNATION, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stagnation = int(stagnation)
        self.steps = 0
        self._since_new = 0
        self._last_distinct = 0
        self.sunk_at_queries: float | None = None
        self.sunk_at_step: int | None = None

    def _fetch(self, u):
        out = super()._fetch(u)          # bucht Budget, kann BudgetExceeded werfen
        self.steps += 1
        distinct = len(self.visits)
        if distinct > self._last_distinct:
            self._last_distinct = distinct
            self._since_new = 0
        else:
            self._since_new += 1
            if self.sunk_at_queries is None and self._since_new >= self.stagnation:
                self.sunk_at_queries = self.queries
                self.sunk_at_step = self.steps
                self.stopped_by = "sink"
                raise WalkSunk(f"{self.stagnation} Schritte ohne neuen Knoten")
        return out


def run_walk(view, rng: random.Random, budget_abs: int, dead_end: str,
             stagnation: int = DEFAULT_STAGNATION) -> dict:
    """Ein Lauf. Endet beim Budgetende oder sobald der Walk stagniert."""
    oracle = _SinkOracle(view, rng, budget_abs, config.DEFAULT_BUDGET_METRIC,
                         stagnation=stagnation)
    sampler = RandomWalkSampler(dead_end=DEAD_ENDS[dead_end]())
    sampler.sample(oracle)
    sunk = oracle.sunk_at_queries is not None
    return {
        "steps": oracle.steps,
        "distinct": oracle.unique_nodes,
        "queries_used": oracle.queries,
        "budget_abs": budget_abs,
        "sunk": sunk,
        "sunk_step": oracle.sunk_at_step,
        "sunk_queries": oracle.sunk_at_queries,
        "sunk_frac": oracle.sunk_at_queries / budget_abs if sunk else float("nan"),
        "stopped_by": oracle.stopped_by,   # "sink" | "budget"
    }


# Vom Fork-Pool geteilt: die View wird -- wie in experiment.runner -- als
# Modul-Global vor dem Pool gesetzt und per copy-on-write vererbt, nie gepickelt.
_VIEW = None
_STAGNATION = DEFAULT_STAGNATION


def _run_one(task):
    run_idx, seed_str, budget_abs, dead_end = task
    res = run_walk(_VIEW, random.Random(seed_str), budget_abs, dead_end, _STAGNATION)
    res["run"] = run_idx
    return res


def _resolve_start(graph_name: str, raw: str | None):
    """--start-node auf einen Eintrag aus config.SEED_NODES abbilden (wie
    run_experiment._resolve_starts, aber nur ein einzelner Knoten)."""
    known = config.seed_nodes(graph_name)
    if raw is None:
        return None
    if not known:
        raise SystemExit(f"Fuer {graph_name} sind keine Einstiegsknoten hinterlegt "
                         "(config.SEED_NODES)")
    match = [k for k in known if str(k).lower() == raw.lower()]
    if not match:
        raise SystemExit(
            f"{raw!r} ist kein Einstiegsknoten von {graph_name}. "
            f"Moeglich: {', '.join(map(str, known))}")
    return match[0]


def _compute(args) -> pd.DataFrame:
    global _VIEW, _STAGNATION
    graph = loader.load_graph(args.graph)
    start = _resolve_start(graph.name, args.start_node)
    code = provenance.code_fingerprint()
    _STAGNATION = args.z

    rows: list[dict] = []
    for view_name in args.views:
        view = build_view(graph, view_name)
        view.seed_ids()                     # vor dem Fork aufloesen (s. graphs.graph)
        if start is not None:
            view.restrict_seeds([start])
        budget_abs = max(int(round(args.budget * view.n_nodes)), 2)
        start_col = (str(start) if start is not None
                     else ("<mixed>" if config.seed_nodes(graph.name) else "<zufaellig>"))

        tasks = [
            (run_idx,
             f"{args.seed}|walk-sink|{args.dead_end}|{view_name}|{run_idx}",
             budget_abs, args.dead_end)
            for run_idx in range(args.runs)
        ]

        _VIEW = view
        gc.freeze()
        if args.jobs > 1:
            with mp.get_context("fork").Pool(args.jobs) as pool:
                results = list(pool.imap_unordered(_run_one, tasks, chunksize=1))
        else:
            results = [_run_one(t) for t in tasks]

        n_sunk = sum(r["sunk"] for r in results)
        print(f"  [{graph.name}/{view_name}] {len(results)} Laeufe, Budget "
              f"{args.budget:g} = {budget_abs:,} Einheiten -- {n_sunk} versunken, "
              f"{len(results) - n_sunk} ueberlebt".replace(",", " "))

        for r in results:
            rows.append({
                "graph": graph.name, "view": view_name, "seed": args.seed,
                "start_node": start_col, "estimator": f"walk-sink__{args.dead_end}",
                "dead_end": args.dead_end, "budget_rel": args.budget, "z": args.z,
                "code": code, **r,
            })

    df = pd.DataFrame(rows).sort_values(["view", "run"]).reset_index(drop=True)
    save = results_io.save_results if args.replace else results_io.append_results
    path = save(df, graph.name, kind=KIND, seed=args.seed, start=args.start_node)
    print("  ->", path)
    provenance.write_readmes()
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", required=True, help="Name oder Kuerzel")
    p.add_argument("--views", nargs="+", default=["directed"],
                   choices=sorted(VIEWS), help="Kantensichten (Default: directed)")
    p.add_argument("--dead-end", default="history", choices=sorted(DEAD_ENDS),
                   help="Sackgassen-Strategie des Walks (Default: history)")
    p.add_argument("--budget", type=float, default=0.20,
                   help="Budget je Lauf, relativ zu |V| (Default: 0.20)")
    p.add_argument("--runs", type=int, default=100, help="Zahl der Laeufe (Default: 100)")
    p.add_argument("--z", type=int, default=DEFAULT_STAGNATION,
                   help=f"Schritte ohne neuen Knoten bis 'Sink' (Default: {DEFAULT_STAGNATION})")
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
        graph_name = config.resolve_graph(args.graph)
        df = results_io.load_results(graph_name, kind=KIND, seed=args.seed,
                                     start=args.start_node)
        df = df[df["dead_end"] == args.dead_end]
        if df.empty:
            raise SystemExit(f"Keine {KIND}-Zeilen fuer {graph_name} "
                             f"(dead_end={args.dead_end}, seed={args.seed})")
    else:
        df = _compute(args)

    if not args.no_plot:
        from plotting.walk_survival import plot_survival
        print("  ->", plot_survival(df))


if __name__ == "__main__":
    main()
