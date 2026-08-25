"""CLI: Experiment fuer einen oder mehrere Graphen ausfuehren.

Laedt jeden Graphen genau einmal, laesst alle gewaehlten Estimators fuer alle
Budgets je n-mal laufen und schreibt Ergebnisse + Besuchsstatistik nach
data/results/.

Beispiele:
    python run_experiment.py --list
    python run_experiment.py --graphs Slashdot0811
    python run_experiment.py --graphs Slashdot0811 --budgets 0.001 0.01 0.1 --runs 20
    python run_experiment.py --graphs Slashdot0811 --views directed undirected reverse
"""

from __future__ import annotations

import argparse

import config
import estimators
from experiment import results as results_io
from experiment.runner import run_graph
from graphs import loader
from graphs.views import VIEWS
import provenance


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true", help="Graphen und Estimators anzeigen")
    p.add_argument("--graphs", nargs="+", default=None, help="Graph-Namen (Default: alle)")
    p.add_argument("--estimators", nargs="+", default=None, help="Estimator-Namen (Default: alle)")
    p.add_argument("--budgets", nargs="+", type=float, default=list(config.DEFAULT_BUDGETS))
    p.add_argument("--views", nargs="+", default=list(config.DEFAULT_VIEWS),
                   choices=sorted(VIEWS), help="Kantensichten (Default: directed undirected)")
    p.add_argument("--runs", type=int, default=config.DEFAULT_N_RUNS)
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    p.add_argument("--no-visits", action="store_true", help="Besuchsstatistik nicht speichern")
    args = p.parse_args()

    if args.list:
        print("Graphen:   ", ", ".join(loader.available_graphs()))
        print("Estimators:", ", ".join(sorted(estimators.REGISTRY)))
        print("Views:     ", ", ".join(sorted(VIEWS)))
        return

    graph_names = args.graphs or loader.available_graphs()
    ests = estimators.build_all(args.estimators)

    for name in graph_names:
        print(f"[{name}] laden ...", flush=True)
        graph = loader.load_graph(name)
        print(f"[{name}] |V| = {graph.n_nodes} "
              f"({graph.n_with_out_edges} mit ausgehenden Kanten, "
              f"{graph.n_nodes - graph.n_with_out_edges} ohne)", flush=True)

        df, visits = run_graph(
            graph,
            ests,
            budgets=args.budgets,
            n_runs=args.runs,
            seed=args.seed,
            views=args.views,
            collect_visits=not args.no_visits,
        )
        print("  ->", results_io.save_results(df, name))
        if visits is not None:
            print("  ->", results_io.save_results(visits, name, kind="visits"))
        loader.clear_cache()

    # Ordner-README aktuell halten -- sonst steht sie nach dem naechsten Lauf falsch da
    for path in provenance.write_readmes():
        print("  ->", path)


if __name__ == "__main__":
    main()
