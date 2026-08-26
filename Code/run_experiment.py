"""CLI: Experiment fuer einen oder mehrere Graphen ausfuehren.

Laedt jeden Graphen genau einmal, laesst alle gewaehlten Estimators fuer alle
Budgets je n-mal laufen und schreibt Ergebnisse + Besuchsstatistik nach
data/results/.

Beispiele:
    python run_experiment.py --list
    python run_experiment.py --graphs Slashdot0811
    python run_experiment.py --graphs Slashdot0811 --budgets 0.001 0.01 0.1 --runs 20
    python run_experiment.py --graphs Slashdot0811 --views directed undirected reverse
    python run_experiment.py --graphs Slashdot0811 --seed 7

Der Seed bestimmt den kompletten Zufallsstrom. Ein zweiter Lauf mit anderem
Seed ist ein zweiter, gleichberechtigter Durchlauf desselben Experiments --
so laesst sich pruefen, ob ein Ergebnis stabil ist oder am Zufall haengt. Er
steht in jeder Ergebniszeile (Spalte `seed`), im Dateinamen (`__seed7__`,
ausser beim Default) und auf jeder daraus erzeugten Grafik.

`--checkpoint-budgets` liest alle Budgets aus einem einzigen Lauf ab, statt je
Budget einen eigenen zu rechnen -- dieselben Zahlen bei rund 40 % weniger
Rechenzeit, dafuer sind die Punkte eines Laufs ueber die Budgets genestet.
Siehe experiment/runner.py und check_nested.py.
"""

from __future__ import annotations

import argparse
import time

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
    p.add_argument("--budgets", nargs="+", type=float, default=None,
                   help=f"Default: {list(config.DEFAULT_BUDGETS)}, bei Graphen ab "
                        f"{config.LARGE_GRAPH_NODES:,} Knoten "
                        f"{list(config.DEFAULT_BUDGETS_LARGE)}".replace(",", " "))
    p.add_argument("--views", nargs="+", default=list(config.DEFAULT_VIEWS),
                   choices=sorted(VIEWS), help="Kantensichten (Default: directed undirected)")
    p.add_argument("--runs", type=int, default=config.DEFAULT_N_RUNS)
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED,
                   help=f"Zufallsstrom des ganzen Laufs (Default: {config.DEFAULT_SEED}). "
                        "Abweichende Seeds schreiben in eigene Dateien "
                        "(<graph>__seed<N>__estimates.csv) und ueberschreiben "
                        "vorhandene Ergebnisse daher nicht.")
    p.add_argument("--jobs", type=int, default=config.DEFAULT_N_JOBS,
                   help="Parallele Prozesse je View (1 = sequentiell)")
    p.add_argument("--no-visits", action="store_true", help="Besuchsstatistik nicht speichern")
    p.add_argument("--checkpoint-budgets", action="store_true",
                   help="alle Budgets aus einem Lauf ablesen statt je Budget einen "
                        "eigenen zu rechnen. Exakt dieselben Zahlen, spart "
                        "Sigma(Budgets)/max(Budget) an Rechenzeit -- die Punkte "
                        "eines Laufs sind danach aber genestet, nicht unabhaengig "
                        "(Spalte `nested`).")
    args = p.parse_args()

    if args.list:
        print("Graphen:   ", ", ".join(loader.available_graphs()))
        print("Estimators:", ", ".join(sorted(estimators.REGISTRY)))
        print("Views:     ", ", ".join(sorted(VIEWS)))
        return

    graph_names = args.graphs or loader.available_graphs()
    ests = estimators.build_all(args.estimators)

    for name in graph_names:
        t0 = time.perf_counter()
        graph = loader.load_graph(name)
        without = graph.n_nodes - graph.n_with_out_edges
        print(f"[{name}] geladen in {time.perf_counter() - t0:.1f}s -- "
              f"|V| = {graph.n_nodes:,} ({graph.n_with_out_edges:,} mit ausgehenden "
              f"Kanten, {without:,} ohne), {graph.n_edges:,} Kanten".replace(",", " "),
              flush=True)

        budgets = args.budgets
        if budgets is None:
            large = graph.n_nodes >= config.LARGE_GRAPH_NODES
            budgets = list(config.DEFAULT_BUDGETS_LARGE if large
                           else config.DEFAULT_BUDGETS)
            if large:
                print(f"[{name}] grosser Graph -- 20-%-Budget ausgelassen "
                      f"(mit --budgets erzwingbar)", flush=True)
        print(f"[{name}] {len(ests)} Estimators x {len(budgets)} Budgets x "
              f"{args.runs} Laeufe x {len(args.views)} Views = "
              f"{len(ests) * len(budgets) * args.runs * len(args.views)} Schaetzungen, "
              f"{args.jobs} Prozesse, Seed {args.seed}", flush=True)

        df, visits = run_graph(
            graph,
            ests,
            budgets=budgets,
            n_runs=args.runs,
            seed=args.seed,
            views=args.views,
            collect_visits=not args.no_visits,
            n_jobs=args.jobs,
            nested_budgets=args.checkpoint_budgets,
            log=lambda m: print(m, flush=True),
        )
        print("  ->", results_io.save_results(df, name, seed=args.seed))
        if visits is not None:
            print("  ->", results_io.save_results(visits, name, kind="visits",
                                                  seed=args.seed))
        loader.clear_cache()

    # Ordner-README aktuell halten -- sonst steht sie nach dem naechsten Lauf falsch da
    for path in provenance.write_readmes():
        print("  ->", path)


if __name__ == "__main__":
    main()
