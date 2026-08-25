"""CLI: Ergebnisse aus data/results plotten.

Pro Panel sind hoechstens 8 Estimators darstellbar (so viele klar
unterscheidbare Farben gibt es); mit --estimators bzw. --match waehlt man eine
Teilmenge aus.

Beispiele:
    python plot_results.py --graphs Slashdot0811
    python plot_results.py --match uniform rw_plain__backtrack
    python plot_results.py --estimators uniform_collision_weighted rw_plain__restart__none
"""

from __future__ import annotations

import argparse

from experiment import results as results_io
from plotting.ranges import plot_ranges


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graphs", nargs="+", default=None, help="Graph-Namen (Default: alle)")
    p.add_argument("--estimators", nargs="+", default=None, help="exakte Estimator-Namen")
    p.add_argument("--match", nargs="+", default=None,
                   help="Estimators, deren Name einen dieser Teilstrings enthaelt")
    args = p.parse_args()

    df = results_io.load_results()
    if df.empty:
        print("Keine Ergebnisse gefunden -- zuerst run_experiment.py ausfuehren.")
        return

    if args.estimators:
        df = df[df["estimator"].isin(args.estimators)]
    if args.match:
        df = df[df["estimator"].str.contains("|".join(args.match), regex=True)]
    if df.empty:
        print("Auswahl trifft auf keine Zeile zu.")
        return

    summary = results_io.summarize(df)
    comparison = results_io.compare_views(df)
    for name in args.graphs or sorted(summary["graph"].unique()):
        plot_ranges(summary, graph_name=name)
        print("  ->", f"data/plots/{name}__ranges.png")
        print("  ->", results_io.save_results(
            comparison[comparison["graph"] == name], name, kind="view_comparison"))


if __name__ == "__main__":
    main()
