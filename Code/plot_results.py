"""CLI: Ergebnisse aus data/results plotten.

Pro Panel sind hoechstens 8 Estimators darstellbar (so viele klar
unterscheidbare Farben gibt es); mit --estimators bzw. --match waehlt man eine
Teilmenge aus.

Beispiele:
    python plot_results.py --graphs Slashdot0811
    python plot_results.py --match uniform rw-plain__backtrack
    python plot_results.py --estimators uniform-collision__weighted rw-plain__restart__none
    python plot_results.py --graphs Slashdot0811 --seed 7

Je Graph *und* Seed entsteht ein eigenes Bild: verschiedene Seeds sind
verschiedene Durchlaeufe des Experiments und gehoeren nicht in dieselbe
Spanne. Der Seed steht oben rechts im Bild.
"""

from __future__ import annotations

import argparse

import config
from experiment import results as results_io
from plotting.ranges import plot_ranges


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graphs", nargs="+", default=None, help="Graph-Namen (Default: alle)")
    p.add_argument("--estimators", nargs="+", default=None, help="exakte Estimator-Namen")
    p.add_argument("--match", nargs="+", default=None,
                   help="Estimators, deren Name einen dieser Teilstrings enthaelt")
    p.add_argument("--seed", type=int, default=None,
                   help="nur diesen Lauf plotten (Default: jeden vorhandenen Seed)")
    p.add_argument("--start-node", default=None,
                   help="nur diesen Einstiegsknoten plotten (Default: jeden vorhandenen)")
    args = p.parse_args()

    df = results_io.load_results(seed=args.seed, start=args.start_node)
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

    # Seed *und* Einstiegsknoten trennen Bedingungen -- je Paar ein Bild.
    for seed, start in results_io.conditions_available(df):
        part = df[(df["seed"] == seed) & (df["start_node"] == start)]
        summary = results_io.summarize(part)
        comparison = results_io.compare_views(part)
        note = f"seed {seed}"
        if start is not None:
            note += f"  |  start: {start}"
        if summary["nested"].any():      # Punkte je Lauf dann korreliert
            note += "  |  nested budgets"
        if summary["shared"].notna().any():
            note += "  |  shared walks"
        for name in ([config.resolve_graph(g) for g in args.graphs]
                     if args.graphs else sorted(summary["graph"].unique())):
            rows = summary[summary["graph"] == name]
            if rows.empty:      # diese Bedingung gibt es fuer den Graphen nicht
                continue
            tag = results_io.seed_tag(seed) + results_io.start_tag(name, start)
            path = config.PLOTS_DIR / f"{name}__{tag}ranges.png"
            plot_ranges(rows, graph_name=name, path=path, note=note)
            print("  ->", path)
            print("  ->", results_io.save_results(
                comparison[comparison["graph"] == name], name,
                kind="view_comparison", seed=seed, start=start))


if __name__ == "__main__":
    main()
