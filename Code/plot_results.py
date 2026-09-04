"""CLI: Ergebnisse aus data/results plotten.

Alle gewaehlten Estimators stehen zusammen in einem Panel je Kantensicht --
die Kategorie (Vergleich / real umsetzbar) teilt das Bild nicht auf. Pro Bild
sind hoechstens 8 Estimators darstellbar (so viele klar unterscheidbare Farben
gibt es); mit --estimators bzw. --match waehlt man eine Teilmenge aus.

Beispiele:
    python plot_results.py --graphs Slashdot0811
    python plot_results.py --match uniform rw-plain__backtrack
    python plot_results.py --estimators uniform-collision__weighted rw-plain__restart__none
    python plot_results.py --graphs Slashdot0811 --seed 7
    python plot_results.py --graphs gpt4_io --views undirected
    python plot_results.py --graphs gpt4_io --budgets 0.001 0.01 0.1

Je Graph *und* Seed entsteht ein eigenes Bild: verschiedene Seeds sind
verschiedene Durchlaeufe des Experiments und gehoeren nicht in dieselbe
Spanne. Der Seed steht oben rechts im Bild.
"""

from __future__ import annotations

import argparse

import pandas as pd

import config
from experiment import results as results_io
from graphs.views import VIEWS
from plotting.compare import VIEW_TITLES, plot_comparison


def _budget_table(costs) -> str:
    """Die Aufschluesselung ueber die Budgets zusammengefasst, eine Zeile je
    View und Estimator -- die volle Aufloesung steht in der CSV."""
    g = (costs.groupby(["view", "estimator"])[
             ["q_per_sample", "share_draw", "share_fetch", "share_cache"]]
         .median().reset_index())
    head = f"     {'view':<12} {'estimator':<34} {'q/Sample':>8} " \
           f"{'draw':>6} {'fetch':>6} {'cache':>6}"
    lines = [head, "     " + "-" * (len(head) - 5)]
    for r in g.itertuples(index=False):
        lines.append(f"     {r.view:<12} {r.estimator:<34} {r.q_per_sample:8.2f} "
                     f"{r.share_draw:6.0%} {r.share_fetch:6.0%} {r.share_cache:6.0%}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graphs", nargs="+", default=None, help="Graph-Namen (Default: alle)")
    p.add_argument("--estimators", nargs="+", default=None, help="exakte Estimator-Namen")
    p.add_argument("--match", nargs="+", default=None,
                   help="Estimators, deren Name einen dieser Teilstrings enthaelt")
    p.add_argument("--views", nargs="+", default=None, choices=sorted(VIEWS),
                   help="nur diese Kantensichten plotten (Default: alle vorhandenen)")
    p.add_argument("--budgets", nargs="+", type=float, default=None,
                   help="nur diese relativen Budgets plotten (wie bei "
                        "run_experiment.py, z. B. 0.001 0.01 0.1). Default: alle "
                        "vorhandenen. Budgets ohne passende Zeile werden ignoriert.")
    p.add_argument("--seed", type=int, default=None,
                   help="nur diesen Lauf plotten (Default: jeden vorhandenen Seed)")
    p.add_argument("--start-node", default=None,
                   help="nur diesen Einstiegsknoten plotten (Default: jeden vorhandenen)")
    p.add_argument("--intersect-budgets", action="store_true",
                   help="nur Budgets plotten, die fuer *jeden* gewaehlten Estimator "
                        "vorliegen (je Graph, Seed und Einstiegsknoten). Sonst zeigt "
                        "das Bild die Vereinigung, mit Luecken wo ein Estimator fehlt.")
    args = p.parse_args()

    df = results_io.load_results(seed=args.seed, start=args.start_node)
    if df.empty:
        print("Keine Ergebnisse gefunden -- zuerst run_experiment.py ausfuehren.")
        return

    if args.estimators:
        df = df[df["estimator"].isin(args.estimators)]
    if args.match:
        df = df[df["estimator"].str.contains("|".join(args.match), regex=True)]
    if args.views:
        df = df[df["view"].isin(args.views)]
    if args.budgets:
        want = df["budget_rel"].round(12).isin([round(b, 12) for b in args.budgets])
        df = df[want]
    if args.intersect_budgets and not df.empty:
        parts = []
        for _, grp in df.groupby(["graph", "seed", "start_node"], dropna=False):
            per_est = [set(g["budget_rel"]) for _, g in grp.groupby("estimator")]
            common = set.intersection(*per_est) if per_est else set()
            parts.append(grp[grp["budget_rel"].isin(common)])
        df = pd.concat(parts) if parts else df.iloc[:0]
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
            path = config.unique_path(
                config.PLOTS_DIR / f"{name}__{tag}ranges.png")
            # Spaltenreihenfolge aus VIEW_TITLES, nicht alphabetisch: sonst
            # stuende `reverse` vor `undirected`.
            views = [v for v in VIEW_TITLES if (rows["view"] == v).any()]
            plot_comparison(
                rows, graph_name=name,
                estimators=sorted(rows["estimator"].unique()), views=views,
                title=f"{config.graph_label(name)}: spread of size estimates "
                      "by edge view",
                path=path, note=note)
            print("  ->", path)
            print("  ->", results_io.save_results(
                comparison[comparison["graph"] == name], name,
                kind="view_comparison", seed=seed, start=start))

            # Wohin das Budget geht -- ohne das ist nicht zu sehen, ob zwei
            # Verfahren beim selben erlaubten Budget auch fuer dasselbe zahlen.
            costs = results_io.budget_breakdown(part[part["graph"] == name])
            print("  ->", results_io.save_results(
                costs, name, kind="budget_breakdown", seed=seed, start=start))
            print(_budget_table(costs))


if __name__ == "__main__":
    main()
