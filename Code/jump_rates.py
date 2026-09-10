"""CLI: Sprunganteil als Tabelle -- Parameter x Budget.

Der Anteil ist n_random_node/extra_n_samples: bei DURW die Spruenge, bei
oracles.name_list die erfolgreichen Ziehungen. Er ist die Groesse, an der
haengt, wie schnell DURW mischt, und damit der Grund, warum derselbe Schaetzer
auf gpt4o_io schon bei 0,5 % Budget trifft und auf gpt4_io bei 10 % noch nicht
(28 % gegen 15 %).

Ausgegeben wird auf die Konsole; mit --csv landet dieselbe Tabelle zusaetzlich
unter data/plots/<graph>__<seed/start>jump-rates[-<index>].csv.

`--index` waehlt die Zeilen: "estimator" (Default) oder der Parameter aus dem
Namen -- `w` (Sprunggewicht), `n` (Listenlaenge) oder `b` (Burn-in). Bei einem
Parameter vorher auf *eine* Familie einschraenken, sonst mitteln sich zwei
Reihen zu einer: `--match uniform__w` zeigt den w-Sweep, `--match durw-top-q__n`
die Listenlaengen.

Beispiele:
    python jump_rates.py --graphs gpt4_io --match uniform__w --index w
    python jump_rates.py --graphs gpt4_io gpt4o_io --views directed --index w \\
        --match uniform__w --csv
    python jump_rates.py --graphs gpt4o_io --match durw-top-q__n --index n
"""

from __future__ import annotations

import argparse

import config
from experiment import results as results_io
from graphs.views import VIEWS


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graphs", nargs="+", default=None,
                   help="Graph-Namen oder Kuerzel (Default: alle vorhandenen)")
    p.add_argument("--estimators", nargs="+", default=None,
                   help="exakte Estimator-Namen")
    p.add_argument("--match", nargs="+", default=None,
                   help="Estimators, deren Name einen dieser Teilstrings enthaelt")
    p.add_argument("--views", nargs="+", default=None, choices=sorted(VIEWS),
                   help="Kantensichten (Default: alle vorhandenen, je einzeln)")
    p.add_argument("--index", default="estimator",
                   choices=["estimator", *sorted(results_io.PARAM_LABELS)],
                   help="Zeilen: Estimator-Name oder ein Parameter daraus")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--start-node", default=None)
    p.add_argument("--csv", action="store_true",
                   help=f"Tabelle zusaetzlich nach {config.PLOTS_DIR} schreiben")
    args = p.parse_args()

    df = results_io.load_results(seed=args.seed, start=args.start_node)
    if df.empty:
        raise SystemExit(f"Keine Ergebnisse in {config.RESULTS_DIR}")
    if args.estimators:
        df = df[df["estimator"].isin(args.estimators)]
    if args.match:
        df = df[df["estimator"].str.contains("|".join(args.match), regex=True)]
    if df.empty:
        raise SystemExit("Kein Estimator passt auf die Auswahl")

    graphs = ([config.resolve_graph(g) for g in args.graphs] if args.graphs
              else sorted(df["graph"].unique()))
    wrote = False
    for graph in graphs:
        part = df[df["graph"] == graph]
        if part.empty:
            print(f"[{graph}] keine Zeilen -- uebersprungen")
            continue
        # Je Sicht eine eigene Tabelle: directed und undirected springen
        # verschieden oft, gemittelt saehe man von beidem nichts.
        views = args.views or [v for v in VIEWS if (part["view"] == v).any()]
        for view in views:
            table = results_io.jump_rate_table(part, index=args.index, view=view)
            if table.empty:
                continue
            head = (f"=== {config.graph_label(graph)} / {view} "
                    f"-- Sprunganteil in % (Spalten: Budget) ===")
            print(f"\n{head}")
            print(table.to_string())
            if args.csv:
                config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
                tag = (results_io.seed_tag(args.seed)
                       + results_io.start_tag(graph, args.start_node))
                name = (f"{graph}__{tag}jump-rates-{args.index}-{view}.csv")
                path = config.unique_path(config.PLOTS_DIR / name)
                table.to_csv(path)
                print(f"  -> {path}")
                wrote = True
    if args.csv and not wrote:
        print("\nNichts geschrieben -- keine Tabelle hatte Zeilen.")


if __name__ == "__main__":
    main()
