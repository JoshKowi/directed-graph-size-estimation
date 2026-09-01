"""CLI: pruefen, dass geteilte Walks exakt dasselbe liefern wie Einzellaeufe.

`--share-walks` steht und faellt mit einer Behauptung: Thinning, Weighting und
Formel kommen *nach* dem Sampler, aendern also nichts an der Trajektorie.
Estimators mit gleichem `walk_key` duerfen sich deshalb einen Walk teilen.

Diese Datei rechnet das nach, statt es zu glauben: je Gruppe einmal
`estimate_group(...)` und einmal `est.estimate(...)` fuer jedes Mitglied --
mit demselben Seed-String -- und vergleicht Schaetzwert, Kosten, Abbruchgrund
und Sample-Zahl auf Gleichheit (`repr`, keine Toleranz). Eine Abweichung
heisst, dass doch etwas vor dem Sampler von der Auswertung abhaengt; dann ist
der Modus kaputt und nicht bloss ungenau.

Beispiele:
    python check_shared.py --graph slashdot
    python check_shared.py --graph gpt-4-io --views undirected
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

import config
import estimators as estimator_registry
from estimators.pipeline import estimate_group
from graphs import loader
from graphs.views import VIEWS, build_view

FIELDS = ("queries", "unique_nodes", "cached_queries", "n_random_node",
          "n_neighbors", "stopped_by")


def compare(view, group, budgets, seed: str) -> list[str]:
    """Abweichungen als Textzeilen; leere Liste heisst identisch."""
    shared = estimate_group(group, view, budgets, random.Random(seed))
    problems = []
    for est in group:
        for b in budgets:
            a = shared[(est.name, b)]
            c = est.estimate(view, b, random.Random(seed))
            diff = [f"{k}: {a.cost.get(k)!r} != {c.cost.get(k)!r}"
                    for k in FIELDS if a.cost.get(k) != c.cost.get(k)]
            if repr(a.value) != repr(c.value):
                diff.append(f"estimate: {a.value!r} != {c.value!r}")
            if a.extra.get("n_samples") != c.extra.get("n_samples"):
                diff.append(f"n_samples: {a.extra.get('n_samples')} != "
                            f"{c.extra.get('n_samples')}")
            if diff:
                problems.append(f"    {est.name} @ {b}: " + "; ".join(diff))
    return problems


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", default="Slashdot0811")
    p.add_argument("--views", nargs="+", default=list(config.DEFAULT_VIEWS),
                   choices=sorted(VIEWS))
    p.add_argument("--estimators", nargs="+", default=None,
                   help="Default: alle mit walk_key")
    p.add_argument("--budgets", nargs="+", type=float, default=[0.001, 0.01])
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    args = p.parse_args()

    graph = loader.load_graph(args.graph)
    groups: dict[str, list] = defaultdict(list)
    for est in estimator_registry.build_all(args.estimators):
        if hasattr(est, "walk_key"):
            groups[est.walk_key].append(est)
    failed = 0

    for view_name in args.views:
        view = build_view(graph, view_name)
        budgets = sorted({max(int(round(b * view.n_nodes)), 2) for b in args.budgets})
        print(f"\n[{args.graph}/{view_name}] {len(groups)} Walk-Gruppen, "
              f"{len(budgets)} Budgets, Seed {args.seed}")
        for key, group in sorted(groups.items()):
            problems = compare(view, group, budgets, f"{args.seed}|{key}")
            failed += bool(problems)
            print(f"  {'ABWEICHUNG' if problems else 'identisch '}  {key}")
            print(f"      {len(group)} Varianten: {', '.join(e.name for e in group)}")
            for line in problems:
                print(line)

    print(f"\n-> {'ALLES IDENTISCH' if not failed else f'{failed} Gruppen weichen ab'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
