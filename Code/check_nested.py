"""CLI: pruefen, dass genestete Budgets exakt dasselbe liefern wie einzelne Laeufe.

Der ganze Sinn von `--checkpoint-budgets` haengt an einer Behauptung: ein Lauf
mit Budget B, abgeschnitten bei Kosten b, ist *bitgleich* mit einem
eigenstaendigen Lauf mit Budget b und demselben Zufallsstrom. Das gilt, weil
kein Sampler sein Budget kennt -- es steuert nur den Abbruch (siehe
oracles/base.py). Diese Datei rechnet die Behauptung nach, statt sie zu
glauben: je Estimator und Budget werden beide Wege mit identischem Seed
gerechnet und Schaetzwert, Kosten, Abbruchgrund und Sample-Zahl verglichen.

Verglichen wird mit `repr`, nicht mit einer Toleranz: gefordert ist
Gleichheit, nicht Aehnlichkeit. Eine Abweichung heisst, dass irgendwo doch
Budget-Wissen in die Ziehung geflossen ist -- dann ist der Modus kaputt und
nicht bloss ungenau.

Beispiele:
    python check_nested.py --graph Slashdot0811
    python check_nested.py --graph Slashdot0811 --views directed --estimators uis__walk5
"""

from __future__ import annotations

import argparse
import random

import config
import estimators as estimator_registry
from graphs import loader
from graphs.views import VIEWS, build_view

# Felder, die uebereinstimmen muessen. Der Schaetzwert allein wuerde ein
# verschobenes Abbruchkriterium nicht zwingend auffallen lassen.
FIELDS = ("queries", "unique_nodes", "cached_queries", "n_random_node",
          "n_neighbors", "stopped_by")


def compare(view, est, budgets, seed: str) -> list[str]:
    """Abweichungen als Textzeilen; leere Liste heisst identisch."""
    nested = est.estimate_nested(view, budgets, random.Random(seed))
    problems = []
    for b in budgets:
        single = est.estimate(view, b, random.Random(seed))
        a, c = nested[b], single
        diff = [f"{k}: {a.cost.get(k)!r} != {c.cost.get(k)!r}"
                for k in FIELDS if a.cost.get(k) != c.cost.get(k)]
        if repr(a.value) != repr(c.value):
            diff.append(f"estimate: {a.value!r} != {c.value!r}")
        if a.extra.get("n_samples") != c.extra.get("n_samples"):
            diff.append(f"n_samples: {a.extra.get('n_samples')} != "
                        f"{c.extra.get('n_samples')}")
        if diff:
            problems.append(f"    Budget {b}: " + "; ".join(diff))
    return problems


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", default="Slashdot0811")
    p.add_argument("--views", nargs="+", default=list(config.DEFAULT_VIEWS),
                   choices=sorted(VIEWS))
    p.add_argument("--estimators", nargs="+", default=None,
                   help="Default: alle mit estimate_nested")
    p.add_argument("--budgets", nargs="+", type=float, default=[0.001, 0.005, 0.01])
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    args = p.parse_args()

    graph = loader.load_graph(args.graph)
    # capture_recapture bringt zwar estimate_nested mit (es ist eine Pipeline),
    # darf es aber nicht benutzen -- siehe supports_nested.
    ests = [e for e in estimator_registry.build_all(args.estimators)
            if hasattr(e, "estimate_nested") and getattr(e, "supports_nested", False)]
    failed = 0

    for view_name in args.views:
        view = build_view(graph, view_name)
        budgets = sorted({max(int(round(b * view.n_nodes)), 2) for b in args.budgets})
        print(f"\n[{args.graph}/{view_name}] {len(ests)} Estimators x "
              f"{len(budgets)} Budgets, Seed {args.seed}")
        for est in ests:
            problems = compare(view, est, budgets, f"{args.seed}|{est.name}|{view_name}")
            failed += bool(problems)
            print(f"  {'ABWEICHUNG' if problems else 'identisch '}  {est.name}")
            for line in problems:
                print(line)

    print(f"\n-> {'ALLES IDENTISCH' if not failed else f'{failed} Estimators weichen ab'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
