"""CLI: pruefen, dass DUFS mit k = 1 exakt DURW ist.

DUFS ist DURW mit k Walkern (sampling.dufs). Bei k = 1 muss davon nichts
uebrig bleiben: dieselbe Trajektorie, dieselben Kosten, derselbe Schaetzwert
-- bitgleich, nicht nur der Verteilung nach. Das ist die schaerfste Aussage,
die sich ueber den neuen Sampler treffen laesst, und sie prueft alles auf
einmal: die Reihenfolge der Buchungen (DUFS fragt die Nachbarn bei der
Ankunft, DURW am Schleifenkopf), das Einfrieren von G_u, die Absprungregel und
vor allem den Zufallsstrom -- bei k = 1 darf die Walker-Auswahl keinen
einzigen Zufallswert verbrauchen.

Geprueft werden Paare (DURW-Name, DUFS-Name) ueber alle Views und Budgets:

    wis-durw__uniform__margin        <-> wis-dufs__uniform__k1__margin
    wis-durw__nojump__margin         <-> wis-dufs__uniform__k1__nojump__margin *
    durw-<quelle>__b0__margin        <-> dufs-<quelle>__k1__b0__margin
    durwunion-<quelle>__b0__margin   <-> dufsunion-<quelle>__k1__b0__margin
    durw-rand<P>__b0__margin         <-> dufs-rand<P>__k1__b0__margin

    * Das eine Paar, das *nicht* gleich sein darf und deshalb nur informativ
      gemeldet wird: DURW ohne Sprung laeuft auf dem CrawlOracle und startet
      bei config.SEED_NODES, DUFS ohne Sprung braucht das Sprung-Oracle fuer
      seine Startknoten (s. estimators/methods/dufs.py). Verschiedene
      Startziehung heisst verschiedener Zufallsstrom.

Verglichen wird die volle Trajektorie Sample fuer Sample (Knoten, deg_Gu,
Schritt, Sprungflagge), dazu Kosten, Abbruchgrund und Schaetzwert per `repr`,
also ohne Toleranz.

Zusaetzlich (--sweep) eine kurze Uebersicht ueber wachsendes k auf demselben
Seed: sie beweist nichts, zeigt aber sofort, ob die Walker-Auswahl ueberhaupt
etwas bewirkt.

Beispiele:
    python check_dufs.py --graph Slashdot0811
    python check_dufs.py --graph gpt4_io --views directed --sweep
"""

from __future__ import annotations

import argparse
import random

import config
import estimators as estimator_registry
import namelists
from graphs import loader
from graphs.views import VIEWS, build_view

FIELDS = ("queries", "unique_nodes", "cached_queries", "n_random_node",
          "n_neighbors", "stopped_by")


def pairs(graph_name: str) -> list[tuple[str, str]]:
    """Die zu vergleichenden (DURW, DUFS)-Paare fuer diesen Graphen."""
    out = [("wis-durw__uniform__margin", "wis-dufs__uniform__k1__margin"),
           ("durw-plain__uniform__none", "dufs-plain__uniform__k1__none")]
    for p in config.JUMP_SUBSET_PERCENTS:
        out.append((f"durw-rand{p:g}__b0__margin", f"dufs-rand{p:g}__k1__b0__margin"))
        out.append((f"durwunion-rand{p:g}__b0__margin",
                    f"dufsunion-rand{p:g}__k1__b0__margin"))
    for src in sorted(namelists.SOURCES):
        out.append((f"durw-{src}__b0__margin", f"dufs-{src}__k1__b0__margin"))
        out.append((f"durwunion-{src}__b0__margin", f"dufsunion-{src}__k1__b0__margin"))
    return out


def _sample_tuple(s):
    return (s.node, s.degree, s.step, s.walk, s.in_jump_set, s.jumped,
            s.sigma_weight, s.walker)


def compare(view, durw_name: str, dufs_name: str, budget: int, seed: str
            ) -> list[str]:
    """Abweichungen als Textzeilen; leere Liste heisst bitgleich."""
    a = estimator_registry.build(durw_name)
    b = estimator_registry.build(dufs_name)
    trace_a, oracle_a = a.run_walk(view, budget, random.Random(seed))
    trace_b, oracle_b = b.run_walk(view, budget, random.Random(seed))

    problems = []
    if len(trace_a) != len(trace_b):
        problems.append(f"Laenge der Trajektorie: {len(trace_a)} != {len(trace_b)}")
    for i, (x, y) in enumerate(zip(trace_a, trace_b)):
        if _sample_tuple(x) != _sample_tuple(y):
            problems.append(f"Sample {i}: {_sample_tuple(x)} != {_sample_tuple(y)}")
            break   # ab der ersten Abweichung sagt der Rest nichts mehr
    cost_a, cost_b = oracle_a.cost(), oracle_b.cost()
    problems += [f"{k}: {cost_a.get(k)!r} != {cost_b.get(k)!r}"
                 for k in FIELDS if cost_a.get(k) != cost_b.get(k)]
    va = a.evaluate(trace_a, cost_a, oracle_a.visits).value
    vb = b.evaluate(trace_b, cost_b, oracle_b.visits).value
    if repr(va) != repr(vb):
        problems.append(f"estimate: {va!r} != {vb!r}")
    return problems


def sweep(view, budget: int, seed: int, jump: str) -> None:
    """Schaetzung ueber wachsendes k -- kein Beweis, nur ein Blick."""
    print(f"\n  k-Sweep, jump={jump}, Budget {budget}:")
    for k in config.DUFS_WALKER_COUNTS:
        name = (f"wis-dufs__uniform__k{k}__margin" if jump == "uniform"
                else f"dufs-{jump}__k{k}__b0__margin")
        est = estimator_registry.build(name)
        res = est.estimate(view, budget, random.Random(f"{seed}|{name}"))
        rel = res.value / view.n_nodes if res.value == res.value else float("nan")
        print(f"    k={k:<5d} {name:42s} n_hat/|V| = {rel:8.4f}  "
              f"({res.extra['n_samples']} Samples, "
              f"{res.cost['n_random_node']} Ziehungen)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", default="Slashdot0811")
    p.add_argument("--views", nargs="+", default=list(config.DEFAULT_VIEWS),
                   choices=sorted(VIEWS))
    p.add_argument("--budgets", nargs="+", type=float, default=[0.001, 0.01])
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[config.DEFAULT_SEED, config.DEFAULT_SEED + 1])
    p.add_argument("--sweep", action="store_true",
                   help="zusaetzlich eine k-Uebersicht ausgeben")
    args = p.parse_args()

    graph = loader.load_graph(args.graph)
    todo, skipped = estimator_registry.applicable(
        [estimator_registry.build(n) for pair in pairs(graph.name) for n in pair],
        graph.name)
    ok_names = {e.name for e in todo}
    checks = [(x, y) for x, y in pairs(graph.name)
              if x in ok_names and y in ok_names]
    if skipped:
        print(f"  ({len(set(skipped))} auf diesem Graphen nicht anwendbar, "
              f"uebersprungen)")

    failed = 0
    for view_name in args.views:
        view = build_view(graph, view_name)
        budgets = sorted({max(int(round(b * view.n_nodes)), 2) for b in args.budgets})
        print(f"\n[{args.graph}/{view_name}] {len(checks)} Paare, "
              f"{len(budgets)} Budgets, {len(args.seeds)} Seeds")
        for durw_name, dufs_name in checks:
            bad = []
            try:
                for seed in args.seeds:
                    for budget in budgets:
                        # Derselbe Seed-String fuer beide -- der Runner leitet
                        # ihn sonst aus dem Estimator-Namen ab, und die sind ja
                        # verschieden.
                        bad += [f"      seed {seed} @ {budget}: {line}" for line
                                in compare(view, durw_name, dufs_name, budget,
                                           f"{seed}|check_dufs")]
            except FileNotFoundError as exc:
                # Namensindex fuer diesen Graphen nicht gebaut -- kein Fehler
                # dieses Skripts (s. build_name_index.py).
                print(f"  uebersprungen  {durw_name}: {str(exc).splitlines()[0]}")
                continue
            failed += bool(bad)
            print(f"  {'ABWEICHUNG' if bad else 'bitgleich '}  "
                  f"{durw_name}  <->  {dufs_name}")
            for line in bad[:10]:
                print(line)
        if args.sweep:
            sweep(view, max(budgets), args.seeds[0], "uniform")

    print(f"\n-> {'ALLES BITGLEICH' if not failed else f'{failed} Paare weichen ab'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
