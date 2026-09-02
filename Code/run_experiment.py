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

`--share-walks` laesst Estimators mit gleichem Oracle und Sampler *einen*
Walk teilen und nur die Auswertung variieren -- gedacht fuer Vergleiche von
Thinning, Safety Margin oder mit/ohne Gewichte, die alle auf derselben
Trajektorie sitzen sollten. Siehe check_shared.py.

Ergebnisse werden **angehaengt, nicht ueberschrieben**. Was schon gerechnet
wurde, laeuft nicht noch einmal; ein Aufruf mit zusaetzlichen Estimators oder
Budgets ergaenzt die Datei nur um das Fehlende. `--replace` erzwingt das
Neurechnen, `--deprecate` schiebt alles Vorhandene beiseite (fuer Aenderungen,
die den Verlauf aendern).

`--start-node` waehlt den Einstiegsknoten des Crawls (Default: der erste aus
config.SEED_NODES, bei den GPT-Basen "Vannevar Bush"); `--start-node all`
rechnet alle hinterlegten nacheinander. Jeder Einstieg landet in eigenen
Dateien -- verschiedene Einstiege sind verschiedene Bedingungen.
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
    p.add_argument("--graphs", nargs="+", default=None,
                   help="Graph-Namen oder Kuerzel wie "
                        f"{', '.join(sorted(config.GRAPH_ALIASES))} (Default: alle)")
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
    p.add_argument("--replace", action="store_true",
                   help="schon vorhandene Laeufe neu rechnen und ersetzen, statt "
                        "sie zu ueberspringen")
    p.add_argument("--deprecate", metavar="GRUND", nargs="?", const="",
                   help="alle vorhandenen Ergebnisse vorher nach "
                        "data/results/deprecated/<Zeit>__<Fingerabdruck>/ "
                        "verschieben. Fuer Aenderungen, die den Verlauf aendern "
                        "(Graphaufbau, Kostenmodell, Sampler) -- danach schreibt "
                        "der Lauf neue Dateien.")
    p.add_argument("--start-node", default=None, metavar="NAME",
                   help="Einstiegsknoten des Crawls. Default: der erste Eintrag "
                        "aus config.SEED_NODES (bei den GPT-Basen 'Vannevar "
                        "Bush'). 'all' rechnet alle hinterlegten Einstiegsknoten "
                        "nacheinander, jeder in eigene Dateien.")
    p.add_argument("--share-walks", action="store_true",
                   help="Estimators, die denselben Walk erzeugen wuerden (gleiches "
                        "Oracle und gleicher Sampler), teilen ihn: Thinning, "
                        "Weighting und Formel sind reine Nachbearbeitung. Exakt "
                        "dieselben Zahlen, deutlich weniger Arbeit -- und der "
                        "Vergleich zwischen den Varianten wird gepaart.")
    p.add_argument("--checkpoint-budgets", action="store_true",
                   help="alle Budgets aus einem Lauf ablesen statt je Budget einen "
                        "eigenen zu rechnen. Exakt dieselben Zahlen, spart "
                        "Sigma(Budgets)/max(Budget) an Rechenzeit -- die Punkte "
                        "eines Laufs sind danach aber genestet, nicht unabhaengig "
                        "(Spalte `nested`).")
    args = p.parse_args()

    if args.list:
        print("Graphen:")
        for name in loader.available_graphs():
            label = config.graph_label(name)
            alias = next((a for a, n in config.GRAPH_ALIASES.items() if n == name), "")
            print(f"    {name:<26}{alias:<12}{'' if label == name else label}")
        print("Estimators:", ", ".join(sorted(estimators.REGISTRY)))
        print("Views:     ", ", ".join(sorted(VIEWS)))
        return

    graph_names = ([config.resolve_graph(g) for g in args.graphs]
                   if args.graphs else loader.available_graphs())
    ests = estimators.build_all(args.estimators)
    code = provenance.code_fingerprint()

    if args.deprecate is not None:
        moved = results_io.deprecate(args.deprecate or None)
        print(f"  -> vorhandene Ergebnisse nach {moved}" if moved
              else "  -> keine Ergebnisse zum Verschieben da", flush=True)

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
        known = config.seed_nodes(name)
        if not known:
            starts = [None]          # kein Eintrag -> gleichverteilter Einstieg
            if args.start_node:
                raise SystemExit(f"Fuer {name} sind keine Einstiegsknoten hinterlegt "
                                 "(config.SEED_NODES)")
        elif args.start_node is None:
            starts = [known[0]]
        elif args.start_node.lower() == "all":
            starts = known
        else:
            match = [k for k in known if str(k).lower() == args.start_node.lower()]
            if not match:
                raise SystemExit(
                    f"{args.start_node!r} ist kein Einstiegsknoten von {name}. "
                    f"Moeglich: {', '.join(map(str, known))} oder 'all'")
            starts = match
        print(f"[{name}] Einstieg: {', '.join(map(str, starts))}"
              if starts != [None] else f"[{name}] Einstieg: gleichverteilt", flush=True)

        print(f"[{name}] {len(ests)} Estimators x {len(budgets)} Budgets x "
              f"{args.runs} Laeufe x {len(args.views)} Views = "
              f"{len(ests) * len(budgets) * args.runs * len(args.views)} Schaetzungen, "
              f"{args.jobs} Prozesse, Seed {args.seed}"
              + (f" x {len(starts)} Einstiegsknoten" if len(starts) > 1 else ""),
              flush=True)

        # Was schon in der Zieldatei steht, wird nicht neu gerechnet. Getrennt
        # je Einstiegsknoten, weil jeder in eine eigene Datei schreibt.
        skip: set = set()
        if not args.replace:
            for start in starts:
                have = results_io.load_one(name, seed=args.seed, start=start)
                skip |= results_io.run_keys(have)
        planned = {(v, e.name, b, r)
                   for v in args.views for e in ests
                   for b in budgets for r in range(args.runs)}
        todo = planned - skip
        if not todo:
            for start in starts:
                path = results_io._path(name, "estimates", args.seed, start)
                if path.exists():
                    print(f"[{name}] alles schon gerechnet -- {path}", flush=True)
            print(f"[{name}] nichts zu tun (mit --replace neu rechnen)", flush=True)
            loader.clear_cache()
            continue
        if skip & planned:
            print(f"[{name}] {len(skip & planned)} von {len(planned)} Schaetzungen "
                  f"liegen schon vor, gerechnet werden {len(todo)}", flush=True)

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
            share_walks=args.share_walks,
            start_nodes=starts,
            skip_keys=set() if args.replace else skip,
            code=code,
            log=lambda m: print(m, flush=True),
        )
        # Je Einstiegsknoten eine eigene Datei: verschiedene Einstiege sind
        # verschiedene Bedingungen und gehoeren nicht in dieselbe Spanne.
        for start in starts:
            part = df[df["start_node"] == start] if start is not None else df
            if part.empty:
                continue
            save = (results_io.save_results if args.replace
                    else results_io.append_results)
            print("  ->", save(part, name, seed=args.seed, start=start))
            if visits is not None:
                vpart = (visits[visits["start_node"] == start]
                         if start is not None else visits)
                print("  ->", save(vpart, name, kind="visits",
                                   seed=args.seed, start=start))
        loader.clear_cache()

    # Ordner-README aktuell halten -- sonst steht sie nach dem naechsten Lauf falsch da
    for path in provenance.write_readmes():
        print("  ->", path)


if __name__ == "__main__":
    main()
