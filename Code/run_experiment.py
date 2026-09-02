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


def _default_budgets(n_nodes: int, name: str, log=print) -> list[float]:
    large = n_nodes >= config.LARGE_GRAPH_NODES
    if large:
        log(f"[{name}] grosser Graph -- 20-%-Budget ausgelassen "
            f"(mit --budgets erzwingbar)")
    return list(config.DEFAULT_BUDGETS_LARGE if large else config.DEFAULT_BUDGETS)


def _known_size(graph_name: str) -> int | None:
    """|V| aus vorhandenen Ergebnissen -- ohne den Graphen zu laden.

    Gebraucht nur, um die Default-Budgets zu waehlen (gross/klein). Das Laden
    dauert bei den grossen Wissensgraphen ueber eine Minute und soll erst
    passieren, wenn feststeht, dass ueberhaupt zu rechnen ist. Gibt es noch
    keine Ergebnisse, ist ohnehin zu rechnen -- dann liefert das hier None und
    der Graph wird geladen.

    |V| haengt nicht an Seed, Einstieg oder View, deshalb genuegt die erste
    beste Ergebnisdatei dieses Graphen; gelesen wird nur ihre erste Zeile.
    """
    import pandas as pd

    for path in sorted(config.RESULTS_DIR.glob(f"{graph_name}__*estimates.csv")):
        try:
            head = pd.read_csv(path, nrows=1)
        except Exception:                                     # noqa: BLE001
            continue
        if "true_size" in head.columns and len(head):
            return int(head["true_size"].iloc[0])
    return None


def _resolve_starts(name: str, args) -> list:
    """Einstiegsknoten des Laufs -- braucht den Graphen nicht."""
    known = config.seed_nodes(name)
    if not known:
        if args.start_node:
            raise SystemExit(f"Fuer {name} sind keine Einstiegsknoten hinterlegt "
                             "(config.SEED_NODES)")
        return [None]                # gleichverteilter Einstieg
    if args.start_node is None:
        return [known[0]]
    if args.start_node.lower() == "all":
        return known
    match = [k for k in known if str(k).lower() == args.start_node.lower()]
    if not match:
        raise SystemExit(
            f"{args.start_node!r} ist kein Einstiegsknoten von {name}. "
            f"Moeglich: {', '.join(map(str, known))} oder 'all'")
    return match


def _existing_runs(name: str, seed: int, starts: list) -> set:
    """Welche (View, Estimator, Budget, Lauf) schon in den Zieldateien stehen."""
    have: set = set()
    for start in starts:
        have |= results_io.run_keys(
            results_io.load_one(name, seed=seed, start=start))
    return have


def _report_done(name: str, seed: int, starts: list) -> None:
    for start in starts:
        path = results_io._path(name, "estimates", seed, start)
        if path.exists():
            print(f"[{name}] alles schon gerechnet -- {path}", flush=True)
    print(f"[{name}] nichts zu tun, Graph wird nicht geladen "
          "(mit --replace neu rechnen)", flush=True)


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
        starts = _resolve_starts(name, args)
        skip = set() if args.replace else _existing_runs(name, args.seed, starts)
        planned_for = lambda bs: {(v, e.name, b, r)             # noqa: E731
                                  for v in args.views for e in ests
                                  for b in bs for r in range(args.runs)}

        # Reihenfolge mit Absicht: erst pruefen, dann laden. Das Laden dauert
        # bei den grossen Wissensgraphen ueber eine Minute und lohnt nicht,
        # wenn alles schon gerechnet ist. Die Default-Budgets brauchen |V| --
        # das steht in den vorhandenen Ergebnissen, sonst ist ohnehin zu tun.
        stored_size = _known_size(name)
        budgets = args.budgets or (None if stored_size is None
                                   else _default_budgets(stored_size, name))
        if budgets is not None and not planned_for(budgets) - skip:
            _report_done(name, args.seed, starts)
            continue

        t0 = time.perf_counter()
        graph = loader.load_graph(name)
        without = graph.n_nodes - graph.n_with_out_edges
        print(f"[{name}] geladen in {time.perf_counter() - t0:.1f}s -- "
              f"|V| = {graph.n_nodes:,} ({graph.n_with_out_edges:,} mit ausgehenden "
              f"Kanten, {without:,} ohne), {graph.n_edges:,} Kanten".replace(",", " "),
              flush=True)

        if stored_size is not None and stored_size != graph.n_nodes:
            raise SystemExit(
                f"[{name}] Die vorhandenen Ergebnisse wurden auf einem Graphen "
                f"mit |V| = {stored_size:,} gerechnet, der Graph hat jetzt "
                f"{graph.n_nodes:,}. Die Zahlen sind nicht vergleichbar -- "
                "alte Ergebnisse mit --deprecate beiseiteschieben."
                .replace(",", " ")
            )
        if budgets is None:
            budgets = _default_budgets(graph.n_nodes, name)
            if not planned_for(budgets) - skip:
                _report_done(name, args.seed, starts)
                loader.clear_cache()
                continue

        print(f"[{name}] Einstieg: {', '.join(map(str, starts))}"
              if starts != [None] else f"[{name}] Einstieg: gleichverteilt", flush=True)

        print(f"[{name}] {len(ests)} Estimators x {len(budgets)} Budgets x "
              f"{args.runs} Laeufe x {len(args.views)} Views = "
              f"{len(ests) * len(budgets) * args.runs * len(args.views)} Schaetzungen, "
              f"{args.jobs} Prozesse, Seed {args.seed}"
              + (f" x {len(starts)} Einstiegsknoten" if len(starts) > 1 else ""),
              flush=True)

        planned = planned_for(budgets)
        if skip & planned:
            print(f"[{name}] {len(skip & planned)} von {len(planned)} Schaetzungen "
                  f"liegen schon vor, gerechnet werden {len(planned - skip)}",
                  flush=True)

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
