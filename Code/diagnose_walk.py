"""CLI: einen Random Walk sezieren -- Sackgassen oder falsche Gewichte?

Warum schaetzt Collision Counting auf gerichteten Graphen so weit daneben?
Zwei Erklaerungen sind moeglich, und sie verlangen verschiedene Konsequenzen:

    H1  Der Walk bleibt in einem Gebiet stecken ("Sinks"). Dann gibt es eine
        harte Decke -- Knoten ausserhalb sind unerreichbar, kein Gewicht hilft.
    H2  Die Gewichte passen nicht. Besucht werden Knoten mit vielen
        *Eingangs*kanten, korrigiert wird aber mit 1/deg_out.

Getrennt werden sie ueber eine Leiter von Groessen, die alle dasselbe messen
sollen -- die Zahl der Knoten -- und an der man abliest, wo es abreisst:

    |V|                     Wahrheit
    erreichbar vom Seed     BFS: was der Walk am Anfang sehen koennte
    erreichbar vom Endknoten was er, dort angekommen, noch erreichen kann -> H1
    verschieden besucht     was er gesehen hat
    C(k,2)/n_col            Kollisionszaehlung ohne Gewichte
    n_hat mit 1/deg_out     die eigentliche Schaetzung                 -> H2

C(k,2)/n_col schaetzt im Erwartungswert genau 1/sum(pi^2), den "effektiven
Traeger" -- ueber wie viele Knoten sich der Walk effektiv verteilt, wenn man
die Konzentration herausrechnet. Beide Groessen getrennt auszuweisen waere
doppelt gemoppelt, deshalb steht hier nur die geschaetzte Variante.

Ein kleiner effektiver Traeger ist fuer sich *kein* Fehler: auch ein perfekter
Walk auf einem ungerichteten Graphen konzentriert sich auf Hubs -- genau dafuer
sind die Gewichte da. Der Korrekturfaktor L = mean(w)*mean(1/w) soll
C(k,2)/n_col wieder auf |V| hochziehen. Bleibt der beobachtete Hub weit unter
dem noetigen |V| / (C(k,2)/n_col), liegt es an den Gewichten.

Der Walk haengt an einem einzigen Zufallsstrom -- welcher Startknoten gezogen
wird, entscheidet auf einem gerichteten Graphen mit darueber, in welcher Senke
er landet. Mit --seed laesst sich derselbe Graph mehrfach diagnostizieren; der
Seed steht im Bericht, auf der Grafik und (ausser beim Default) im Dateinamen.

Beispiele:
    python diagnose_walk.py --graph gpt4o_io --views directed undirected
    python diagnose_walk.py --graph Slashdot0811 --dead-end backtrack --budget 0.01
    python diagnose_walk.py --graph gpt4o_io --seed 7
    python diagnose_walk.py --graph gpt4_io --connectivity --jump-weight 10
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.stats import spearmanr

import config
from graphs import loader
from graphs.views import build_view
from oracles.local_access import CrawlOracle, JumpCrawlOracle
from sampling.dead_ends import DEAD_ENDS
from sampling.durw import DurwSampler
from sampling.samplers import RandomWalkSampler


class _TracingOracle(CrawlOracle):
    """CrawlOracle, das nebenher die Abdeckungskurve mitschreibt.

    Aufgezeichnet wird an logarithmisch verteilten Schritten, wie viele
    *verschiedene* Knoten bis dahin beruehrt wurden. Der Speicher bleibt damit
    bei der Zahl der Stuetzstellen, nicht bei der Zahl der Schritte.
    """

    def __init__(self, *args, checkpoints=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = 0
        self.seed_node = None
        self._checkpoints = [] if checkpoints is None else list(checkpoints)
        self._next = 0
        self.coverage: list[tuple[int, int]] = []

    def _charge(self, node, amount):
        super()._charge(node, amount)
        if self.seed_node is None:
            self.seed_node = node

    def _fetch(self, u):
        out = super()._fetch(u)
        self.steps += 1
        while self._next < len(self._checkpoints) and self.steps >= self._checkpoints[self._next]:
            self.coverage.append((self.steps, len(self.visits)))
            self._next += 1
        return out


def _degrees(view, base):
    """Aus- und Eingangsgrad je Knoten in der *gerichteten* Originalsicht."""
    out_deg = np.diff(base.indptr)
    in_deg = np.bincount(base.indices, minlength=base.n_nodes)
    return out_deg, in_deg


def diagnose(graph, view_name, dead_end="history", budget_rel=0.01, top=15,
             seed: int = config.DEFAULT_SEED):
    base = build_view(graph, "directed")          # Grade immer aus dem Original
    view = build_view(graph, view_name)
    n = view.n_nodes
    budget = max(int(round(budget_rel * n)), 2)

    checkpoints = np.unique(np.geomspace(1, budget * 60, 220).astype(np.int64))
    # Der Seed haengt nicht an der View: directed und undirected starten damit
    # beim selben Knoten und sind gepaart vergleichbar.
    oracle = _TracingOracle(view, random.Random(f"diag|{seed}|{dead_end}"),
                            budget, config.DEFAULT_BUDGET_METRIC,
                            checkpoints=checkpoints)
    sampler = RandomWalkSampler(dead_end=DEAD_ENDS[dead_end]())
    trace = sampler.sample(oracle)

    counts = np.array(list(oracle.visits.values()), dtype=np.float64)
    nodes = np.array(list(oracle.visits.keys()), dtype=np.int64)
    k = len(trace)
    pi = counts / counts.sum()

    n_col = float(np.sum(counts * (counts - 1) / 2))
    # schaetzt im Erwartungswert 1/sum(pi^2), den effektiven Traeger
    uis = (k * (k - 1) / 2) / n_col if n_col else float("nan")

    deg = np.array([view.degree(int(u)) for u in nodes], dtype=np.float64)
    w = 1.0 / np.maximum(deg, 1.0)   # Gewicht = 1/Grad in der benutzten Sicht
    # Gewichte je *Sample*, nicht je Knoten -> mit der Besuchszahl wichten
    lift = (np.sum(counts * w) / k) * (np.sum(counts / w) / k)
    wis = uis * lift

    # H1 hat zwei Seiten, und nur zusammen sind sie aussagekraeftig:
    #   vom Seed aus -- was der Walk am Anfang haette erreichen koennen
    #   vom Endknoten -- was er, dort angekommen, noch erreichen kann
    # Faellt der zweite Wert auf eine Handvoll, sitzt der Walk in einem
    # terminalen Zyklus fest; dann ist die Schaetzung strukturell gedeckelt
    # und kein Gewicht der Welt hilft.
    mat = csr_matrix((np.ones(view.n_edges, dtype=np.int8), view.indices, view.indptr),
                     shape=(n, n))
    reach = len(breadth_first_order(mat, oracle.seed_node, directed=True,
                                    return_predecessors=False))
    reach_end = len(breadth_first_order(mat, trace[-1].node, directed=True,
                                        return_predecessors=False)) if trace else 0

    out_deg, in_deg = _degrees(view, base)
    rho_out = spearmanr(counts, out_deg[nodes]).statistic
    rho_in = spearmanr(counts, in_deg[nodes]).statistic
    rho_view = spearmanr(counts, deg).statistic
    # Auf der symmetrisierten Sicht gibt es keinen Aus-/Eingangsgrad mehr --
    # dort ist nur der (eine) Grad der View aussagekraeftig.
    symmetric = view_name == "undirected"

    order = np.argsort(counts)[::-1][:top]
    top_rows = [
        {"name": graph.name_of(int(nodes[i])), "visits": int(counts[i]),
         "share": counts[i] / k, "deg_out": int(out_deg[nodes[i]]),
         "deg_in": int(in_deg[nodes[i]])}
        for i in order
    ]

    return {
        "graph": graph.name, "view": view_name, "dead_end": dead_end, "seed": seed,
        "budget_rel": budget_rel, "budget_abs": budget, "steps": k,
        "n_nodes": n, "reachable": reach, "reachable_end": reach_end,
        "distinct": len(nodes),
        "uis": uis, "wis": wis, "lift": lift,
        "lift_needed": n / uis if uis else float("nan"),
        "rho_out": rho_out, "rho_in": rho_in, "rho_view": rho_view,
        "deg_view": deg, "symmetric": symmetric,
        "coverage": oracle.coverage, "counts": counts, "nodes": nodes,
        "out_deg": out_deg, "in_deg": in_deg, "top": top_rows,
    }


def print_report(d):
    n = d["n_nodes"]
    print(f"\n=== {d['graph']} / {d['view']} / dead_end={d['dead_end']} "
          f"/ seed={d['seed']} ===")
    print(f"Budget {d['budget_rel']:g} = {d['budget_abs']:,} Einheiten, "
          f"{d['steps']:,} Schritte\n".replace(",", " "))
    rows = [
        ("|V| (Wahrheit)", n),
        ("erreichbar vom Seed", d["reachable"]),
        ("erreichbar vom Endknoten", d["reachable_end"]),
        ("verschieden besucht", d["distinct"]),
        ("Schaetzung ohne Gewichte C(k,2)/n_col", d["uis"]),
        ("Schaetzung mit 1/deg_out", d["wis"]),
    ]
    print(f"{'Groesse':40}{'Wert':>16}{'Anteil |V|':>13}")
    for label, val in rows:
        print(f"{label:40}{val:>16,.0f}{val / n:>13.5f}".replace(",", " "))
    verdict = ("Sackgasse: der Walk sitzt fest, die Schaetzung ist strukturell "
               "gedeckelt" if d["reachable_end"] < 0.01 * n else
               "keine Sackgasse: vom Endknoten aus ist der Graph noch offen")
    print(f"\n-> {verdict}")
    print(f"Korrekturfaktor L = mean(w)*mean(1/w): beobachtet {d['lift']:.1f}, "
          f"noetig waeren {d['lift_needed']:.1f}")
    if d["symmetric"]:
        print(f"Spearman(Besuche, Grad): {d['rho_view']:+.3f} "
              "(symmetrisiert -- nur ein Grad)")
    else:
        print(f"Spearman(Besuche, Grad):  Ausgangsgrad {d['rho_out']:+.3f}   "
              f"Eingangsgrad {d['rho_in']:+.3f}")
    print(f"\nMeistbesuchte Knoten:")
    print(f"  {'Entitaet':<46}{'Besuche':>10}{'Anteil':>9}{'deg_out':>9}{'deg_in':>9}")
    for r in d["top"]:
        name = str(r["name"])
        if len(name) > 44:
            name = f"{name[:21].rstrip()}\u2026{name[-21:].lstrip()}"
        print(f"  {name:<46}{r['visits']:>10,}{r['share']:>9.2%}"
              f"{r['deg_out']:>9}{r['deg_in']:>9}".replace(",", " "))


def diagnose_connectivity(graph, top: int = 10):
    """Fragmentierung des *gerichteten* Graphen -- unabhaengig von jedem Walk.

    Zwei Zerlegungen derselben Kantenmenge:
        SCC (stark)  -- u und v gegenseitig erreichbar. Der Walk kann eine
                        SCC nie verlassen und wieder betreten; eine kleine
                        SCC am Ende eines Laufs ist die strukturelle Decke
                        hinter `reachable_end` in diagnose().
        WCC (schwach) -- u und v verbunden, wenn man Kantenrichtung ignoriert.
                        Obergrenze dafuer, was ein ungerichtetes G_u (DURW)
                        ueberhaupt je verbinden koennte: mehr WCCs als 1
                        heisst, selbst *alle* Kanten zusammen reichen nicht.
    Beides ist O(|E|) ueber scipy, kein BFS je Seed noetig.
    """
    view = build_view(graph, "directed")
    n = view.n_nodes
    mat = csr_matrix((np.ones(view.n_edges, dtype=np.int8), view.indices, view.indptr),
                     shape=(n, n))
    n_scc, labels = connected_components(mat, directed=True, connection="strong")
    n_wcc, _ = connected_components(mat, directed=True, connection="weak")

    sizes = np.bincount(labels)
    order = np.argsort(sizes)[::-1]
    giant = int(sizes[order[0]])
    n_singleton = int(np.sum(sizes == 1))

    return {
        "graph": graph.name, "n_nodes": n,
        "n_scc": int(n_scc), "n_wcc": int(n_wcc),
        "giant_scc_size": giant, "giant_scc_share": giant / n,
        "n_singleton_scc": n_singleton, "singleton_share": n_singleton / n,
        "scc_sizes_top": [int(s) for s in sizes[order[:top]]],
    }


def print_connectivity_report(d):
    n = d["n_nodes"]
    print(f"\n=== {d['graph']} / Zusammenhang (gerichtet) ===")
    rows = [
        ("|V|", n),
        ("SCCs (stark zusammenhaengend)", d["n_scc"]),
        ("groesste SCC", d["giant_scc_size"]),
        ("WCCs (schwach zusammenhaengend)", d["n_wcc"]),
        ("Knoten in Ein-Knoten-SCCs", d["n_singleton_scc"]),
    ]
    print(f"{'Groesse':40}{'Wert':>16}{'Anteil |V|':>13}")
    for label, val in rows:
        print(f"{label:40}{val:>16,.0f}{val / n:>13.5f}".replace(",", " "))
    print(f"\ngroesste SCCs: {d['scc_sizes_top']}")


def diagnose_durw(graph, view_name="directed", jump_weight: float = config.DURW_JUMP_WEIGHT,
                  budget_rel=0.01, seed: int = config.DEFAULT_SEED, n_checkpoints=60):
    """Sprungrate, Sackgassen-Fluchtrate und deg_Gu ueber einen echten DURW-Lauf.

    Anders als diagnose(): laeuft DurwSampler statt RandomWalkSampler, weil
    DURW nie absorbierend steckenbleibt -- "weniger verbunden" zeigt sich hier
    nicht als Plateau, sondern als haeufigerer Zwangssprung und niedriger
    deg_Gu bei Erstbesuch (siehe sampling.durw Docstring fuer die Zahlen, die
    das hier reproduzierbar macht).
    """
    base = build_view(graph, "directed")
    view = build_view(graph, view_name)
    n = view.n_nodes
    budget = max(int(round(budget_rel * n)), 2)
    out_deg = np.diff(base.indptr)

    oracle = JumpCrawlOracle(view, random.Random(f"diag-durw|{seed}|{jump_weight:g}"),
                             budget, config.DEFAULT_BUDGET_METRIC)
    sampler = DurwSampler(jump_weight=jump_weight)
    trace = sampler.sample(oracle)
    k = len(trace)

    checkpoints = np.unique(np.geomspace(1, max(k, 2), n_checkpoints).astype(np.int64))
    seen: set[int] = set()
    jump_rate_curve, escape_curve, deg_gu_curve = [], [], []
    n_jumped = n_dead_ends = n_dead_end_escapes = n_first_visits = 0
    deg_gu_sum = 0.0
    ci = 0
    for i, s in enumerate(trace, start=1):
        if s.jumped:
            n_jumped += 1
        if s.node not in seen:
            seen.add(s.node)
            n_first_visits += 1
            deg_gu_sum += s.degree
            if out_deg[s.node] == 0:
                n_dead_ends += 1
                if s.degree >= 1:
                    n_dead_end_escapes += 1
        while ci < len(checkpoints) and i >= checkpoints[ci]:
            jump_rate_curve.append((i, n_jumped / i))
            escape_curve.append(
                (i, n_dead_end_escapes / n_dead_ends if n_dead_ends else float("nan")))
            deg_gu_curve.append(
                (i, deg_gu_sum / n_first_visits if n_first_visits else float("nan")))
            ci += 1

    return {
        "graph": graph.name, "view": view_name, "jump_weight": jump_weight, "seed": seed,
        "budget_rel": budget_rel, "budget_abs": budget, "steps": k,
        "n_nodes": n, "distinct": len(seen),
        "jump_rate": n_jumped / k if k else float("nan"),
        "dead_ends_visited": n_dead_ends,
        "dead_end_escape_rate": n_dead_end_escapes / n_dead_ends if n_dead_ends else float("nan"),
        "mean_deg_gu": deg_gu_sum / n_first_visits if n_first_visits else float("nan"),
        "jump_rate_curve": jump_rate_curve,
        "dead_end_escape_curve": escape_curve,
        "deg_gu_curve": deg_gu_curve,
    }


def print_durw_report(d):
    n = d["n_nodes"]
    print(f"\n=== {d['graph']} / {d['view']} / DURW w={d['jump_weight']:g} "
          f"/ seed={d['seed']} ===")
    print(f"Budget {d['budget_rel']:g} = {d['budget_abs']:,} Einheiten, "
          f"{d['steps']:,} Schritte, {d['distinct']:,} verschieden besucht\n"
          .replace(",", " "))
    print(f"Sprungrate:                              {d['jump_rate']:>8.1%}")
    print(f"Besuchte Sackgassen (Anteil |V|):         {d['dead_ends_visited']:>8,} "
          f"({d['dead_ends_visited'] / n:.1%})".replace(",", " "))
    print(f"davon mit deg_Gu >= 1 (koennen zurueck):  {d['dead_end_escape_rate']:>8.1%}")
    print(f"mittlerer deg_Gu bei Erstbesuch:          {d['mean_deg_gu']:>8.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", required=True)
    p.add_argument("--views", nargs="+", default=["directed"])
    p.add_argument("--dead-end", default="history", choices=sorted(DEAD_ENDS))
    p.add_argument("--budget", type=float, default=0.01)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED,
                   help=f"Zufallsstrom des Walks (Default: {config.DEFAULT_SEED})")
    p.add_argument("--connectivity", action="store_true",
                   help="zusaetzlich SCC/WCC-Zerlegung und einen echten "
                        "DURW-Lauf (Sprungrate, Sackgassen-Fluchtrate, deg_Gu) "
                        "diagnostizieren")
    p.add_argument("--jump-weight", type=float, default=config.DURW_JUMP_WEIGHT,
                   help=f"w fuer --connectivity's DURW-Lauf "
                        f"(Default: {config.DURW_JUMP_WEIGHT:g})")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    graph = loader.load_graph(args.graph)   # Kuerzel loest der Loader auf
    results = []
    for view in args.views:
        d = diagnose(graph, view, args.dead_end, args.budget, args.top, args.seed)
        print_report(d)
        results.append(d)

    if not args.no_plot:
        from plotting.walk_diagnosis import plot_diagnosis
        path = plot_diagnosis(results)
        print("\n  ->", path)

    if args.connectivity:
        print_connectivity_report(diagnose_connectivity(graph))
        durw_results = [
            diagnose_durw(graph, view, args.jump_weight, args.budget, args.seed)
            for view in args.views
        ]
        for dd in durw_results:
            print_durw_report(dd)
        if not args.no_plot:
            from plotting.walk_diagnosis import plot_durw_diagnosis
            path = plot_durw_diagnosis(durw_results)
            print("\n  ->", path)


if __name__ == "__main__":
    main()
