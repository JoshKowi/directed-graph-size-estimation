"""CLI: pruefen, dass die IE2-Formeln rechnen, was ihre Definition sagt.

IE2 (estimators.formulas.IE2SetEstimator) zaehlt Treffer gegen
A = Vereinigung aller beobachteten Nachbarschaften. Definiert ist das ueber
Doppelsummen ueber alle Indexpaare; gerechnet wird es in
sampling.observed.NeighborIndex ohne jede Doppelsumme -- der Zeugen-Index
(kleinster/groesster Sample-Index, an dem ein Knoten als Nachbar auftrat)
ersetzt sie. Diese Umformung ist die eigentliche Behauptung, und nur sie kann
still falsch sein: eine Verschiebung um eins im Fenster oder ein falsches
Intervallende faellt nirgends auf, sondern verschiebt nur die Schaetzung.

Geprueft wird deshalb gegen eine naive Implementierung der *Definition*, die
die Doppelsummen ausschreibt -- auf kleinen Stichproben, wo O(n^2) noch geht:

    1. Set-Form == Definition,       m in 0, 1, 5, 20
    2. Multiset-Form == Definition,  m in 0, 1, 5, 20
    3. m = 0 gegen Gl. 17 wortwoertlich -- die Differenz muss *genau* der
       Diagonalterm sein (s. IE2SetEstimator, "m = 0 ist nicht Gl. 17
       wortwoertlich")
    4. Konvergenz: unabhaengige WIS-Ziehungen, m = 0 -> N_hat ~ N
    5. Am echten Graphen: DURW ohne Sprung, Sweep ueber m. Erwartet werden die
       drei Regime -- Unterschaetzung (m zu klein), Plateau (der Schaetzwert),
       NaN (m zu gross, keine Kreuzkollision mehr uebrig). Dazu `in_a_frac`:
       ohne Margin muss es nahe 1 liegen, denn s_{i+1} ist per Konstruktion ein
       Nachbar von s_i -- das ist der numerische Beleg dafuer, dass der Margin
       bei einem Random Walk nicht optional ist.

Die Punkte 1-3 sind exakte Vergleiche (np.isclose mit rtol=1e-12), 4 und 5 sind
Messungen und werden nur berichtet.

Beispiele:
    python check_ie2.py
    python check_ie2.py --graph Slashdot0811 --views directed --budget 0.05
"""

from __future__ import annotations

import argparse
import random

import numpy as np

import config
import estimators as estimator_registry
from estimators.formulas import (IE2MultisetEstimator, IE2SetEstimator,
                                 WISCollisionEstimatorKatzir, _collisions)
from graphs import loader
from graphs.graph import Graph, _simplify
from graphs.views import build_view
from sampling.base import Sample
from sampling.observed import ObservedNeighborhoods
from weighting.schemes import InverseDegreeWeighting

RTOL = 1e-12


# --------------------------------------------------------------- Testgraph
def synthetic(n: int, avg_deg: float, rng: random.Random) -> Graph:
    """Kleiner ungerichteter Graph mit schiefer Gradverteilung.

    Schief, weil genau dort der Unterschied zwischen Set- und Multiset-Form
    liegen soll (s. IE2SetEstimator): die Kantenenden werden proportional zu
    einem Pareto-Gewicht gezogen, nicht gleichverteilt.
    """
    w = np.array([rng.paretovariate(1.5) for _ in range(n)])
    p = w / w.sum()
    m = int(n * avg_deg / 2)
    a = np.random.default_rng(rng.randrange(2**32)).choice(n, size=2 * m, p=p)
    src, dst = a[:m], a[m:]
    # symmetrisieren: beide Richtungen in eine CSR-Struktur, _simplify raeumt
    # Schlingen und Mehrfachkanten weg (wie beim echten Laden)
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    order = np.argsort(u, kind="stable")
    u, v = u[order], v[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(u, minlength=n), out=indptr[1:])
    indptr, indices = _simplify(indptr, v.astype(np.int64), n)
    return Graph(indptr, indices, list(range(n)), name=f"synth{n}")


def wis_sample(graph: Graph, k: int, rng: random.Random):
    """Unabhaengige WIS-Ziehung mit pi ~ deg, plus die Nachbarschaften dazu.

    Bewusst privilegiert (zieht direkt aus dem Graphen) -- das ist der Fall,
    fuer den das Paper IE2 herleitet, und damit der Fall, in dem die Formel
    ohne jedes Gegenmittel |V| treffen muss.
    """
    deg = np.diff(graph.indptr).astype(float)
    p = deg / deg.sum()
    nodes = np.random.default_rng(rng.randrange(2**32)).choice(
        len(deg), size=k, p=p)
    samples = [Sample(int(u), int(deg[u]), i) for i, u in enumerate(nodes)]
    nbrs = {int(u): np.asarray(graph.neighbors(int(u))) for u in set(nodes.tolist())}
    return samples, ObservedNeighborhoods(raw=nbrs, gu=nbrs)


# ------------------------------------------------------- naive Definition
def naive_set(samples, weights, nbrs: dict, margin: int) -> float:
    """Gl. 17 mit Exklusion 1{|j-i| > m}, Doppelsumme ausgeschrieben."""
    n = len(samples)
    num = den = 0.0
    for i in range(n):
        far: set = set()
        for j in range(n):
            if abs(j - i) > margin:
                far.update(int(v) for v in nbrs[samples[j].node])
        num += weights[i] * len(far)
        if samples[i].node in far:
            den += weights[i]
    return float("nan") if den <= 0 else num / den


def naive_multiset(samples, weights, nbrs: dict, margin: int) -> float:
    """Gl. 24, Doppelsumme ausgeschrieben."""
    n = len(samples)
    sets = {u: set(int(v) for v in a) for u, a in nbrs.items()}
    num = den = 0.0
    for i in range(n):
        for j in range(n):
            if abs(j - i) > margin:
                num += weights[i] * len(nbrs[samples[j].node])
                den += weights[i] * (samples[i].node in sets[samples[j].node])
    return float("nan") if den <= 0 else num / den


def gl17_literal(samples, weights, nbrs: dict) -> float:
    """Gl. 17 ganz ohne Exklusion -- auch das Paar j = i zaehlt mit."""
    a: set = set()
    for s in samples:
        a.update(int(v) for v in nbrs[s.node])
    den = sum(w for s, w in zip(samples, weights) if s.node in a)
    return float("nan") if den <= 0 else len(a) * float(np.sum(weights)) / den


# ------------------------------------------------------------------ Checks
def _cmp(label: str, got: float, want: float, fails: list) -> None:
    ok = (np.isnan(got) and np.isnan(want)) or np.isclose(got, want, rtol=RTOL)
    print(f"  {'OK ' if ok else 'FEHLER'}  {label:<38} "
          f"linear={got:.10g}  naiv={want:.10g}")
    if not ok:
        fails.append(label)


def check_against_definition(rng: random.Random, fails: list) -> None:
    print("\n[1/2] Linearisierung gegen die Definition (kleine Stichproben)")
    for n_nodes, k in ((300, 120), (1000, 200)):
        g = synthetic(n_nodes, 6.0, rng)
        samples, obs = wis_sample(g, k, rng)
        w = InverseDegreeWeighting().weights(samples)
        nbrs = obs.raw
        print(f"  Graph |V|={g.n_nodes}, Stichprobe k={k}, "
              f"|A|={obs.index(samples, 'raw').a_size}")
        for m in (0, 1, 5, 20):
            _cmp(f"Set-Form, m={m}",
                 IE2SetEstimator(margin=m).compute(samples, w, obs),
                 naive_set(samples, w, nbrs, m), fails)
            _cmp(f"Multiset-Form, m={m}",
                 IE2MultisetEstimator(margin=m).compute(samples, w, obs),
                 naive_multiset(samples, w, nbrs, m), fails)


def check_gl17(rng: random.Random, fails: list) -> None:
    print("\n[3] m = 0 gegen Gl. 17 wortwoertlich")
    g = synthetic(800, 8.0, rng)
    samples, obs = wis_sample(g, 300, rng)
    w = InverseDegreeWeighting().weights(samples)
    idx = obs.index(samples, "raw")
    ours = IE2SetEstimator(margin=0).compute(samples, w, obs)
    lit = gl17_literal(samples, w, obs.raw)
    # Die Differenz muss genau der Diagonalterm sein: Gl. 17 zaehlt im Zaehler
    # auch j = i mit, wir lassen ihn aus. Im Nenner ist beides gleich (kein
    # Knoten ist sein eigener Nachbar).
    hit = idx.in_a & idx.far_self(0)
    den = float(np.sum(w[hit]))
    diag = float(np.sum(w * idx.deg)) - float(np.sum(
        w * (idx.a_size - idx.a_far(0))))
    print(f"  m=0 (Gl. 24)  = {ours:.10g}")
    print(f"  Gl. 17 wortw. = {lit:.10g}   (Differenz {lit - ours:+.4g}, "
          f"relativ {(lit - ours) / ours:+.2%})")
    # Gl. 17 - unser Zaehler == der Beitrag der Knoten, deren einziger Zeuge i
    # selbst ist; ueber die Summe ist das genau a_size - a_far(0).
    want = ours + float(np.sum(w * (idx.a_size - idx.a_far(0)))) / den
    _cmp("Differenz == Diagonalterm", lit, want, fails)
    print(f"  (nur zur Groessenordnung: Diagonalterm ueber deg waere {diag:.4g})")


def check_convergence(rng: random.Random) -> None:
    print("\n[4] Konvergenz bei unabhaengigen WIS-Ziehungen (m = 0)")
    g = synthetic(2000, 10.0, rng)
    print(f"  |V| = {g.n_nodes}, |E| = {g.n_edges // 2}")
    for k in (200, 500, 1000, 2000):
        vals_s, vals_m, vals_c = [], [], []
        for _ in range(15):
            samples, obs = wis_sample(g, k, rng)
            w = InverseDegreeWeighting().weights(samples)
            vals_s.append(IE2SetEstimator(margin=0).compute(samples, w, obs))
            vals_m.append(IE2MultisetEstimator(margin=0).compute(samples, w, obs))
            vals_c.append(_katzir(samples, w))
        print(f"  k={k:>5}  IE2-Set={np.median(vals_s):>9.1f}  "
              f"IE2-Multiset={np.median(vals_m):>9.1f}  "
              f"Kollisionen={np.nanmedian(vals_c):>9.1f}  "
              f"(NaN-Anteil Kollisionen: "
              f"{np.mean(np.isnan(vals_c)):.0%})")


def _katzir(samples, w) -> float:
    from estimators.formulas import WISCollisionEstimatorKatzir
    return WISCollisionEstimatorKatzir(margin=0).compute(samples, w)


def check_real(graph_name: str, views, budget: float, margins, seed: int) -> None:
    """DURW ohne Sprung, ein Walk, alle Auswertungen -- Sweep ueber m.

    Gezogen wird *einmal*; danach bekommen beide IE2-Formeln und der
    Kollisions-Schaetzer dieselbe Trajektorie. Der Vergleich ist damit gepaart
    und enthaelt kein RNG-Rauschen -- dieselbe Idee wie --share-walks im Runner
    (und moeglich, weil collect_nbrs die Trajektorie nicht aendert).
    """
    print(f"\n[5] Echter Graph: {graph_name}, DURW ohne Sprung, Sweep ueber m")
    graph = loader.load_graph(graph_name)
    for view_name in views:
        view = build_view(graph, view_name)
        b = max(int(round(budget * view.n_nodes)), 2)
        # Der Walk: vom IE2-Estimator gezogen, weil nur dessen Sampler die
        # Nachbarschaften aufzeichnet. Bitgleich mit dem des Kollisions-
        # Estimators (s. Modul-Docstring von sampling.durw, `collect_nbrs`).
        owner = estimator_registry.build("ie2-durw__nojump__margin")
        trace, oracle = owner.run_walk(view, b, random.Random(f"{seed}|ie2"))
        obs = owner.sampler.observed
        w = owner.weighting.weights(trace)
        k = len(trace)
        katzir = WISCollisionEstimatorKatzir(margin=config.SAFETY_MARGIN)
        base = katzir.compute(trace, w)
        print(f"\n  {view_name}: |V| = {view.n_nodes:,}, Budget = {b:,} Queries, "
              f"k = {k:,} Samples")
        print(f"  {'m':>5}  {'IE2-Set':>9}  {'IE2-Mset':>9}  {'in_A':>6}  "
              f"{'xcol(Set)':>10}  {'xcol(Mset)':>12}  {'n_col':>9}")
        for m in margins:
            forms = [IE2SetEstimator(margin=m), IE2MultisetEstimator(margin=m)]
            vals = [f.compute(trace, w, obs) for f in forms]
            ex = [f.extras([trace], [w], obs) for f in forms]
            print(f"  {m:>5}  {vals[0] / view.n_nodes:>8.3f}x  "
                  f"{vals[1] / view.n_nodes:>8.3f}x  "
                  f"{ex[0]['in_a_frac']:>6.3f}  {ex[0]['n_xcol']:>10,}  "
                  f"{ex[1]['n_xcol']:>12,}  {_collisions(trace, m):>9,.0f}")
        print(f"  Referenz auf demselben Walk: wis-durw__nojump__margin "
              f"(m={config.SAFETY_MARGIN}) = {base / view.n_nodes:.3f}x")
        print("  Werte relativ zu |V|, 1.000x ist exakt. n_col sind die Knoten-")
        print("  Kollisionen derselben Trajektorie -- der Vergleich xcol gegen")
        print("  n_col ist der Ereignisgewinn, den IE2 verspricht. Die Set-Form")
        print("  kann nie mehr als k Ereignisse zaehlen (ein Indikator je Sample).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", help="echter Graph fuer den m-Sweep (Teil 5)")
    ap.add_argument("--views", nargs="+", default=["directed"])
    ap.add_argument("--budget", type=float, default=0.02)
    ap.add_argument("--margins", nargs="+", type=int,
                    default=[0, 1, 5, 20, 50, 200, 1000])
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    ap.add_argument("--skip-definition", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fails: list = []
    if not args.skip_definition:
        check_against_definition(rng, fails)
        check_gl17(rng, fails)
        check_convergence(rng)
    if args.graph:
        check_real(args.graph, args.views, args.budget, args.margins, args.seed)

    print()
    if fails:
        print(f"FEHLER in {len(fails)} Vergleich(en): {', '.join(fails)}")
        raise SystemExit(1)
    print("Alle exakten Vergleiche bestanden.")


if __name__ == "__main__":
    main()
