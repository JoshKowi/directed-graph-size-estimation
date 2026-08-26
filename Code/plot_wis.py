"""CLI: die vier WIS-Vergleichsgrafiken erzeugen.

Alle vier enthalten "uniform_collision" als Referenz (gleichverteiltes Ziehen,
ungewichtete Collision-Formel) und die gestrichelte Linie bei 1.0.

    1  wis_indep__undirected   unabhaengige gradgewichtete Ziehungen, symmetrisiert
    2  wis_indep__directed     dasselbe auf dem Originalgraphen
    3  wis_rw__undirected      alle drei Random-Walk-Varianten, symmetrisiert
    4  wis_rw_history__views   nur history, gerichtet vs symmetrisiert
    5  deadend_uis__directed   Sackgassen-Strategien mit UIS-Formel, gerichtet

Grafik 5 laeuft bewusst auf `directed`: nur dort gibt es ueberhaupt
Sackgassen (Slashdot0811 8,35 %, symmetrisiert 0,00 %) -- siehe
sampling/dead_ends.py.

Die Kurant-Variante ist entfallen: nach der Faktor-2-Korrektur in
estimators/formulas.py ist Kurant Eq.(6) rechnerisch identisch mit Katzir.

Liegen fuer einen Graphen mehrere Seeds vor, wird jeder Satz Grafiken einzeln
gezeichnet -- Laeufe verschiedener Zufallsstroeme in ein Bild zu mischen waere
irrefuehrend, weil die gezeigte Spanne dann zwei Dinge auf einmal misst. Der
Seed steht oben rechts im Bild und (ausser beim Default) im Dateinamen.

Beispiel:
    python plot_wis.py --graphs Slashdot0811 gpt4o_io
    python plot_wis.py --graphs Slashdot0811 --seed 7
"""

from __future__ import annotations

import argparse

import config
from experiment import results as results_io
from plotting.compare import plot_comparison
from plotting.style import color_for
import provenance

REFERENCE = "uniform_collision"

# Feste Farbzuordnung ueber alle vier Grafiken: derselbe Estimator hat
# ueberall dieselbe Farbe, damit die Bilder nebeneinander lesbar sind.
# Feste Slots, damit derselbe Estimator in allen Bildern dieselbe Farbe hat.
# indep und rw-restart teilen sich Slot 1 -- sie stehen nie in derselben Grafik.
# Die Referenz behaelt Slot 0 ueberall.
SLOTS = {
    REFERENCE: 0,
    "wis-katzir__indep": 1,
    # Sackgassen-Strategie -> feste Farbe, unabhaengig von der Formel:
    # restart blau-orange 1, backtrack 3, history 5.
    "wis-katzir__rw-restart": 1,
    "wis-katzir__rw-backtrack": 3,
    "wis-katzir__rw-history": 5,
    "rw_plain__restart__none": 1,
    "rw_plain__backtrack__none": 3,
    "rw_plain__history__none": 5,
}
COLORS = {e: color_for(i) for e, i in SLOTS.items()}

FIGURES = [
    ("wis_indep__undirected",
     [REFERENCE, "wis-katzir__indep"], ["undirected"],
     "WIS (Katzir) -- independent degree-weighted draws (undirected)"),
    ("wis_indep__directed",
     [REFERENCE, "wis-katzir__indep"], ["directed"],
     "WIS (Katzir) -- independent degree-weighted draws (directed)"),
    ("wis_rw__undirected",
     [REFERENCE, "wis-katzir__rw-restart", "wis-katzir__rw-backtrack",
      "wis-katzir__rw-history"], ["undirected"],
     "WIS with a true random walk -- all dead-end strategies (undirected)"),
    ("wis_rw_history__views",
     [REFERENCE, "wis-katzir__rw-history"], ["directed", "undirected"],
     "WIS with random walk (history) -- directed vs undirected"),
    ("deadend_uis__directed",
     [REFERENCE, "rw_plain__restart__none", "rw_plain__backtrack__none",
      "rw_plain__history__none"], ["directed"],
     "UIS collision counting with a random walk -- dead-end strategies (directed)"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graphs", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=None,
                   help="nur diesen Lauf plotten (Default: jeden vorhandenen Seed)")
    args = p.parse_args()

    for graph in (config.resolve_graph(g) for g in args.graphs):
        df = results_io.load_results(graph, seed=args.seed)
        for seed in results_io.seeds_available(df) or [config.DEFAULT_SEED]:
            summary = results_io.summarize(df[df["seed"] == seed])
            tag = results_io.seed_tag(seed)
            # Genestete Budgets gehoeren ins Bild: die Punkte einer Zeile sind
            # dann nicht unabhaengig voneinander (siehe experiment/runner.py).
            note = f"seed {seed}"
            if summary["nested"].any():
                note += "  |  nested budgets"
            for slug, ests, views, title in FIGURES:
                # Ein Bild, in dem nur die Referenz uebrig ist, zeigt nichts --
                # das passiert, wenn ein Lauf nur einen Teil der Estimators oder
                # der Views gerechnet hat. Geprueft wird deshalb genau der
                # Ausschnitt, den die Grafik zeigen soll.
                have = set(summary.loc[summary["view"].isin(views)
                                       & summary["estimator"].isin(ests),
                                       "estimator"].unique())
                if len(have) < 2:
                    print(f"  -- {slug} uebersprungen (Seed {seed}: nur "
                          f"{sorted(have)} in {views} vorhanden)")
                    continue
                path = config.PLOTS_DIR / f"{graph}__{tag}{slug}.png"
                plot_comparison(summary, graph, ests, views,
                                f"{config.graph_label(graph)}: {title}",
                                path=path, colors=COLORS, note=note)
                print("  ->", path)
    for path in provenance.write_readmes():
        print("  ->", path)


if __name__ == "__main__":
    main()
