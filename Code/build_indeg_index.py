"""CLI: In-Grad-Index aus dem Partnergraphen bauen.

NMMC braucht zu jedem vorgeschlagenen Knoten dessen Eingangsgrad (siehe
sampling/nmmc.py). Der ist im gecrawlten Graphen nicht verfuegbar -- wohl aber
in einem *anderen* Graphen, der dieselben Entitaeten beschreibt: fuer einen
Lauf auf gpt4_io liefert gpt4o_io die Eingangsgrade und umgekehrt
(config.CROSS_GRAPHS). Das ist externes Wissen ueber die Entitaeten, keine
Kenntnis der Knotenmenge des geschaetzten Graphen -- dieselbe Begruendung, die
auch die Namenslisten real umsetzbar macht.

Wie bei build_name_index.py darf der Laufzeitpfad keine Strings anfassen.
Deshalb wird der Abgleich *einmal vorab* gemacht und als int32-Array abgelegt:

    Knoten-ID im Zielgraphen  ->  Eingangsgrad im Partnergraphen, oder -1

Zur Laufzeit ist die Abfrage arr[v] -- O(1), ohne Woerterbuch, 4 Byte je
Knoten. Die teure Struktur, ein dict aus allen Namen des Partners, entsteht nur
hier: bei 5,7 Mio. Eintraegen sind das ein bis zwei GB, die im Experiment je
Kindprozess erneut anfielen (graphs/graph.py:seed_ids warnt aus demselben Grund
vor Graph.id_of()).

Die Graphen werden nacheinander geladen -- graphs.loader haelt bewusst nur
einen im RAM. Zwischen beiden ueberlebt nur das dict.

Normalisierung: Default ist "roh", also exakter Namensvergleich. Beide GPT-
Basen benutzen denselben Schluesselraum, und ohne Normalisierung kann kein
Namenspaar kollidieren. --norm probiert eine Variante aus title_overlap.VARIANTS;
der Bericht nennt Abdeckungsgewinn *und* Kollisionen, damit die Abwaegung
sichtbar bleibt.

Beispiele:
    python build_indeg_index.py --graphs gpt4_io gpt4o_io
    python build_indeg_index.py --graphs gpt4_io --norm +casefold --dry-run
    python build_indeg_index.py --graphs gpt4_io --force
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

import config
import provenance
from graphs import loader
from title_overlap import VARIANTS

# -1 heisst "im Partner nicht vorhanden" und ist deshalb kein gueltiger Grad.
MISS = -1
INDEX_DTYPE = np.int32


def index_path(graph: str, partner: str):
    return config.INDEG_INDEX_DIR / f"{graph}__from__{partner}.npy"


def load_index(graph: str, partner: str) -> np.ndarray:
    """Fertigen Index laden. Wirft, wenn er fehlt -- mit dem Bau-Befehl."""
    path = index_path(graph, partner)
    if not path.exists():
        raise SystemExit(
            f"In-Grad-Index fehlt: {path}\n"
            f"Bauen mit: python build_indeg_index.py --graphs {graph}")
    return np.load(path)


def build(graph_name: str, norm_name: str = "roh") -> tuple[np.ndarray, dict]:
    """Index fuer einen Zielgraphen. Laedt beide Graphen nacheinander."""
    graph_name = config.resolve_graph(graph_name)
    partner = config.cross_graph(graph_name)
    if partner is None:
        raise SystemExit(
            f"Fuer {graph_name} ist kein Partnergraph hinterlegt "
            f"(config.CROSS_GRAPHS). Moeglich: {', '.join(sorted(config.CROSS_GRAPHS))}")
    norm = VARIANTS[norm_name]

    # -- 1. Partner: Name -> Eingangsgrad ---------------------------------
    t0 = time.perf_counter()
    src = loader.load_graph(partner)
    d_in = src.in_degrees
    collisions = 0
    if norm_name == "roh":
        # Schnellpfad: "roh" laesst den Namen unveraendert, der Aufruf je Name
        # waere reine Schleifenlast. Die Schluessel eines Graphen sind
        # eindeutig (graphs.graph.load_pickle bildet sie auf IDs ab), ohne
        # Normalisierung kann hier also nichts kollidieren.
        by_name: dict = dict(zip(src.names, d_in.tolist()))
    else:
        by_name = {}
        for i, nm in enumerate(src.names):
            key = norm.apply(str(nm))
            if key in by_name:
            # Nur bei Normalisierung moeglich: zwei Entitaeten fallen zusammen.
            # Der groessere Grad gewinnt -- ein zu kleiner Nenner in b_ij macht
            # die Annahme zu grosszuegig, und genau das soll der Index abbauen.
                collisions += 1
                if d_in[i] > by_name[key]:
                    by_name[key] = int(d_in[i])
            else:
                by_name[key] = int(d_in[i])
    n_partner = src.n_nodes
    print(f"  Partner {partner}: {n_partner:,} Knoten, {len(by_name):,} Namen"
          f"{f', {collisions:,} Kollisionen' if collisions else ''} "
          f"({time.perf_counter()-t0:.0f}s)".replace(",", " "))

    # -- 2. Zielgraph: ID -> Grad im Partner -------------------------------
    t0 = time.perf_counter()
    graph = loader.load_graph(graph_name)      # verdraengt den Partner
    get = by_name.get
    if norm_name == "roh":
        arr = np.fromiter((get(nm, MISS) for nm in graph.names),
                          dtype=INDEX_DTYPE, count=graph.n_nodes)
    else:
        arr = np.fromiter((get(norm.apply(str(nm)), MISS) for nm in graph.names),
                          dtype=INDEX_DTYPE, count=graph.n_nodes)
    found = int((arr >= 0).sum())
    stats = {
        "graph": graph_name, "partner": partner, "norm": norm_name,
        "n_nodes": int(graph.n_nodes), "n_partner_nodes": int(n_partner),
        "n_partner_names": len(by_name), "collisions": collisions,
        "found": found, "coverage": found / graph.n_nodes,
        "zero_indeg_hits": int((arr == 0).sum()),
        "median_indeg_hit": float(np.median(arr[arr > 0])) if (arr > 0).any() else 0.0,
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": provenance.code_fingerprint(),
    }
    print(f"  Ziel {graph_name}: {found:,} von {graph.n_nodes:,} Knoten gefunden "
          f"({stats['coverage']:.1%}), Median-Grad der Treffer "
          f"{stats['median_indeg_hit']:.0f} ({time.perf_counter()-t0:.0f}s)"
          .replace(",", " "))
    return arr, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graphs", nargs="+", required=True, metavar="GRAPH",
                   help="Zielgraphen; der Partner kommt aus config.CROSS_GRAPHS")
    p.add_argument("--norm", default="roh", choices=sorted(VARIANTS),
                   help="Normalisierung der Namen (Default: roh = exakt)")
    p.add_argument("--force", action="store_true",
                   help="vorhandenen Index neu bauen")
    p.add_argument("--dry-run", action="store_true",
                   help="nur messen, nichts schreiben")
    args = p.parse_args()

    config.INDEG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    for g in args.graphs:
        name = config.resolve_graph(g)
        partner = config.cross_graph(name)
        path = index_path(name, partner) if partner else None
        print(f"[{name}] Partner {partner}")
        if path is not None and path.exists() and not args.force and not args.dry_run:
            print(f"  liegt schon vor: {path} (--force zum Neubau)")
            continue
        arr, stats = build(name, args.norm)
        if args.dry_run:
            print("  --dry-run: nichts geschrieben")
            continue
        np.save(path, arr)
        path.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n")
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
