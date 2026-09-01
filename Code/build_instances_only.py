"""CLI: aus einer Adjazenzliste die Variante "nur Instanzen" bauen.

Die GPT-Wissensgraphen enthalten neben echten Entitaeten auch Literale
("person", "1890-03-11", "American") und unbrauchbare Fragmente ("<pre>",
"about 708,127 (2020) "). Fuer die Groessenschaetzung sind sie Ballast: sie
blaehen |V| auf, haben fast nie ausgehende Kanten und sind als "Knoten" auch
inhaltlich fragwuerdig. Welcher Name was ist, steht in den nodes-Tabellen
unter `nodes/` -- Spalte `type` mit den Werten instance / literal / undefined.

Gefiltert wird auf beiden Seiten jeder Kante: ein Schluessel bleibt nur, wenn
er selbst eine Instanz ist, und seine Nachbarliste behaelt nur Instanzen. Ein
Schluessel, dessen Nachbarn alle Literale waren, bleibt als Knoten *ohne*
ausgehende Kanten erhalten -- er ist weiterhin Teil von V (siehe README,
"Entwurfsentscheidungen").

Mehrfachkanten und Schlingen bleiben in der Datei stehen; entfernt werden sie
beim Laden (graphs.graph._simplify), damit die .pkl eine reine Projektion der
Quelle bleibt und man den Unterschied noch sehen kann.

Beispiel:
    python build_instances_only.py --adjacency adjacency_list_uni \\
        --nodes gpt4_nodes --out gpt4_io
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import config

NODES_DIR = config.ROOT / "nodes"


def _log(msg: str) -> None:
    print(msg, flush=True)


def instance_names(nodes_file: Path) -> set:
    """Namen mit type == "instance" aus der nodes-Tabelle."""
    t0 = time.perf_counter()
    with open(nodes_file, "rb") as fh:
        df = pickle.load(fh)
    counts = df["type"].value_counts().to_dict()
    names = set(df.loc[df["type"] == "instance", "name"].astype(str))
    _log(f"[{nodes_file.name}] {len(df):,} Zeilen in {time.perf_counter()-t0:.1f}s "
         f"-- {counts}".replace(",", " "))
    return names


def filter_adjacency(adj: dict, keep: set) -> dict:
    """Beide Kantenseiten auf `keep` einschraenken."""
    out = {}
    for k, nbrs in adj.items():
        if k in keep:
            out[k] = [v for v in nbrs if v in keep]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adjacency", required=True, help="Name unter adjacencies/ (ohne .pkl)")
    p.add_argument("--nodes", required=True, help="Name unter nodes/ (ohne .pkl)")
    p.add_argument("--out", required=True, help="Zielname unter adjacencies/ (ohne .pkl)")
    p.add_argument("--force", action="store_true", help="vorhandene Zieldatei ueberschreiben")
    args = p.parse_args()

    src = config.ADJACENCIES_DIR / f"{config.resolve_graph(args.adjacency)}.pkl"
    dst = config.ADJACENCIES_DIR / f"{args.out}.pkl"
    if dst.exists() and not args.force:
        raise SystemExit(f"{dst} existiert bereits -- mit --force ueberschreiben")

    keep = instance_names(NODES_DIR / f"{args.nodes}.pkl")

    t0 = time.perf_counter()
    with open(src, "rb") as fh:
        adj = pickle.load(fh)
    edges_in = sum(len(v) for v in adj.values())
    _log(f"[{src.name}] {len(adj):,} Schluessel, {edges_in:,} Kanten in "
         f"{time.perf_counter()-t0:.1f}s".replace(",", " "))

    t0 = time.perf_counter()
    out = filter_adjacency(adj, keep)
    del adj
    edges_out = sum(len(v) for v in out.values())
    _log(f"[{args.out}] {len(out):,} Schluessel ({len(out)/len(keep):.1%} der "
         f"Instanzen), {edges_out:,} Kanten ({edges_out/edges_in:.1%} der "
         f"Quelle) in {time.perf_counter()-t0:.1f}s".replace(",", " "))

    with open(dst, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    _log(f"  -> {dst} ({dst.stat().st_size / 1e9:.2f} GB)")
    _log("Nicht vergessen: config.GRAPH_LABELS / GRAPH_ALIASES und "
         "adjacencies/README.txt ergaenzen.")


if __name__ == "__main__":
    main()
