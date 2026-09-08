"""CLI: Namensindex fuer eine externe Ziehungsquelle bauen.

Der Ziehungspfad zur Laufzeit darf keine Strings anfassen. Deshalb wird der
Abgleich zwischen Namensliste und Graph *einmal vorab* gemacht und als
int32-Array abgelegt:

    Position in der Liste  ->  Knoten-ID im Graphen, oder -1 (kein Treffer)

Ziehen ist danach arr[rng.randrange(len(arr))] -- O(1), ohne Woerterbuch, und
das Array ist 4 Byte je Listeneintrag (enwiki 77 MB, dewiki 20 MB, top-q
0,4 MB). Die teure Struktur, ein dict aus allen normalisierten Knotennamen,
entsteht nur hier: bei 6,5 Mio Eintraegen sind das ein bis zwei GB, die im
Experiment je Kindprozess erneut anfielen. graphs/graph.py:seed_ids() warnt aus
genau diesem Grund vor Graph.id_of().

Positionstreue: auch ein Eintrag ohne einzigen Kandidaten bekommt seinen Platz
im Array (als -1). Wuerde er uebersprungen, verschoebe sich alles danach und
der Index passte nicht mehr zur Liste.

Beispiele:
    python build_name_index.py --graphs gpt4_io --sources enwiki
    python build_name_index.py --graphs gpt4_io gpt4o_io --sources enwiki dewiki top-q
    python build_name_index.py --graphs slashdot --sources enwiki --force
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

import config
import namelists
import provenance
from graphs import loader

# Die IDs passen in int32, solange kein Graph mehr als 2^31-1 Knoten hat --
# dieselbe Annahme, die graphs.graph.ID_DTYPE ohnehin trifft. -1 heisst
# "kein Treffer" und ist deshalb kein gueltiger Index.
MISS = -1
INDEX_DTYPE = np.int32


def index_path(graph_name: str, source: str):
    return config.NAME_INDEX_DIR / f"{graph_name}__{source}.npy"


def meta_path(graph_name: str, source: str):
    return config.NAME_INDEX_DIR / f"{graph_name}__{source}.json"


def _log(msg: str) -> None:
    print(msg.replace(",", " "), flush=True)


def build(graph, source: str, log=_log):
    """(Array, Metadaten) fuer einen Graphen und eine Quelle."""
    src = namelists.resolve(source)

    t0 = time.perf_counter()
    lookup: dict[str, int] = {}
    for i, k in enumerate(graph.names):
        if isinstance(k, str):
            # Bei Kollisionen gewinnt der erste Knoten. Willkuerlich, aber
            # deterministisch -- und die Zahl steht unten im Protokoll, damit
            # sichtbar bleibt, wie oft es vorkommt.
            lookup.setdefault(src.norm.apply(k), i)
    n_named = sum(1 for k in graph.names if isinstance(k, str))
    log(f"  Knotenindex: {len(lookup):,} verschiedene Namen aus {n_named:,} "
        f"({n_named - len(lookup):,} durch Normalisierung zusammengefallen) "
        f"in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    out: list[int] = []
    # Wie oft welcher Kandidat den Treffer gebracht hat -- bei top-q zeigt das,
    # ob die Reihenfolge enwiki -> dewiki -> label ueberhaupt etwas beitraegt.
    by_rank: dict[int, int] = {}
    empty = 0
    for cands in src.entries():
        if not cands:
            empty += 1
        hit = MISS
        for rank, c in enumerate(cands):
            found = lookup.get(c)
            if found is not None:
                hit = found
                by_rank[rank] = by_rank.get(rank, 0) + 1
                break
        out.append(hit)
    arr = np.array(out, dtype=INDEX_DTYPE)
    del out, lookup

    n_hit = int((arr != MISS).sum())
    n_nodes_hit = int(len(np.unique(arr[arr != MISS])))
    meta = {
        "graph": graph.name,
        "source": source,
        "path": str(src.path),
        "normalizer": src.norm.label(),
        "columns": list(src.columns),
        "n_entries": int(arr.size),
        "n_hits": n_hit,
        "n_nodes": int(graph.n_nodes),
        "n_nodes_hit": n_nodes_hit,
        "titles_matched": n_hit / arr.size if arr.size else 0.0,
        "nodes_covered": n_nodes_hit / graph.n_nodes if graph.n_nodes else 0.0,
        "draws_per_hit": arr.size / n_hit if n_hit else float("inf"),
        "hits_by_candidate_rank": {str(k): v for k, v in sorted(by_rank.items())},
        "n_entries_without_candidate": empty,
        "code": provenance.code_fingerprint(),
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    log(f"  Abgleich in {time.perf_counter()-t0:.1f}s: "
        f"Treffer {meta['titles_matched']:.2%}, Abdeckung "
        f"{meta['nodes_covered']:.2%}, {meta['draws_per_hit']:.1f} Ziehungen "
        f"je Treffer")
    if src.columns:
        log(f"  Treffer nach Kandidat: "
            + ", ".join(f"{src.columns[int(r)]}={n:,}"
                        for r, n in meta["hits_by_candidate_rank"].items()))
    return arr, meta


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graphs", nargs="+", required=True,
                   help="Graph-Namen oder Kuerzel (s. config.GRAPH_ALIASES)")
    p.add_argument("--sources", nargs="+", default=sorted(namelists.SOURCES),
                   choices=sorted(namelists.SOURCES))
    p.add_argument("--force", action="store_true",
                   help="vorhandene Indizes neu bauen statt zu ueberspringen")
    args = p.parse_args()

    config.NAME_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    for gname in args.graphs:
        todo = [s for s in args.sources
                if args.force or not index_path(config.resolve_graph(gname), s).exists()]
        if not todo:
            _log(f"[{gname}] alle Indizes vorhanden -- mit --force neu bauen")
            continue
        loader.clear_cache()
        t0 = time.perf_counter()
        graph = loader.load_graph(gname)
        _log(f"[{graph.name}] geladen in {time.perf_counter()-t0:.0f}s -- "
             f"|V| = {graph.n_nodes:,}")
        for source in todo:
            _log(f"[{graph.name} / {source}]")
            arr, meta = build(graph, source)
            np.save(index_path(graph.name, source), arr)
            meta_path(graph.name, source).write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
            dst = index_path(graph.name, source)
            _log(f"  -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
