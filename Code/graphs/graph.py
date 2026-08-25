"""Graph als CSR-Struktur: Knoten sind fortlaufende Integer-IDs.

Die .pkl-Dateien bringen beliebige Original-Schluessel mit (bei gpt4o_io etwa
Strings wie 'Vannevar Bush'). Beim Laden werden sie einmal auf IDs 0..n-1
abgebildet; `names` haelt die Rueckabbildung, `name_of()` / `id_of()` machen
sie zugaenglich. Ergebnisse werden erst beim Schreiben zurueckuebersetzt.

Warum: mit Original-Schluesseln besteht der Graph aus Millionen einzelner
Python-Objekte. Das kostet viel Speicher (gpt4o_io: ~37 GB), verlangsamt jeden
Zugriff -- und verhindert echte Parallelisierung: `fork` teilt Speicher zwar
per Copy-on-Write, aber CPythons Referenzzaehler schreibt bei *jedem* Zugriff
in den Objekt-Header und macht die Seite damit privat. Als CSR besteht der
Graph aus drei numpy-Arrays; deren Datenpuffer werden nie einzeln
referenzgezaehlt, also bleiben sie zwischen Prozessen wirklich geteilt.

Aufbau (Compressed Sparse Row):
    indptr[u] .. indptr[u+1]   Bereich in `indices` mit den Nachbarn von u
    indices                    Nachbar-IDs, nach u gruppiert
Knoten ohne ausgehende Kanten ("dangling") haben indptr[u] == indptr[u+1] und
damit Grad 0 -- sie brauchen keinen Sonderfall.

Schnittstelle:
    class Graph
        .name, .view, .indptr, .indices, .names
        .n_nodes, .n_with_out_edges, .n_edges
        .neighbors(u) -> np.ndarray   (Sicht in `indices`, keine Kopie)
        .degree(u) -> int
        .name_of(u), .id_of(name)
        .random_node(rng), .random_neighbor(u, rng), .random_node_by_degree(rng)
    load_pickle(path) -> Graph
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np

ID_DTYPE = np.int32          # reicht bis 2,1 Mrd Knoten
PTR_DTYPE = np.int64


class Graph:
    view = "directed"

    def __init__(self, indptr, indices, names: list, name: str) -> None:
        self.indptr = indptr
        self.indices = indices
        self.names = names          # ID -> Original-Schluessel
        self.name = name
        self._index: dict | None = None   # lazy, s. id_of()

    # -- Groessen ---------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        """Wahre Graph-Groesse |V|: alle vorkommenden Knoten."""
        return len(self.names)

    @property
    def n_edges(self) -> int:
        return int(self.indptr[-1])

    @property
    def n_with_out_edges(self) -> int:
        return int(np.count_nonzero(np.diff(self.indptr)))

    # -- Zugriff ----------------------------------------------------------
    def neighbors(self, u):
        return self.indices[self.indptr[u]:self.indptr[u + 1]]

    def degree(self, u) -> int:
        return int(self.indptr[u + 1] - self.indptr[u])

    def name_of(self, u):
        """Original-Schluessel zu einer ID."""
        return self.names[u]

    def id_of(self, name):
        """ID zu einem Original-Schluessel (baut den Index beim ersten Aufruf)."""
        if self._index is None:
            self._index = {k: i for i, k in enumerate(self.names)}
        return self._index[name]

    # -- Ziehen -----------------------------------------------------------
    def random_node(self, rng: random.Random) -> int:
        return rng.randrange(self.n_nodes)

    def random_neighbor(self, u, rng: random.Random):
        lo, hi = self.indptr[u], self.indptr[u + 1]
        if hi == lo:
            return None
        return self.indices[lo + rng.randrange(hi - lo)]

    def random_node_by_degree(self, rng: random.Random):
        """Knoten v mit P(v) = deg(v)/sum(deg) -- unabhaengig gezogen.

        Nicht zu verwechseln mit "gleichverteiltes u, dann zufaelliger Nachbar
        von u": das liefert P(v) = (1/N) * sum_{u->v} 1/deg(u) (das
        Freundschaftsparadox) und ist *nicht* proportional zum Grad.

        `indptr` ist bereits die Praefixsumme der Grade -- es braucht also
        keine zusaetzliche Struktur, nur eine Binaersuche darin.
        """
        total = int(self.indptr[-1])
        if total == 0:
            return None
        r = rng.randrange(total)
        return int(np.searchsorted(self.indptr, r, side="right")) - 1

    def __repr__(self) -> str:
        return (f"Graph(name={self.name!r}, view={self.view!r}, "
                f"n_nodes={self.n_nodes}, n_edges={self.n_edges})")


def _node_ids(adjacency: dict) -> list:
    """Vollstaendige Knotenliste: dict-Keys, dann reine Nachbar-Referenzen.

    Die dangling Knoten werden nach str sortiert angehaengt, damit die
    ID-Vergabe -- und damit jedes indexbasierte Ziehen -- prozessuebergreifend
    reproduzierbar ist (die Iterationsreihenfolge eines set ist es nicht).
    """
    keys = list(adjacency)
    known = set(keys)
    dangling = set()
    for nbrs in adjacency.values():
        for v in nbrs:
            if v not in known:
                known.add(v)
                dangling.add(v)
    return keys + sorted(dangling, key=str)


def load_pickle(path: Path) -> Graph:
    """Laedt eine .pkl-Adjazenzliste und baut daraus die CSR-Struktur.

    Das dict wird danach freigegeben -- ab hier lebt der Graph nur noch in den
    drei Arrays plus der Namensliste. Die .pkl-Datei wird nie geschrieben.
    """
    with open(path, "rb") as f:
        adjacency = pickle.load(f)

    names = _node_ids(adjacency)
    index = {k: i for i, k in enumerate(names)}
    keys = list(adjacency)

    n = len(names)
    deg = np.zeros(n, dtype=PTR_DTYPE)
    for k in keys:
        deg[index[k]] = len(adjacency[k])

    indptr = np.zeros(n + 1, dtype=PTR_DTYPE)
    np.cumsum(deg, out=indptr[1:])

    # Ein einziger Durchlauf auf C-Ebene statt einer Python-Schleife ueber
    # alle Kanten. keys ist die ID-Reihenfolge der ersten len(keys) Knoten,
    # deshalb passt die Reihenfolge zu indptr.
    indices = np.fromiter(
        (index[v] for k in keys for v in adjacency[k]),
        dtype=ID_DTYPE, count=int(indptr[-1]),
    )

    del adjacency, index
    return Graph(indptr, indices, names, name=path.stem)
