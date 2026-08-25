"""Graph-Wrapper um die originale Adjazenzliste (dict) aus den .pkl-Dateien.

Knoten behalten ihren Original-Key (int oder str), damit Ergebnisse und
Besuchsstatistiken direkt lesbar sind.

Zur Knotenmenge zaehlen **alle** vorkommenden Knoten, auch solche ohne
ausgehende Kanten ("dangling": sie stehen nur in fremden Nachbarlisten). Diese
werden bewusst *nicht* ins Adjazenz-dict eingefuegt -- die Original-Struktur
bleibt unangetastet und der Speicher gespart. Sie stehen nur in `nodes`;
`neighbors()` liefert fuer sie ueber `.get(u, ())` automatisch ein leeres Tupel.

Schnittstelle:
    class Graph
        .name, .adjacency, .nodes            (alle Original-Keys, inkl. dangling)
        .n_nodes                             (== wahre Groesse |V|)
        .n_with_out_edges                    (Eintraege im Adjazenz-dict)
        .neighbors(u) -> tuple               (leer, falls u unbekannt)
        .degree(u) -> int
        .random_node(rng), .random_neighbor(u, rng)
        .random_node_by_degree(rng)          (P(v) ~ deg(v), Praefixsummen gecacht)
        .view                                (Name der Kantensicht, s. graphs.views)
    all_nodes(adjacency) -> list          (Keys + dangling, deterministisch)
    load_pickle(path, normalize=True) -> Graph
"""

from __future__ import annotations

import bisect
import itertools
import pickle
import random
from pathlib import Path


class Graph:
    view = "directed"

    def __init__(self, adjacency: dict, name: str, nodes: list | None = None) -> None:
        self.adjacency = adjacency
        self.name = name
        # nodes=None -> nur die dict-Keys. Fuer die volle Knotenmenge (inkl.
        # dangling) all_nodes() benutzen, siehe load_pickle().
        self.nodes = list(adjacency) if nodes is None else nodes
        self._cum_deg: list | None = None   # lazy, s. random_node_by_degree()

    @property
    def n_nodes(self) -> int:
        """Wahre Graph-Groesse |V|: alle vorkommenden Knoten, auch solche ohne
        ausgehende Kanten."""
        return len(self.nodes)

    @property
    def n_with_out_edges(self) -> int:
        """Knoten mit eigenem Eintrag im Adjazenz-dict."""
        return len(self.adjacency)

    def neighbors(self, u) -> tuple:
        return self.adjacency.get(u, ())

    def degree(self, u) -> int:
        return len(self.adjacency.get(u, ()))

    def random_node(self, rng: random.Random):
        return self.nodes[rng.randrange(len(self.nodes))]

    def random_node_by_degree(self, rng: random.Random):
        """Knoten v mit P(v) = deg(v)/sum(deg) -- unabhaengig gezogen.

        Nicht zu verwechseln mit "gleichverteiltes u, dann zufaelliger Nachbar
        von u": das liefert P(v) = (1/N) * sum_{u->v} 1/deg(u) (das
        Freundschaftsparadox) und ist *nicht* proportional zum Grad.

        Die Praefixsummen werden einmal je Graph/View gebaut und gehalten:
        O(|V|) Speicher, dafuer O(log |V|) je Ziehung.
        """
        if self._cum_deg is None:
            self._cum_deg = list(itertools.accumulate(
                len(self.adjacency.get(u, ())) for u in self.nodes))
        total = self._cum_deg[-1]
        if total == 0:
            return None
        return self.nodes[bisect.bisect_right(self._cum_deg, rng.randrange(total))]

    def random_neighbor(self, u, rng: random.Random):
        nbrs = self.neighbors(u)
        if not nbrs:
            return None
        return nbrs[rng.randrange(len(nbrs))]

    def __repr__(self) -> str:
        return (f"Graph(name={self.name!r}, view={self.view!r}, "
                f"n_nodes={self.n_nodes}, n_with_out_edges={self.n_with_out_edges})")


def all_nodes(adjacency: dict) -> list:
    """Vollstaendige Knotenmenge: dict-Keys plus reine Nachbar-Referenzen.

    Die dangling Knoten werden nach str sortiert angehaengt, damit die
    Reihenfolge -- und damit jedes indexbasierte Ziehen -- prozessuebergreifend
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


def load_pickle(path: Path, normalize: bool = True) -> Graph:
    """Laedt eine .pkl-Adjazenzliste.

    normalize=True ersetzt die Nachbar-Container in-place durch Tupel: das
    macht sie indizierbar (fuer Random Walks) und spart gegenueber sets
    deutlich Speicher. Das betrifft nur die geladene Kopie im Speicher -- die
    .pkl-Datei wird nie geschrieben, und es kommen keine Keys hinzu.
    """
    with open(path, "rb") as f:
        adjacency = pickle.load(f)

    if normalize:
        for k in adjacency:
            v = adjacency[k]
            if not isinstance(v, tuple):
                adjacency[k] = tuple(v)

    return Graph(adjacency, name=path.stem, nodes=all_nodes(adjacency))
