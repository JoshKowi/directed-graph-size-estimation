"""Graph-Views: dieselbe Knotenmenge, andere Sicht auf die Kanten.

Damit laesst sich derselbe Estimator auf dem gerichteten und dem
symmetrisierten Graphen laufen lassen und direkt vergleichen. Die Knotenmenge
(und damit die wahre Groesse |V|) bleibt in allen Views identisch -- nur so
sind die Schaetzungen vergleichbar. Auch die IDs bleiben dieselben: alle Views
teilen sich `names` des Basisgraphen.

    directed    -- Originalgraph, Nachbarn = Ausgangskanten
    undirected  -- Nachbarn = Aus- und Eingangskanten (symmetrisiert)
    reverse     -- Nachbarn = nur Eingangskanten

Die Umbauten laufen vollstaendig in numpy (kein Python-Loop ueber Kanten).
Speicher: `undirected` legt zwischenzeitlich zwei int64-Arrays der doppelten
Kantenzahl an; bei sehr grossen Graphen ist das der Spitzenverbrauch.

Schnittstelle:
    VIEWS: dict[str, Callable[[Graph], Graph]]
    build_view(graph, view) -> Graph
    class UndirectedView(Graph), class ReverseView(Graph)
"""

from __future__ import annotations

import numpy as np

from graphs.graph import ID_DTYPE, PTR_DTYPE, Graph


def _sources(graph: Graph) -> np.ndarray:
    """Quellknoten je Kante, passend zur Reihenfolge in `indices`."""
    return np.repeat(np.arange(graph.n_nodes, dtype=ID_DTYPE), np.diff(graph.indptr))


def _csr_from_counts(counts: np.ndarray) -> np.ndarray:
    indptr = np.zeros(len(counts) + 1, dtype=PTR_DTYPE)
    np.cumsum(counts, out=indptr[1:])
    return indptr


def _reverse_csr(graph: Graph):
    """Eingangs-Nachbarschaften."""
    n = graph.n_nodes
    counts = np.bincount(graph.indices, minlength=n)
    indptr = _csr_from_counts(counts)
    # Kanten nach Ziel sortieren; an jeder Position steht dann die Quelle.
    order = np.argsort(graph.indices, kind="stable")
    indices = _sources(graph)[order]
    return indptr, indices


def _symmetric_csr(graph: Graph, dedup: bool = True):
    """Vereinigung aus Aus- und Eingangskanten."""
    n = graph.n_nodes
    src = _sources(graph).astype(np.int64)
    dst = graph.indices.astype(np.int64)
    u = np.concatenate((src, dst))
    v = np.concatenate((dst, src))
    del src, dst

    # (u, v) in eine Zahl packen, dann sortieren/deduplizieren: danach ist die
    # Liste bereits nach u und innerhalb von u nach v geordnet -- genau das
    # CSR-Layout, ohne je eine Python-Schleife.
    key = u * n + v
    del u, v
    key = np.unique(key) if dedup else np.sort(key)
    counts = np.bincount(key // n, minlength=n)
    return _csr_from_counts(counts), (key % n).astype(ID_DTYPE)


class ReverseView(Graph):
    """Nachbarn = Eingangskanten des Originalgraphen."""

    def __init__(self, base: Graph) -> None:
        indptr, indices = _reverse_csr(base)
        super().__init__(indptr, indices, base.names, base.name)
        self.view = "reverse"


class UndirectedView(Graph):
    """Nachbarn = Vereinigung aus Aus- und Eingangskanten.

    dedup=True (Default) macht daraus einen einfachen Graphen; mit dedup=False
    bleiben reziproke Kanten doppelt (Multigraph). Beides ist fuer den
    Random-Walk-Schaetzer konsistent, solange Grad und Walk dieselbe Sicht
    benutzen -- dedup=True entspricht dem ueblichen "ungerichteten Graphen".

    Nebenwirkung auf den Random Walk: hier gibt es praktisch keine Sackgassen
    mehr (jede Kante ist auch rueckwaerts begehbar), die `dead_end`-Strategien
    aus sampling.dead_ends laufen also ins Leere. Details dort im Docstring.
    """

    def __init__(self, base: Graph, dedup: bool = True) -> None:
        indptr, indices = _symmetric_csr(base, dedup=dedup)
        super().__init__(indptr, indices, base.names, base.name)
        self.view = "undirected"


class DirectedView(Graph):
    """Originalgraph, unveraendert (nur damit alle Views gleich aussehen)."""

    def __init__(self, base: Graph) -> None:
        super().__init__(base.indptr, base.indices, base.names, base.name)
        self.view = "directed"


VIEWS = {
    "directed": DirectedView,
    "undirected": UndirectedView,
    "reverse": ReverseView,
}


def build_view(graph: Graph, view: str) -> Graph:
    if view not in VIEWS:
        raise ValueError(f"Unbekannte View {view!r} -- bekannt: {sorted(VIEWS)}")
    return VIEWS[view](graph)
