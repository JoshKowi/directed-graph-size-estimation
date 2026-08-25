"""Graph-Views: dieselbe Knotenmenge, andere Sicht auf die Kanten.

Damit laesst sich derselbe Estimator auf dem gerichteten und dem
symmetrisierten Graphen laufen lassen und direkt vergleichen. Die Knotenmenge
(und damit die wahre Groesse |V|) bleibt in allen Views identisch -- nur so
sind die Schaetzungen vergleichbar. Sie umfasst wie im Basisgraphen alle
vorkommenden Knoten, auch die ohne ausgehende Kanten.

    directed    -- Originalgraph, Nachbarn = Ausgangskanten
    undirected  -- Nachbarn = Aus- und Eingangskanten (symmetrisiert)
    reverse     -- Nachbarn = nur Eingangskanten

Speicher: `undirected` und `reverse` bauen einmalig einen Reverse-Index bzw.
eine symmetrisierte Adjazenz auf. Bei den grossen Graphen (>1 GB Pickle) heisst
das grob doppelter Speicherbedarf, solange beide Views gleichzeitig leben --
notfalls die Views in getrennten Laeufen fahren (`--views undirected`).

Schnittstelle:
    VIEWS: dict[str, Callable[[Graph], Graph]]
    build_view(graph, view) -> Graph
    class UndirectedView(Graph), class ReverseView(Graph)
"""

from __future__ import annotations

from graphs.graph import Graph


def _reverse_adjacency(adjacency: dict) -> dict:
    """Eingangs-Nachbarschaften. Knoten ohne Eingangskanten fehlen im dict."""
    rev: dict = {}
    for u, nbrs in adjacency.items():
        for v in nbrs:
            rev.setdefault(v, []).append(u)
    for k in rev:
        rev[k] = tuple(rev[k])
    return rev


class ReverseView(Graph):
    """Nachbarn = Eingangskanten des Originalgraphen."""

    def __init__(self, base: Graph) -> None:
        # Knotenmenge bleibt die des Basisgraphen, damit |V| vergleichbar ist.
        # Knoten ohne Eingangskanten fehlen im dict und haben ueber .get() Grad 0.
        super().__init__(_reverse_adjacency(base.adjacency), base.name, nodes=base.nodes)
        self.view = "reverse"


class UndirectedView(Graph):
    """Nachbarn = Vereinigung aus Aus- und Eingangskanten.

    Nebenwirkung auf den Random Walk: hier gibt es praktisch keine Sackgassen
    mehr (jede Kante ist auch rueckwaerts begehbar), die `dead_end`-Strategien
    aus sampling.dead_ends laufen also ins Leere. Details dort im Docstring.

    dedup=True (Default) macht daraus einen einfachen Graphen; mit dedup=False
    bleiben reziproke Kanten doppelt (Multigraph). Beides ist fuer den
    Random-Walk-Schaetzer konsistent, solange Grad und Walk dieselbe Sicht
    benutzen -- dedup=True entspricht dem ueblichen "ungerichteten Graphen".
    """

    def __init__(self, base: Graph, dedup: bool = True) -> None:
        rev = _reverse_adjacency(base.adjacency)
        sym: dict = {}
        # ueber base.nodes iterieren statt ueber eine Vereinigungs-Menge: die
        # Knotenmenge ist dort bereits vollstaendig, das spart den Speicher-Peak.
        for u in base.nodes:
            out = base.adjacency.get(u, ())
            inc = rev.get(u, ())
            sym[u] = tuple(dict.fromkeys(out + inc)) if dedup else tuple(out) + tuple(inc)
        del rev

        super().__init__(sym, base.name, nodes=base.nodes)
        self.view = "undirected"


class DirectedView(Graph):
    """Originalgraph, unveraendert (nur damit alle Views gleich aussehen)."""

    def __init__(self, base: Graph) -> None:
        super().__init__(base.adjacency, base.name, nodes=base.nodes)
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
