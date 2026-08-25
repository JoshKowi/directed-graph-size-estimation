"""Oracles mit lokalem Zugriff: nur Nachbarschafts-Abfragen ausgehend von
bekannten Einstiegsknoten, kein Zugriff auf die Knotenmenge V. Das entspricht
dem, was sich real crawlen laesst.

Schnittstelle:
    class CrawlOracle(Oracle)   -- seed_nodes(), neighbors(), degree()
"""

from __future__ import annotations

from oracles.base import Oracle


class CrawlOracle(Oracle):
    """Crawl-Zugriff: Nachbarn eines bekannten Knotens abfragen.

    seed_nodes() liefert Einstiegspunkte und kostet COST_RANDOM_NODE je Seed.
    Die Seeds werden hier der Reproduzierbarkeit halber zufaellig gezogen -- in
    einer echten Anwendung waeren es fest bekannte Knoten, ihre Beschaffung
    also gratis: dann config.COST_RANDOM_NODE = 0 setzen.
    """

    def seed_nodes(self, k: int = 1) -> list:
        return [self._draw() for _ in range(k)]

    def neighbors(self, u) -> tuple:
        return self._fetch(u)

    def degree(self, u) -> int:
        return len(self._fetch(u))
