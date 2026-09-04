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
    In einer echten Anwendung sind die Einstiegsknoten fest bekannt, ihre
    Beschaffung also gratis: dann config.COST_RANDOM_NODE = 0 setzen.

    Gestartet wird bei den festen Einstiegsknoten aus config.SEED_NODES (siehe
    README, "Entwurfsentscheidungen"). Gleichverteilt ueber V zu ziehen setzte
    voraus, dass V bereits bekannt ist -- also genau das, was geschaetzt werden
    soll. Welcher der hinterlegten Knoten es wird, entscheidet der Zufall des
    Laufs: sonst starteten alle Wiederholungen an derselben Stelle und die
    Streuung ueber die Laeufe waere kuenstlich klein.

    Fuer Graphen ohne Eintrag in SEED_NODES bleibt es beim gleichverteilten
    Ziehen.
    """

    def seed_nodes(self, k: int = 1) -> list:
        seeds = self.graph.seed_ids()
        if seeds is None:
            return [self._draw() for _ in range(k)]
        out = []
        for _ in range(k):
            u = seeds[self.rng.randrange(len(seeds))]
            # gleicher Preis wie ein Zufallsknoten, damit die Budgets zwischen
            # Verfahren mit und ohne feste Einstiegsknoten vergleichbar bleiben
            self._charge(u, self.cost_random_node)
            self.n_random_node += 1
            out.append(u)
        return out

    def neighbors(self, u) -> tuple:
        return self._fetch(u)

    def degree(self, u) -> int:
        return len(self._fetch(u))


class JumpCrawlOracle(CrawlOracle):
    """CrawlOracle plus gleichverteilte Knotenziehung -- fuer den Zufallssprung
    von DURW (sampling.jumps.UniformJump).

    Der einzige Unterschied zum CrawlOracle ist random_node(). Damit steht
    dieses Oracle bewusst zwischen den beiden Zugriffsmodellen: gecrawlt wird
    weiterhin nur ueber Nachbarschaftsabfragen, aber der Sprung setzt voraus,
    dass sich ein Knoten gleichverteilt aus V ziehen laesst. Ob das als real
    umsetzbar gilt, haengt daran, wie der Sprung in der Anwendung beschafft
    wird -- die Kategorie wird deshalb nicht hier, sondern in
    estimators/__init__.py vergeben.

    Ein Sprung kostet COST_RANDOM_NODE, ein Schritt COST_NEIGHBORS (bzw.
    COST_CACHE_HIT beim Wiederbesuch): das Sprunggewicht w steuert damit
    zugleich, wie sich das Budget auf Sprungen und Schritte verteilt.
    """

    def random_node(self):
        return self._draw()
