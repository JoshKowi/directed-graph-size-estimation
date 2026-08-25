"""Oracle-Basis: gekapselter Graph-Zugriff mit Kostenzaehlung und Budget.

Getrennte Zugriffsarten mit eigenen, frei einstellbaren Preisen (config.COST_*):

    _draw()     "gib mir einen zufaelligen Knoten"   -- COST_RANDOM_NODE
    _fetch(u)   Nachbarn von u, erster Zugriff       -- COST_NEIGHBORS
    _fetch(u)   Nachbarn von u, Cache-Treffer        -- COST_CACHE_HIT

Cache: `_fetch` kostet beim ersten Zugriff auf einen Knoten den vollen Preis,
danach nur noch COST_CACHE_HIT -- ein realer Crawler haelt die einmal geholte
Nachbarschaft, muss sie aber weiterhin nachschlagen. `_draw` ist dagegen nicht
cachebar: jede Ziehung ist eine neue Anfrage, auch wenn zufaellig derselbe
Knoten herauskommt. Mehrere Oracles koennen sich einen Cache teilen (Argument
`cache`), wenn ein Estimator mehrere Crawler nacheinander oder parallel
laufen laesst -- siehe estimators/methods/capture_recapture.py.

Dass ein Cache-Treffer *etwas* kostet, ist die entscheidende Eigenschaft: mit
Preis 0 wuerde ein Walk, der sich in einer kleinen bekannten Region verfaengt,
endlos gratis weiterlaufen. So terminiert das Budget jeden Lauf von selbst und
jeder Estimator schoepft es aus -- erst dadurch sind "erlaubtes" und
"genutztes" Budget vergleichbare Groessen.

Gezaehlt wird:
    queries        -- bezahlte Kosten, gewichtet (die Budget-Waehrung)
    unique_nodes   -- Anzahl verschiedener beruehrter Knoten (nur Statistik)
    cached_queries -- Nachbar-Abfragen, die der Cache beantwortet hat
    n_random_node / n_neighbors -- Zugriffe je Art zum vollen Preis
    stopped_by     -- warum der Lauf endete, meist "budget"

Ein globales Aufruf-Limit gibt es bewusst *nicht*. Weil jeder Zugriff einen
Preis > 0 hat -- auch der Cache-Treffer -- terminiert das Budget jeden Lauf von
selbst. Ein hergeleiteter Schritt-Deckel waere per Konstruktion nie bindend und
wuerde nur verschleiern, wo tatsaechlich Schluss ist. Wo eine Schleife nicht
ueber den Preis endet, wird sie an ihrer eigenen Stelle begrenzt (siehe
oracles.global_access.ShortWalkIndependentOracle) und meldet das ueber
`stopped_by`.

Deshalb ist budget_metric="unique_nodes" nicht mehr zulaessig: diese Metrik
waechst bei einem Walk, der nur bekannte Knoten besucht, gar nicht mehr, und
der Lauf wuerde nie enden. `unique_nodes` bleibt als Statistik erhalten.

Welche Methoden ein Oracle anbietet, entscheidet den Zugriffsumfang und damit,
ob ein darauf aufbauender Estimator real umsetzbar ist (siehe
oracles.global_access / oracles.local_access). Die Kategorie selbst wird erst
in estimators/__init__.py vergeben.

Schnittstelle:
    class BudgetExceeded(Exception)
    class Oracle
        .graph, .rng, .visits, .queries, .unique_nodes, .remaining
        .cached_queries, .n_random_node, .n_neighbors, .stopped_by
        .cost() -> dict[str, int]
        .neighbors(u), .degree(u), .random_node(), .seed_nodes(k)
"""

from __future__ import annotations

import random
from collections import Counter

import config
from graphs.graph import Graph


class BudgetExceeded(Exception):
    """Wird geworfen, wenn ein Zugriff das Budget ueberschreiten wuerde."""


class Oracle:
    """Basisklasse. Unterklassen erlauben/verbieten einzelne Zugriffsarten."""

    def __init__(
        self,
        graph: Graph,
        rng: random.Random,
        budget: int,
        budget_metric: str = "queries",
        cache: set | None = None,
        cost_random_node: float = config.COST_RANDOM_NODE,
        cost_neighbors: float = config.COST_NEIGHBORS,
        cost_cache_hit: float = config.COST_CACHE_HIT,
    ) -> None:
        self.graph = graph
        self.rng = rng
        self.budget = budget
        self.budget_metric = budget_metric
        self.cost_random_node = cost_random_node
        self.cost_neighbors = cost_neighbors
        self.cost_cache_hit = cost_cache_hit

        self.queries = 0          # bezahlte Kosten (gewichtet)
        self.n_random_node = 0
        self.n_neighbors = 0
        self.cached_queries = 0
        self.stopped_by: str | None = None
        if budget_metric != "queries":
            # "unique_nodes" waechst bei einem Walk in bekanntem Gebiet nicht
            # mehr -- das Budget wuerde nie erschoepft und der Lauf nie enden.
            raise ValueError(
                f"budget_metric={budget_metric!r} terminiert nicht: nur 'queries' "
                "verbraucht bei jedem Zugriff Budget. unique_nodes steht "
                "weiterhin als Statistik in cost()."
            )
        self.visits: Counter = Counter()  # Original-Knotenname -> Anzahl Zugriffe
        # Knoten, deren Nachbarschaft bereits geholt wurde. Von aussen
        # uebergeben, wenn mehrere Crawler sich den Cache teilen sollen.
        self._fetched: set = set() if cache is None else cache

    # -- Kosten -----------------------------------------------------------
    @property
    def unique_nodes(self) -> int:
        return len(self.visits)

    @property
    def remaining(self) -> float:
        used = self.queries if self.budget_metric == "queries" else self.unique_nodes
        return self.budget - used

    def cost(self) -> dict[str, int]:
        return {
            "queries": self.queries,
            "unique_nodes": self.unique_nodes,
            "cached_queries": self.cached_queries,
            "n_random_node": self.n_random_node,
            "n_neighbors": self.n_neighbors,
            "stopped_by": self.stopped_by,
        }

    # -- Buchhaltung ------------------------------------------------------
    def _charge(self, node, amount: float) -> None:
        """Bucht einen bezahlten Zugriff; prueft das Budget *vor* der Ausfuehrung."""
        queries = self.queries + amount
        if queries > self.budget:
            self.stopped_by = "budget"
            raise BudgetExceeded(f"Budget {self.budget} ({self.budget_metric}) erschoepft")
        self.queries = queries
        if node is not None:
            self.visits[node] += 1

    # -- Zugriffsarten (von Unterklassen benutzt) -------------------------
    def _draw(self):
        """Gleichverteilt gezogener Knoten. Nicht cachebar: jede Ziehung ist
        eine eigene Anfrage, auch wenn zweimal derselbe Knoten herauskommt."""
        u = self.graph.random_node(self.rng)
        self._charge(u, self.cost_random_node)
        self.n_random_node += 1
        return u

    def _draw_by_degree(self):
        """Knoten mit P(v) ~ deg(v). Wie _draw() eine eigene Anfrage nach
        aussen und ebenso wenig cachebar."""
        u = self.graph.random_node_by_degree(self.rng)
        self._charge(u, self.cost_random_node)
        self.n_random_node += 1
        return u

    def _fetch(self, u) -> tuple:
        """Nachbarn von u -- der erste Zugriff kostet voll, jeder weitere
        COST_CACHE_HIT.

        Der Cache haelt nur die Knoten-Keys: die Nachbarliste selbst steht
        ohnehin im Adjazenz-dict des Graphen und wird nicht kopiert.
        """
        if u in self._fetched:
            self._charge(u, self.cost_cache_hit)
            self.cached_queries += 1
        else:
            self._charge(u, self.cost_neighbors)
            self.n_neighbors += 1
            self._fetched.add(u)
        return self.graph.neighbors(u)

    # -- Zugriffe (Unterklassen ueberschreiben, was sie anbieten) ---------
    def neighbors(self, u) -> tuple:
        raise NotImplementedError

    def degree(self, u) -> int:
        raise NotImplementedError

    def random_node(self):
        """Gleichverteilter Knoten aus V -- in der Realitaet nicht verfuegbar."""
        raise NotImplementedError

    def seed_nodes(self, k: int = 1) -> list:
        """Bekannte Einstiegsknoten (realistischer Startpunkt eines Crawls)."""
        raise NotImplementedError
