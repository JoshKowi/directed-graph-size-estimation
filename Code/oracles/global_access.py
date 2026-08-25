"""Oracles mit globalem Zugriff: sie setzen Kenntnis der gesamten Knotenmenge
voraus (gleichverteiltes Ziehen aus V). Estimators, die darauf aufbauen, sind
bei uns typischerweise nicht real umsetzbar -- die Kategorie steht aber in der
REGISTRY, nicht hier.

Schnittstelle:
    class UniformNodeOracle(Oracle)            -- random_node(), degree(), neighbors()
    class DegWeightedIndependentOracle(Oracle) -- wie oben, aber pi(u) ~ deg(u)
    class ShortWalkIndependentOracle(Oracle)   -- Endknoten kurzer freier Walks
"""

from __future__ import annotations

from oracles.base import BudgetExceeded, Oracle


class UniformNodeOracle(Oracle):
    """Zieht gleichverteilt aus allen Knoten."""

    def random_node(self):
        return self._draw()

    def degree(self, u) -> int:
        return len(self._fetch(u))

    def neighbors(self, u) -> tuple:
        return self._fetch(u)


class DegWeightedIndependentOracle(Oracle):
    """Simuliert die Gradverzerrung eines Random-Walk-Schritts, aber ohne
    dessen Abhaengigkeit zwischen aufeinanderfolgenden Samples.

    Zieht jedes Sample direkt mit P(v) = deg(v)/sum(deg) -- dieselbe
    Verteilung, die ein Random Walk auf einem ungerichteten, zusammenhaengenden
    Graphen im Grenzwert erreicht, aber ohne Abhaengigkeit zwischen
    aufeinanderfolgenden Samples.

    **Auf gerichteten Graphen sieht dieses Oracle einen Teil der Knoten nie.**
    Gezogen wird mit P(v) ~ deg_out(v); Knoten ohne ausgehende Kanten haben
    P(v) = 0 und koennen prinzipiell nicht als Sample auftreten. Der Schaetzer
    schaetzt dann korrekt die Groesse von {v : deg_out(v) > 0} -- nicht |V|.

    Bei gpt4o_io ist das die Haelfte des Graphen: 2 657 109 von 5 693 001
    Knoten haben ausgehende Kanten, Anteil 0.4667. Gemessen liefert
    wis-katzir__indep dort ueber alle Budgets stabil 0.4635 x |V|. Auf
    Slashdot0811 faellt es nicht auf (0,06 % Knoten ohne Ausgangskanten), auf
    der symmetrisierten Sicht ebenfalls nicht (dort hat jeder erreichte Knoten
    mindestens die Rueckkante).

    ACHTUNG, frueherer Fehler: hier stand "gleichverteiltes u, dann
    gleichverteilter Nachbar von u". Das liefert
    P(v) = (1/N) * sum_{u->v} 1/deg(u) -- das Freundschaftsparadox, *nicht*
    pi(v) ~ deg(v). Damit war w_i = 1/deg(v) nicht mehr proportional zu 1/pi,
    und Katzirs Schaetzer wurde graphabhaengig verzerrt: auf Slashdot0811
    nachgerechnet Faktor 2.602 (gerichtet) bzw. 1.939 (symmetrisiert) -- was
    die damals gemessenen Mediane 2.599 / 1.942 exakt erklaert. Mit
    pi ~ deg ergibt dieselbe Rechnung 0.999 / 1.000.
    """

    def random_node(self):
        return self._draw_by_degree()

    def degree(self, u) -> int:
        return len(self._fetch(u))

    def neighbors(self, u) -> tuple:
        return self._fetch(u)


class ShortWalkIndependentOracle(Oracle):
    """Unabhaengige Samples aus kurzen Random Walks fester Laenge.

    Pro Sample: gleichverteilter Startknoten, dann `steps` Schritte, der
    Endknoten ist das Sample. Die Samples sind untereinander unabhaengig (jeder
    Walk startet neu), tragen aber die Verzerrung, die ein Random Walk nach
    `steps` Schritten aufgebaut hat.

    Zweck: zeigen, dass die Gradgewichtung auf *gerichteten* Graphen auch dann
    nicht funktioniert, wenn die Abhaengigkeit zwischen den Samples weg ist.
    Auf ungerichteten Graphen konvergiert die Verteilung eines Random Walks
    gegen pi(v) ~ deg(v), dort passt w_i = 1/deg(v). Auf gerichteten gilt das
    nicht -- die Verteilung nach k Schritten haengt von der Struktur ab und ist
    keine einfache Funktion des Ausgangsgrades. Der Vergleich der beiden Views
    isoliert genau diesen Effekt, weil alles andere gleich bleibt.

    Sackgassen (kein nutzbarer Nachbar): der Walk geht einen Schritt zurueck
    und schliesst den gerade probierten Zweig aus. Ist der ganze Pfad
    ausgeschoepft, wird ein neuer Startknoten gezogen.

    Kosten: **ein Sample = eine Query** (COST_RANDOM_NODE). Die Schritte, das
    Backtracking und die Gradabfrage des gelieferten Knotens sind gratis -- die
    Antwort liefert die Nachbarliste gleich mit.

    Terminierung: die Suche je Sample endet immer, weil das Backtracking mit
    `tried` jeden Zweig nur einmal probiert und der Pfad auf `steps` begrenzt
    ist -- durchsucht wird also hoechstens der erreichbare Teilgraph. Was nicht
    von selbst endet, ist das *Neuziehen*: findet kein Startknoten einen Pfad
    der Laenge `steps` (etwa in einem flachen DAG ohne Zyklen), liefe die
    aeussere Schleife ewig. Deshalb `max_restarts`; greift die Grenze, endet
    der Lauf mit stopped_by="no_path" statt stillschweigend weiterzudrehen.
    """

    #: aufeinanderfolgende erfolglose Startknoten, bevor aufgegeben wird
    max_restarts = 1000

    def __init__(self, *args, steps: int = 5, allow_self_loops: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.steps = steps
        self.allow_self_loops = allow_self_loops
        # Knoten, die als Sample geliefert wurden -- ihre Nachbarliste kam mit.
        self._delivered: set = set()

    def _walk(self):
        """Endknoten eines freien Walks der Laenge `steps`. Kostet nichts."""
        for _ in range(self.max_restarts):
            path = [self.graph.random_node(self.rng)]
            tried: list[set] = [set()]      # je Pfadposition die schon probierten Zweige
            while len(path) <= self.steps:
                u = path[-1]
                nbrs = self.graph.neighbors(u)
                if not self.allow_self_loops:
                    nbrs = nbrs[nbrs != u]
                options = [v for v in nbrs if v not in tried[-1]] if tried[-1] else nbrs
                if not len(options):
                    if len(path) == 1:
                        break               # Startknoten ist selbst Sackgasse
                    dead = path.pop()
                    tried.pop()
                    tried[-1].add(dead)     # diesen Zweig nicht noch einmal
                    continue
                path.append(options[self.rng.randrange(len(options))])
                tried.append(set())
            if len(path) == self.steps + 1:
                return path[-1]
        self.stopped_by = "no_path"
        raise BudgetExceeded(
            f"kein Pfad der Laenge {self.steps} nach {self.max_restarts} "
            "Startknoten gefunden"
        )

    def random_node(self):
        v = self._walk()
        self._charge(v, self.cost_random_node)
        self._delivered.add(v)
        self.n_random_node += 1
        return v

    def neighbors(self, u) -> tuple:
        if u in self._delivered:
            return self.graph.neighbors(u)  # kam mit dem Sample -> gratis
        return self._fetch(u)

    def degree(self, u) -> int:
        return len(self.neighbors(u))
