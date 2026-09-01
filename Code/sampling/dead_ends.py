"""Strategien fuer Sackgassen im Random Walk (Knoten ohne ausgehende Kanten).

Bei den gerichteten Testgraphen ist das kein Randfall: bei gpt4o_io haben ueber
50 % der Knoten keine ausgehenden Kanten. Wie der Walk dort weitermacht,
beeinflusst Abdeckung und Kollisionsrate stark -- deshalb austauschbar.

**Nur die gerichteten Views sind betroffen.** In `undirected` (graphs.views)
gilt: erreicht der Walk einen Knoten ueber eine Kante, existiert diese Kante
auch zurueck -- ein *erreichter* Knoten kann dort also nie eine Sackgasse
sein. Offen bleibt allein ein voellig isolierter Knoten (weder Aus- noch
Eingangskanten), der als Seed gezogen wird; in den Testgraphen kommt das nicht
vor. Praktisch wird keine der Strategien unten je aufgerufen. Ausgezaehlt auf
Slashdot0811:

    View         Grad 0            nur Selbstkante      Sackgassen gesamt
    directed     44 (0,06 %)       6418 (8,30 %)        8,35 %
    undirected   0                 0                    0,00 %

Praktische Folge fuer Experimente: ein Vergleich der drei Strategien ist nur
auf `directed` (oder `reverse`) aussagekraeftig. Auf `undirected` laufen sie
denselben Algorithmus. Beobachtete Unterschiede sind dort reines RNG-Rauschen,
weil der Seed in experiment.runner den Estimator-Namen enthaelt und die
Varianten deshalb auf verschiedenen Zufallsstroemen laufen -- mit
`--share-walks` faellt auch das weg, dort liefern die drei Strategien auf
`undirected` exakt dieselben Zahlen.

Sackgasse heisst: *gar kein* nutzbarer Nachbar. Ein Knoten, dessen Nachbarn
alle schon besucht sind, ist keine -- Wiederbesuche sind normales Laufen und
fuer Collision Counting sogar der Zweck der Uebung. Siehe
sampling.samplers.RandomWalkSampler._step().

Aufrufkonvention: `path` ist der aktuelle Pfad (Backtracking kuerzt ihn),
`trace` die vollstaendige Besuchsfolge, `start` der Startknoten des Walks.
Abfragen, die eine Strategie selbst stellt, laufen ueber das Oracle und kosten
damit Budget -- sie erzeugen aber keine Samples: sie sind Navigation, keine
Beobachtung.

Schnittstelle:
    class DeadEndStrategy
        .name, .next_node(oracle, path, trace, start) -> node
    RestartToStart, Backtrack, HistoryJump
    DEAD_ENDS: dict[str, type[DeadEndStrategy]]
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class DeadEndStrategy(ABC):
    name: str = "dead_end"

    @abstractmethod
    def next_node(self, oracle, path: list, trace: list, start):
        """Naechster Knoten, nachdem der Walk in einer Sackgasse gelandet ist."""


class RestartToStart(DeadEndStrategy):
    """Sprung zurueck zum Anfang des Walks."""

    name = "restart"

    def next_node(self, oracle, path: list, trace: list, start):
        path.clear()
        return start


class Backtrack(DeadEndStrategy):
    """Schritte zurueck, bis ein Vorgaenger eine andere Abzweigung anbietet.

    Der Pfad wird dabei bis zu diesem Vorgaenger gekuerzt. Jede Rueckfrage nach
    dessen Nachbarn kostet Budget.
    """

    name = "backtrack"

    def next_node(self, oracle, path: list, trace: list, start):
        while len(path) >= 2:
            child = path.pop()          # die Sackgasse bzw. der schon genommene Zweig
            u = path[-1]
            nbrs = oracle.neighbors(u)
            # vektorisiert statt Python-Schleife: bei gpt4o_io haben ueber 50 %
            # der Knoten keine ausgehenden Kanten, dieser Zweig laeuft also
            # staendig.
            alternatives = nbrs[nbrs != child]
            if len(alternatives):
                return alternatives[oracle.rng.randrange(len(alternatives))]
        path.clear()
        return start


class HistoryJump(DeadEndStrategy):
    """Zufaellige Umverteilung auf die History des Walks.

    Gezogen wird gleichverteilt aus der bisherigen Besuchsfolge -- mit
    Vielfachheit, oft besuchte Knoten kommen also entsprechend haeufiger dran - so wie das NMMC fordert.
    """

    name = "history"

    def next_node(self, oracle, path: list, trace: list, start):
        if not trace:
            return start
        return trace[oracle.rng.randrange(len(trace))].node


DEAD_ENDS: dict[str, type[DeadEndStrategy]] = {
    "restart": RestartToStart,
    "backtrack": Backtrack,
    "history": HistoryJump,
}
