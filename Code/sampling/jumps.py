"""Sprungstrategien fuer DURW: wohin der Walker springt, wenn die
Sprungregel w/(w + deg) zieht.

Der Sprung ist bei DURW keine Notloesung wie eine Sackgassen-Strategie
(sampling.dead_ends), sondern Teil des Verfahrens: er ist die Kante zum
virtuellen Knoten sigma aus Ribeiro & Towsley. Erst dadurch ist die
Stationaerverteilung des Walks

    pi(v) = (w + deg_Gu(v)) / (vol(V) + w|V|)

geschlossen bekannt -- bis auf die Normierung, die der Kollisionsschaetzer
ohnehin herauskuerzt. Der virtuelle Knoten selbst wird nie gebaut: ein Sprung
ist einfach eine Ziehung, und das Sprungziel ist danach ein ganz gewoehnlicher
Zustand des Walks -- es wird bemustert und in G_u eingetragen wie jeder andere
besuchte Knoten (siehe sampling.durw).

Welche Sprungart ein Estimator benutzt, entscheidet zugleich, welches Oracle er
braucht und damit, ob er real umsetzbar ist. Die Zuordnung Sprungart -> Oracle
steht in estimators/methods/durw.py, die Kategorie -- wie im Repo ueblich --
erst in estimators/__init__.py.

Abfragen, die eine Strategie stellt, laufen ueber das Oracle und kosten Budget.

Schnittstelle:
    class JumpStrategy
        .name, .next_node(oracle) -> node
    UniformJump
    JUMPS: dict[str, type[JumpStrategy]]
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class JumpStrategy(ABC):
    name: str = "jump"

    @abstractmethod
    def next_node(self, oracle):
        """Ziel des Zufallssprungs."""


class UniformJump(JumpStrategy):
    """Gleichverteilt gezogener Knoten aus V -- der Sprung des Papers.

    Braucht ein Oracle mit random_node() (oracles.local_access.JumpCrawlOracle)
    und kostet config.COST_RANDOM_NODE. Ribeiro & Towsley rechtfertigen die
    Annahme damit, dass sich ein gleichverteilter Knoten oft durch Rejection
    Sampling im ID-Raum ziehen laesst, ohne |V| zu kennen. Ob man das als real
    umsetzbar gelten laesst, entscheidet die Kategorie in
    estimators/__init__.py -- nicht diese Klasse.
    """

    name = "uniform"

    def next_node(self, oracle):
        return oracle.random_node()


JUMPS: dict[str, type[JumpStrategy]] = {
    "uniform": UniformJump,
}
