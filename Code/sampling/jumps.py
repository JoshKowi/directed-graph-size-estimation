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
    UniformJump, NameListJump, RandomSubsetJump
    JUMPS: dict[str, Callable[[], JumpStrategy]]
    subset_percent(name) -> float | None     -- "rand10" -> 10.0
    jump_strategy(name) -> JumpStrategy       -- JUMPS plus rand<P>
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial

import re

import namelists


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


class NameListJump(JumpStrategy):
    """Sprung auf einen Knoten aus einer externen Namensliste.

    Ruft dieselbe Methode wie UniformJump -- *wie* gezogen wird, entscheidet
    das Oracle (oracles.name_list.NameListOracle), nicht diese Klasse. Sie
    traegt nur den Namen, damit Estimator- und Walk-Schluessel die Quelle
    ausweisen; sonst hiessen zwei voellig verschiedene Verfahren gleich.

    Anders als UniformJump setzt sie *keine* Kenntnis von V voraus: die Liste
    ist externes Wissen. Dafuer trifft sie nicht jeden Knoten -- die Abdeckung
    liegt je nach Graph und Liste bei 8 bis 38 %, und die erreichbare Teilmenge
    ist gradverzerrt. Was das fuer die Stationaerverteilung bedeutet, steht im
    Docstring von oracles.name_list.
    """

    def __init__(self, source: str) -> None:
        self.source = source
        self.name = f"list-{source}"

    def next_node(self, oracle):
        return oracle.random_node()


class RandomSubsetJump(JumpStrategy):
    """Sprung auf eine feste Zufallsteilmenge S mit P % der Knoten.

    Wie NameListJump nur ein Name -- gezogen wird in
    oracles.random_subset.RandomSubsetOracle. Gedacht als Stellvertreter einer
    Namensliste auf Graphen, fuer die es keine gibt, und als Gegenprobe ohne
    Gradverzerrung: S ist gleichverteilt, nur ihre Groesse ist eingestellt.
    """

    def __init__(self, percent: float) -> None:
        self.percent = float(percent)
        self.name = f"rand{self.percent:g}"

    def next_node(self, oracle):
        return oracle.random_node()


# Die Sprungarten. Die Listenquellen kommen aus namelists.SOURCES, damit eine
# neue Liste nur dort eingetragen werden muss.
JUMPS: dict[str, Callable[[], JumpStrategy]] = {
    "uniform": UniformJump,
}
for _src in sorted(namelists.SOURCES):
    JUMPS[_src] = partial(NameListJump, source=_src)
del _src


# Zufallsteilmengen stehen nicht in JUMPS: der Anteil ist frei waehlbar
# ("rand10", "rand0.5"), die Namen werden deshalb hier aufgeloest.
_RAND_RE = re.compile(r"^rand(?P<p>\d+(?:\.\d+)?)$")


def subset_percent(name: str) -> float | None:
    """Anteil P aus "rand<P>", sonst None."""
    m = _RAND_RE.match(name)
    return float(m.group("p")) if m else None


def jump_strategy(name: str) -> JumpStrategy:
    p = subset_percent(name)
    if p is not None:
        return RandomSubsetJump(p)
    return JUMPS[name]()
