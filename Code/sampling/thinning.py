"""Dependency Reduction: aus der Trajektorie eines Random Walks Sample-Sets
gewinnen.

Thinning ist *kein* Sampling -- der Walk ist gelaufen, die Queries sind bezahlt.
Es ist reine Nachbearbeitung der Trajektorie und deshalb eine eigene Stufe
zwischen Sampler und Weighting:

    Oracle -> Walk -> Trace -> Thinning -> (1..n Sets) -> Weighting/Formula -> Aggregation

Aufeinanderfolgende Schritte eines Walks sind stark korreliert; der
Collision-Schaetzer setzt aber unabhaengige Ziehungen voraus. Thinning
vergroessert den Abstand zwischen den benutzten Samples.

Schnittstelle:
    class Thinning
        .name, .apply(trace) -> list[list[Sample]]
    NoThinning, SimpleThinning, ShiftedThinning, ByWalkThinning
    THINNINGS: dict[str, type[Thinning]]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from sampling.base import Sample


class Thinning(ABC):
    name: str = "thinning"

    @abstractmethod
    def apply(self, trace: Sequence[Sample]) -> list[list[Sample]]:
        """Ein oder mehrere Sample-Sets aus der Trajektorie."""


class NoThinning(Thinning):
    """Ganze Trajektorie als ein Set."""

    name = "none"

    def apply(self, trace: Sequence[Sample]) -> list[list[Sample]]:
        return [list(trace)]


class SimpleThinning(Thinning):
    """Nur jedes n-te Sample, ab Offset 0. Verwirft (n-1)/n des Budgets."""

    name = "simple"

    def __init__(self, step: int = 5) -> None:
        self.step = step

    def apply(self, trace: Sequence[Sample]) -> list[list[Sample]]:
        return [list(trace[:: self.step])]


class ShiftedThinning(Thinning):
    """n verschobene Sets: Offset 0, 1, ..., n-1, jeweils jedes n-te Sample.

    Nutzt die volle Trajektorie. Die Sets stammen aus demselben Walk und sind
    daher korreliert -- die Mittelung ueber ihre Schaetzungen reduziert die
    Varianz weniger als n unabhaengige Laeufe (eher Batch-Means als echte
    Replikation).
    """

    name = "shifted"

    def __init__(self, step: int = 5) -> None:
        self.step = step

    def apply(self, trace: Sequence[Sample]) -> list[list[Sample]]:
        return [list(trace[offset :: self.step]) for offset in range(self.step)]


class ByWalkThinning(Thinning):
    """Ein Set je Durchgang des Samplers (`Sample.walk`).

    Die Aufteilung, die Capture-Recapture braucht: zwei Faenge, die nicht einer
    die Fortsetzung des anderen sind (siehe RandomWalkSampler mit n_walks > 1).

    Die Zahl der Sets steht fest und wird nicht aus dem Trace abgeleitet: kam
    der zweite Fang gar nicht mehr zum Zug, muss trotzdem ein -- dann leeres --
    Set entstehen, sonst rechnete die Formel still mit einem Fang weiter.

    Bewusst *nicht* in THINNINGS: dieses dict erzeugt in estimators/__init__.py
    das Kreuzprodukt rw_plain__<dead_end>__<thinning>, und mit dem
    ueblichen Ein-Walk-Sampler waere das hier nur ein umstaendliches "none".
    """

    name = "by_walk"

    def __init__(self, n_walks: int = 2) -> None:
        self.n_walks = n_walks

    def apply(self, trace: Sequence[Sample]) -> list[list[Sample]]:
        sets: list[list[Sample]] = [[] for _ in range(self.n_walks)]
        for s in trace:
            if s.walk < self.n_walks:
                sets[s.walk].append(s)
        return sets


THINNINGS: dict[str, type[Thinning]] = {
    "none": NoThinning,
    "simple": SimpleThinning,
    "shifted": ShiftedThinning,
}
