"""Sampler: *wie* ein Oracle genutzt wird, um eine Stichprobe zu ziehen.

Bewusst getrennt vom Oracle (was darf abgefragt werden) und vom Weighting
(wie wird die Verzerrung korrigiert): Random Walk und unabhaengiges Ziehen
nutzen dasselbe Oracle, erzeugen aber unterschiedlich verzerrte Stichproben.

Schnittstelle:
    class Sample            -- node (Original-Key), degree, step
    class Sampler
        .sample(oracle) -> list[Sample]   (laeuft, bis das Budget erschoepft ist)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Sample:
    node: Any  # Original-Knotenname aus der Adjazenzliste
    degree: int
    step: int


class Sampler(ABC):
    name: str = "sampler"

    @abstractmethod
    def sample(self, oracle) -> list[Sample]:
        """Zieht Samples, bis BudgetExceeded auftritt; gibt das Bisherige zurueck."""
