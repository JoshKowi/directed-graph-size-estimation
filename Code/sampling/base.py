"""Sampler: *wie* ein Oracle genutzt wird, um eine Stichprobe zu ziehen.

Bewusst getrennt vom Oracle (was darf abgefragt werden) und vom Weighting
(wie wird die Verzerrung korrigiert): Random Walk und unabhaengiges Ziehen
nutzen dasselbe Oracle, erzeugen aber unterschiedlich verzerrte Stichproben.

`key()` sagt, wann zwei Sampler dieselbe Trajektorie erzeugen wuerden. Der
Runner gruppiert danach (`--share-walks`): Thinning, Weighting und Formel
kommen erst *nach* dem Sampler und aendern nichts am Walk, also duerfen sich
alle Estimators mit gleichem Oracle und gleichem Sampler-Schluessel einen
einzigen Walk teilen.

Schnittstelle:
    class Sample            -- node (Original-Key), degree (ggf. None), step, walk
    class Sampler
        .sample(oracle) -> list[Sample]   (laeuft, bis das Budget erschoepft ist)
        .key() -> str                     (gleiche Ziehung == gleicher Schluessel)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Sample:
    node: Any  # Original-Knotenname aus der Adjazenzliste
    # `None`, wenn der Sampler den Grad gar nicht erst abgefragt hat -- er
    # kostet dann auch nichts (siehe UniformSampler(with_degree=False)).
    # Gelesen wird er ausschliesslich von weighting.InverseDegreeWeighting.
    degree: int | None
    step: int
    # Aus welchem Durchgang das Sample stammt. Nur Sampler, die mehrere Walks
    # laufen (Capture-Recapture), setzen das -- alle anderen bleiben bei 0.
    walk: int = 0


class Sampler(ABC):
    name: str = "sampler"

    @abstractmethod
    def sample(self, oracle) -> list[Sample]:
        """Zieht Samples, bis BudgetExceeded auftritt; gibt das Bisherige zurueck."""

    def key(self) -> str:
        """Identitaet der Ziehung -- alles, was die Trajektorie beeinflusst.

        Default ist der Name; Sampler mit Parametern muessen diese ergaenzen,
        sonst wuerden zwei verschieden parametrierte Sampler faelschlich einen
        Walk teilen.
        """
        return self.name
