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
    # Kann der Zufallssprung diesen Knoten ueberhaupt treffen? Gelesen
    # ausschliesslich von weighting.DurwJumpSetWeighting.
    #
    # Default True, und das ist der Normalfall: wer gleichverteilt aus V zieht,
    # erreicht jeden Knoten. Nur ein Sprung aus einer externen Namensliste
    # (oracles.name_list) deckt V nicht ab -- dort trennt dieses Feld die
    # erreichbare Teilmenge S vom Rest, weil beide verschiedene
    # Stationaerwahrscheinlichkeiten haben.
    in_jump_set: bool = True
    # Wurde dieser Knoten per Zufallssprung erreicht (statt durch einen
    # normalen Schritt)? Nur DurwSampler setzt das -- alle anderen Sampler
    # kennen keinen Sprung und bleiben beim Default False. Gelesen von
    # diagnose_walk.diagnose_durw() zur Sprungrate; ansonsten ungenutzt.
    jumped: bool = False
    # Gewicht der Kante zum virtuellen Sprungknoten sigma, in Einheiten von w.
    # Gelesen ausschliesslich von weighting.DurwSigmaWeighting. Default 1.0 ist
    # das Original-DURW: sigma ist mit jedem Knoten mit Gewicht w verbunden.
    # Beim Sprung auf S u H (sampling.durw, history_jumps=True) ist es
    # m(u) + beta -- Vielfachheit in der Liste plus Historienanteil --, beim
    # Erstbesuch eingefroren wie deg_Gu.
    sigma_weight: float = 1.0


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
