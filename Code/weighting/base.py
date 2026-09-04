"""Weighting-Modul: korrigiert die Verzerrung der Stichprobe.

Ein Weighting liefert zu jedem Sample ein Gewicht w_i, das proportional zu
1/pi(u_i) ist (pi = Ziehwahrscheinlichkeit des Samplers). Die Schaetzformel
(estimators.formulas) arbeitet nur noch mit (Samples, Gewichte) und muss den
Sampler nicht kennen.

Schnittstelle:
    class WeightingScheme
        .name, .needs_degree
        .weights(samples) -> np.ndarray[float]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from sampling.base import Sample


class WeightingScheme(ABC):
    name: str = "weighting"

    # Braucht dieses Schema `Sample.degree`? Steuert, ob der Sampler den Grad
    # ueberhaupt abfragt -- eine Gradabfrage kostet Budget (oracles.base
    # ._fetch), und wofuer nicht gerechnet wird, soll auch nicht bezahlt
    # werden. Nur InverseDegreeWeighting liest den Grad; die gewichteten
    # Formeln bekommen ihn ausschliesslich ueber das `weights`-Array.
    needs_degree: bool = False

    @abstractmethod
    def weights(self, samples: Sequence[Sample]) -> np.ndarray:
        """Gewicht je Sample, proportional zu 1/pi(u_i)."""
