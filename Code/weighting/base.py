"""Weighting-Modul: korrigiert die Verzerrung der Stichprobe.

Ein Weighting liefert zu jedem Sample ein Gewicht w_i, das proportional zu
1/pi(u_i) ist (pi = Ziehwahrscheinlichkeit des Samplers). Die Schaetzformel
(estimators.formulas) arbeitet nur noch mit (Samples, Gewichte) und muss den
Sampler nicht kennen.

Schnittstelle:
    class WeightingScheme
        .weights(samples) -> np.ndarray[float]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from sampling.base import Sample


class WeightingScheme(ABC):
    name: str = "weighting"

    @abstractmethod
    def weights(self, samples: Sequence[Sample]) -> np.ndarray:
        """Gewicht je Sample, proportional zu 1/pi(u_i)."""
