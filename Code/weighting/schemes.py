"""Konkrete Weighting-Schemata.

Schnittstelle:
    class UniformWeighting(WeightingScheme)        -- w_i = 1
    class InverseDegreeWeighting(WeightingScheme)  -- w_i = 1/deg(u_i),
                                                      needs_degree = True
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sampling.base import Sample
from weighting.base import WeightingScheme


class UniformWeighting(WeightingScheme):
    """Fuer unverzerrte (gleichverteilte) Stichproben."""

    name = "uniform"

    def weights(self, samples: Sequence[Sample]) -> np.ndarray:
        return np.ones(len(samples), dtype=float)


class InverseDegreeWeighting(WeightingScheme):
    """Fuer Random-Walk-Stichproben mit pi(u) ~ deg(u)."""

    name = "inv_degree"
    needs_degree = True

    def weights(self, samples: Sequence[Sample]) -> np.ndarray:
        # Ohne Gradabfrage steht hier None. Das laut zu melden ist wichtig:
        # max(None, 1) waere ein TypeError, aber ein 0 an dieser Stelle wuerde
        # stillschweigend zu Gewicht 1 und damit zu einer falschen Schaetzung
        # ohne jede Fehlermeldung.
        if any(s.degree is None for s in samples):
            raise ValueError(
                "InverseDegreeWeighting braucht Sample.degree, die Stichprobe "
                "wurde aber ohne Gradabfrage gezogen. Der Sampler muss mit "
                "with_degree=True laufen -- die build()-Funktionen leiten das "
                "aus weighting.needs_degree ab."
            )
        deg = np.array([max(s.degree, 1) for s in samples], dtype=float)
        return 1.0 / deg
