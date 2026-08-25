"""Konkrete Weighting-Schemata.

Schnittstelle:
    class UniformWeighting(WeightingScheme)        -- w_i = 1
    class InverseDegreeWeighting(WeightingScheme)  -- w_i = 1/deg(u_i)
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

    def weights(self, samples: Sequence[Sample]) -> np.ndarray:
        deg = np.array([max(s.degree, 1) for s in samples], dtype=float)
        return 1.0 / deg
