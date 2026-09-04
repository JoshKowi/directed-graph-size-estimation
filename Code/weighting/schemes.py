"""Konkrete Weighting-Schemata.

Schnittstelle:
    class UniformWeighting(WeightingScheme)        -- w_i = 1
    class InverseDegreeWeighting(WeightingScheme)  -- w_i = 1/deg(u_i),
                                                      needs_degree = True
    class DurwWeighting(WeightingScheme)           -- w_i = 1/(w + deg_Gu(u_i)),
                                                      needs_degree = True
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import config
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


class DurwWeighting(WeightingScheme):
    """Fuer DURW-Stichproben mit pi(v) ~ w + deg_Gu(v) (sampling.durw).

    `Sample.degree` traegt bei DURW den Grad im aufgebauten ungerichteten
    Graphen G_u, nicht den Ausgangsgrad. Zusammen mit dem Sprunggewicht w ist
    die Stationaerverteilung geschlossen bekannt:

        pi(v) = (w + deg_Gu(v)) / (vol(V) + w|V|)

    Das Gewicht ist der Kehrwert des Zaehlers; die unbekannte Normierung
    kuerzt der Kollisionsschaetzer heraus (estimators.formulas).

    Anders als bei InverseDegreeWeighting braucht es hier kein max(deg, 1):
    mit w > 0 ist der Nenner immer positiv. Das ist kein Detail -- der Fall
    deg = 0 ist bei DURW auf den gerichteten Views haeufig (ein Knoten, dessen
    Ausgangskanten alle auf bereits Besuchtes zeigen) und dort ein regulaerer,
    korrekt gewichteter Zustand, kein Notfall.
    """

    needs_degree = True

    def __init__(self, jump_weight: float = config.DURW_JUMP_WEIGHT) -> None:
        self.jump_weight = float(jump_weight)
        if self.jump_weight <= 0:
            raise ValueError(f"jump_weight muss > 0 sein, ist {self.jump_weight}")
        # w gehoert in den Namen: zwei Gewichtungen mit verschiedenem w sind
        # verschiedene Schaetzer, auch wenn sie auf derselben Trajektorie
        # laufen koennten.
        self.name = f"inv_deg_plus_w{self.jump_weight:g}"

    def weights(self, samples: Sequence[Sample]) -> np.ndarray:
        # Wie bei InverseDegreeWeighting laut melden statt still falsch rechnen.
        if any(s.degree is None for s in samples):
            raise ValueError(
                "DurwWeighting braucht Sample.degree (den Grad in G_u), die "
                "Stichprobe wurde aber ohne Gradabfrage gezogen. DurwSampler "
                "liefert ihn immer -- kommt die Stichprobe von einem anderen "
                "Sampler, passt diese Gewichtung nicht."
            )
        deg = np.array([s.degree for s in samples], dtype=float)
        return 1.0 / (deg + self.jump_weight)
