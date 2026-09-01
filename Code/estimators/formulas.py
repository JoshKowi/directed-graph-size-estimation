"""Schaetzformeln: rechnen aus (Samples, Gewichten) eine Zahl aus.

Letzte Stufe der Pipeline -- ein eigener Begriff neben `Estimator`, weil hier
nicht der Graph befragt wird, sondern nur noch gerechnet.

Zur Quelle: Kurant/Butts/Markopoulou, "Graph Size Estimation" (arXiv 1210.0460)
definiert n_col in Eq.(4) ueber *ungeordnete* Paare i<j, setzt in Eq.(5) und
Eq.(6) aber k^2 statt C(k,2) in den Zaehler. Beide Gleichungen liegen dadurch
um 2k/(k-1) ~ 2 zu hoch (Eq.(6) erbt den Faktor von Eq.(5)). Eq.(5) ist hier auf C(k,2) korrigiert -- siehe den Kommentar an der Stelle.
Eq.(6) wird nicht als eigene Formel gefuehrt: nach derselben Korrektur ist sie
rechnerisch identisch mit Katzirs Form, von der Kurant sie uebernommen hat
([19] in Kurant = Katzir/Liberty/Somekh, WWW 2011). Der einzige Unterschied
zwischen beiden war genau dieser Faktor 2.

Alle liefern NaN, wenn keine Kollision beobachtet wurde -- aus einer
kollisionsfreien Stichprobe laesst sich |V| nicht schaetzen, und ein
Ersatzwert wuerde beim Mitteln ueber Sample-Sets alles dominieren.

Alle Formeln kennen den Safety Margin (`margin`, Default 0 = aus): mit m > 0
zaehlen nur Paare mit Abstand > m im Walk als Kollision, und die Normierung
sinkt entsprechend. Siehe _collisions() und _pair_count().

Schnittstelle:
    class EstimationFormula
        .name, .margin, .compute(samples, weights) -> float
    class CollisionCountEstimator(EstimationFormula)       -- k^2 / n_col
    class WISCollisionEstimatorKatzir(EstimationFormula)   -- gradkorrigiert (Katzir)
    FORMULAS: dict[str, type[EstimationFormula]]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from sampling.base import Sample


def _collisions(samples: Sequence[Sample], margin: int = 0) -> float:
    """Anzahl kollidierender Paare (i<j mit u_i == u_j).
    Entspricht Kurants Definition [4] (Kollisionen zählen mehrfach im selben Knoten).

    `margin` (Safety Margin) laesst Paare aus, die im Walk weniger als m+1
    Schritte auseinanderliegen: u_i und u_{i+1} sind Nachbarn, u_i und u_{i+2}
    oft derselbe Knoten (hin und zurueck). Solche Treffer sagen nichts ueber
    |V|, sondern nur, dass der Walk noch nicht gemischt hat -- gezaehlt
    verkleinern sie die Schaetzung systematisch.

    Gezaehlt wird als Differenz, nicht ueber Paare: bei k = 8,5 Mio. Samples
    (so lang wurden die Traces auf gpt-4-io) gibt es 3,6e13 Paare, die sich
    nicht aufzaehlen lassen.

        n_col_m = alle Kollisionen - die mit Abstand 1..m

    Die nahen Kollisionen kosten m verschobene Array-Vergleiche, also O(k*m)
    in reinem numpy -- bei k = 8,5 Mio. und m = 10 rund 85 Mio. Vergleiche und
    damit weniger als das ohnehin noetige Sortieren in np.unique.
    """
    k = len(samples)
    if k < 2:
        return 0.0
    # np.fromiter fuellt direkt ein int64-Array; ohne das entstuende erst eine
    # Python-Liste und np.unique muesste ein object-Array sortieren (bei
    # 700k Samples Faktor 12 langsamer).
    nodes = np.fromiter((s.node for s in samples), dtype=np.int64, count=k)
    _, counts = np.unique(nodes, return_counts=True)
    total = float(np.sum(counts * (counts - 1) / 2))
    if margin <= 0:
        return total
    near = 0
    for d in range(1, min(margin, k - 1) + 1):
        near += int(np.count_nonzero(nodes[d:] == nodes[:-d]))
    return total - near


def _pair_count(k: int, margin: int = 0) -> float:
    """Zahl der *betrachteten* Paare -- die Normierung der Schaetzformel.

    Ohne Margin sind es C(k,2). Mit Margin fallen alle Indexpaare mit Abstand
    1..m weg, davon gibt es sum_{d=1..m} (k-d) = m*k - m(m+1)/2. Diese
    Korrektur gehoert zwingend zur Auslassung: bliebe C(k,2) im Zaehler,
    waere die Schaetzung um genau den ausgelassenen Anteil zu hoch.

    Bei k = 100 000 und m = 10 fallen ~1e6 von 5e9 Paaren weg -- 0,02 %.
    Zum Vergleich wirft `SimpleThinning(step=5)` 80 % der Samples weg.
    """
    if k < 2:
        return 0.0
    m = min(max(int(margin), 0), k - 1)
    return k * (k - 1) / 2 - (m * k - m * (m + 1) / 2)


class EstimationFormula(ABC):
    name: str = "formula"
    # Braucht die Formel echte Gewichte (WIS) oder rechnet sie mit w_i == 1?
    # Steht hier statt als Namensvergleich in den build()-Funktionen.
    weighted: bool = False

    def __init__(self, margin: int = 0) -> None:
        # Safety Margin, s. _collisions(). 0 = aus, dann rechnet die Formel
        # bitgleich wie vorher.
        self.margin = int(margin)

    @abstractmethod
    def compute(self, samples: Sequence[Sample], weights: np.ndarray) -> float:
        """Schaetzwert fuer |V| aus gewichteten Samples."""


class CollisionCountEstimator(EstimationFormula):
    """Collision Counting: n_hat = C(k,2) / n_col.

    Kurant Eq.(5) -- dort als k^2/n_col geschrieben, hier korrigiert (s.u.).

    k ist die Anzahl Samples, n_col die Anzahl kollidierender Paare. Gewichte
    werden nicht benutzt -- der Schaetzer unterstellt gleichverteilte,
    unabhaengige Ziehungen. Es ist der klassische Birthday-Schaetzer und der
    Spezialfall w_i == 1 von WISCollisionEstimatorKatzir.
    """

    name = "uis-collision"

    def compute(self, samples: Sequence[Sample], weights: np.ndarray) -> float:
        k = len(samples)
        # KORREKTUR gegenueber Kurant Eq.(5), die k^2/n_col schreibt.
        # Eq.(4) zaehlt ungeordnete Paare i<j, also E[n_col] = C(k,2)/N.
        # Damit ist k^2/n_col um 2k/(k-1) ~ 2 zu hoch; hier steht C(k,2)
        # -- bzw. mit Safety Margin die Zahl der betrachteten Paare.
        # Nachgerechnet (UIS, N=2000): k^2/n_col -> 4002, C(k,2)/n_col -> 2000.
        pairs = _pair_count(k, self.margin)
        if pairs <= 0:
            return float("nan")
        collisions = _collisions(samples, self.margin)
        if collisions == 0:
            return float("nan")
        return pairs / collisions


class WISCollisionEstimatorKatzir(EstimationFormula):
    """Gradkorrigierter Collision-Schaetzer nach Katzir et al. (2011).

    Identisch zu Kurant Eq.(6), sobald deren k^2 durch C(k,2) ersetzt ist.

    Mit n_col kollidierenden Paaren und Gewichten w_i ~ 1/pi(u_i) gilt
        n_hat = C(k,2) * mean(w) * mean(1/w) / n_col.
    Fuer w_i == 1 reduziert sich das auf den klassischen Birthday-Schaetzer
    C(k,2)/n_col.

    C(k,2) passt zur ungeordneten Zaehlweise in _collisions() -- der Schaetzer
    schaetzt die Kollisions*wahrscheinlichkeit* p = n_col/C(k,2) und setzt
    n_hat = mean(w)*mean(1/w)/p. Diese Form ist unveraendert; korrigiert wurden
    die beiden anderen. Nachgerechnet (WIS, pi ~ deg, N=2000): 1999.9.
    """

    name = "wis-col-katzir"
    weighted = True

    def compute(self, samples: Sequence[Sample], weights: np.ndarray) -> float:
        k = len(samples)
        pairs = _pair_count(k, self.margin)
        if pairs <= 0:
            return float("nan")
        collisions = _collisions(samples, self.margin)
        if collisions == 0:
            return float("nan")

        w = np.asarray(weights, dtype=float)
        correction = w.mean() * (1.0 / w).mean()
        return pairs * correction / collisions


FORMULAS: dict[str, type[EstimationFormula]] = {
    "uis-collision": CollisionCountEstimator,
    "wis-col-katzir": WISCollisionEstimatorKatzir,
}
