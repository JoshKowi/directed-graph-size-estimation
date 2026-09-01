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

Zwei Arten von Formeln:

    EstimationFormula  rechnet je Sample-Set eine Zahl; die Pipeline ruft sie
                       fuer jedes Set und aggregiert (Default: Median).
    SetsFormula        rechnet *einmal* ueber alle Sets gemeinsam. Das braucht
                       Capture-Recapture: |S1|*|S2|/|S1 geschnitten S2| laesst
                       sich nicht je Set und danach mitteln.

Schnittstelle:
    class EstimationFormula
        .name, .margin, .compute(samples, weights) -> float
        .extras(subsets, weights) -> dict
    class SetsFormula
        .name, .compute_sets(subsets, weights) -> float
        .extras(subsets, weights) -> dict
    LincolnPetersen, ChapmanEstimator, SchnabelEstimator
    CrossCollisionEstimator, WISCrossCollisionEstimator
    SETS_FORMULAS: dict[str, type[SetsFormula]]

Gewichte nehmen entgegen: WISCollisionEstimatorKatzir (je Set) und
WISCrossCollisionEstimator (ueber Sets). Lincoln-Petersen, Chapman und
Schnabel rechnen mit Mengen *verschiedener* Knoten -- dort gibt es keine
saubere Gewichtung, siehe CrossCollisionEstimator.
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

    def extras(self, subsets, weights) -> dict:
        """Verfahrensspezifische Zwischenwerte fuer die Ergebnis-CSV
        (Praefix `extra_`). Default: keine."""
        return {}


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


class SetsFormula(ABC):
    """Formel, die alle Sample-Sets *gemeinsam* auswertet."""

    name: str = "sets-formula"
    weighted: bool = False

    @abstractmethod
    def compute_sets(self, subsets, weights) -> float:
        """Schaetzwert fuer |V| aus mehreren Sample-Sets."""

    def extras(self, subsets, weights) -> dict:
        return {}


class LincolnPetersen(SetsFormula):
    """Capture-Recapture: n_hat = |S1| * |S2| / |S1 geschnitten S2|.

    |S1| und |S2| sind die Zahlen *verschiedener* besuchter Knoten je Fang.
    Ohne Ueberschneidung ist keine Schaetzung moeglich -> NaN.

    Der Schaetzer unterstellt, dass die beiden Faenge unabhaengig sind und
    jeder Knoten in beiden dieselbe Fangwahrscheinlichkeit hat. Ein Random Walk
    verletzt beides; genau deshalb steht das Verfahren hier im Vergleich.

    Zwei Sets sind Pflicht: mit einem laesst sich nichts schneiden, mit dreien
    waere Lincoln-Petersen die falsche Formel (dann Schnabel/Chapman). Lieber
    hier scheitern als still etwas anderes rechnen.
    """

    name = "lincoln-petersen"

    def _sets(self, subsets):
        return [{s.node for s in part} for part in subsets]

    def compute_sets(self, subsets, weights) -> float:
        if len(subsets) != 2:
            raise ValueError(
                f"Lincoln-Petersen braucht genau zwei Faenge, bekam "
                f"{len(subsets)}. Passt das Thinning zum Sampler "
                "(ByWalkThinning(n_walks) und RandomWalkSampler(n_walks))?"
            )
        s1, s2 = self._sets(subsets)
        overlap = len(s1 & s2)
        return float("nan") if overlap == 0 else len(s1) * len(s2) / overlap

    def extras(self, subsets, weights) -> dict:
        s1, s2 = self._sets(subsets)
        return {"n_unique_s1": len(s1), "n_unique_s2": len(s2),
                "overlap": len(s1 & s2)}


class ChapmanEstimator(SetsFormula):
    """Capture-Recapture nach Chapman (1951) -- verzerrungskorrigiertes
    Lincoln-Petersen.

        n_hat = (n1+1)(n2+1)/(m+1) - 1

    Zwei Vorteile gegenueber n1*n2/m: der Schaetzer ist bei kleinen
    Stichproben praktisch erwartungstreu (exakt, wenn n1+n2 >= N, sonst mit
    kleinem Rest), und er ist **immer definiert**. Ohne Ueberschneidung liefert
    Lincoln-Petersen NaN -- was bei kleinen Budgets regelmaessig vorkommt --,
    Chapman dagegen (n1+1)(n2+1)-1, also eine grosse, aber endliche Schaetzung.

    Achtung: korrigiert wird die Kleinstichproben-Verzerrung, *nicht* die
    ungleiche Fangwahrscheinlichkeit. Ein Random Walk faengt Knoten
    proportional zum Grad; die Hubs landen in beiden Faengen, die
    Ueberschneidung ist dadurch zu gross und n_hat zu klein. Daran aendert
    Chapman nichts.
    """

    name = "chapman"

    def compute_sets(self, subsets, weights) -> float:
        if len(subsets) != 2:
            raise ValueError(
                f"Chapman braucht genau zwei Faenge, bekam {len(subsets)}. "
                "Fuer mehr Faenge ist Schnabel zustaendig."
            )
        s1, s2 = ({s.node for s in part} for part in subsets)
        m = len(s1 & s2)
        return (len(s1) + 1) * (len(s2) + 1) / (m + 1) - 1

    def extras(self, subsets, weights) -> dict:
        s1, s2 = ({s.node for s in part} for part in subsets)
        return {"n_unique_s1": len(s1), "n_unique_s2": len(s2),
                "overlap": len(s1 & s2)}


class SchnabelEstimator(SetsFormula):
    """Capture-Recapture ueber k Faenge nach Schnabel (1938).

        n_hat = sum_t (C_t * M_t) / sum_t R_t

    Je Fang t: C_t die Zahl verschiedener gefangener Knoten, M_t die Zahl der
    vor diesem Fang bereits markierten, R_t die davon wiedergefangenen. Im Kern
    ein gewichtetes Mittel der aufeinanderfolgenden Lincoln-Petersen-
    Schaetzungen; fuer k = 2 faellt es exakt auf Lincoln-Petersen zurueck
    (C_2 * M_2 / R_2 = n2 * n1 / m).

    Der erste Fang traegt nichts bei (M_1 = 0) und laeuft trotzdem mit durch die
    Summe -- das spart eine Fallunterscheidung und aendert nichts.

    Ohne jeden Wiederfang (sum R_t = 0) ist keine Schaetzung moeglich -> NaN.
    Fuer diesen Fall gibt es Chapman, allerdings nur fuer zwei Faenge.
    """

    name = "schnabel"

    def compute_sets(self, subsets, weights) -> float:
        if len(subsets) < 2:
            raise ValueError(
                f"Schnabel braucht mindestens zwei Faenge, bekam {len(subsets)}"
            )
        marked: set = set()
        num = rec = 0.0
        for part in subsets:
            caught = {s.node for s in part}
            num += len(caught) * len(marked)
            rec += len(caught & marked)
            marked |= caught
        return float("nan") if rec == 0 else num / rec

    def extras(self, subsets, weights) -> dict:
        marked: set = set()
        rec = 0
        sizes = []
        for part in subsets:
            caught = {s.node for s in part}
            sizes.append(len(caught))
            rec += len(caught & marked)
            marked |= caught
        return {"n_captures": len(subsets), "recaptures": rec,
                "n_unique_total": len(marked), "n_unique_s1": sizes[0],
                "n_unique_s2": sizes[1] if len(sizes) > 1 else 0}


class CrossCollisionEstimator(SetsFormula):
    """Kollisionen *zwischen* den Faengen -- die gewichtbare Form von
    Capture-Recapture.

        n_hat = P_cross / n_cross,  P_cross = sum_{t<t\'} k_t * k_t\'

    n_cross zaehlt Paare (i, j) aus *verschiedenen* Faengen mit u_i == u_j,
    Mehrfachbesuche eingeschlossen; P_cross ist die Zahl solcher Paare
    ueberhaupt.

    **Warum nicht einfach Lincoln-Petersen gewichten?** LP rechnet mit den
    Mengen *verschiedener* Knoten: n1*n2/m. Ein Knoten zaehlt dort einmal, egal
    wie oft er gefangen wurde. Fuer eine Gewichtung nach 1/pi braeuchte man die
    Einschlusswahrscheinlichkeit 1-(1-pi)^k -- nichtlinear und von der
    Fanggroesse abhaengig. Zaehlt man stattdessen *Paare mit Vielfachheit*,
    steht wieder Katzirs Identitaet zur Verfuegung, und die Gewichtung ist
    exakt dieselbe wie beim Collision Counting.

    Herleitung wie bei WISCollisionEstimatorKatzir: fuer i, j aus
    verschiedenen (unabhaengigen) Faengen ist P(u_i == u_j) = sum_v pi_v^2,
    also E[n_cross] = P_cross * sum pi^2. Mit w ~ 1/pi gilt E[w] = N und
    E[1/w] = sum pi^2, der Korrekturfaktor mean(w)*mean(1/w) hebt sum pi^2
    also gerade weg.

    Fuer w == 1 bleibt P_cross/n_cross -- die Vielfachheits-Variante von
    Lincoln-Petersen. Sie ist mit LP *nicht* identisch: LP zaehlt Knoten, das
    hier zaehlt Paare. Bei k Faengen (k > 2) verallgemeinert es sich von
    selbst, ohne Schnabels Summenformel.
    """

    name = "cross-collision"
    weighted = False

    def _counts(self, subsets):
        """(n_cross, P_cross, alle Knoten) -- ohne Paare aufzuzaehlen.

        Kollisionen zwischen den Faengen = alle Kollisionen minus die
        innerhalb der Faenge. Beides ueber np.unique, also O(K log K); die
        Zahl der Paare selbst waere bei grossen Stichproben nicht darstellbar.
        """
        nodes = [np.fromiter((s.node for s in part), dtype=np.int64,
                             count=len(part)) for part in subsets]
        sizes = np.array([len(x) for x in nodes], dtype=np.float64)

        def collisions(arr):
            if arr.size < 2:
                return 0.0
            _, counts = np.unique(arr, return_counts=True)
            return float(np.sum(counts * (counts - 1) / 2))

        total = collisions(np.concatenate(nodes)) if nodes else 0.0
        within = sum(collisions(x) for x in nodes)
        n_cross = total - within
        p_cross = float((sizes.sum() ** 2 - np.sum(sizes ** 2)) / 2)
        return n_cross, p_cross, nodes

    def compute_sets(self, subsets, weights) -> float:
        if len(subsets) < 2:
            raise ValueError(
                f"Kollisionen zwischen Faengen brauchen mindestens zwei, "
                f"bekam {len(subsets)}"
            )
        n_cross, p_cross, _ = self._counts(subsets)
        if n_cross == 0 or p_cross <= 0:
            return float("nan")
        value = p_cross / n_cross
        if self.weighted:
            w = np.concatenate([np.asarray(x, dtype=float) for x in weights])
            value *= w.mean() * (1.0 / w).mean()
        return value

    def extras(self, subsets, weights) -> dict:
        n_cross, p_cross, nodes = self._counts(subsets)
        marked: set = set()
        for x in nodes:
            marked |= set(x.tolist())
        return {"n_captures": len(subsets), "cross_collisions": n_cross,
                "cross_pairs": p_cross, "n_unique_total": len(marked)}


class WISCrossCollisionEstimator(CrossCollisionEstimator):
    """Wie CrossCollisionEstimator, aber mit Gradkorrektur (Katzir).

    Auf `undirected` ist das die richtige Rechnung fuer einen Random Walk:
    er faengt Knoten proportional zum Grad, w = 1/deg korrigiert das. Auf
    `directed` gilt pi ~ deg_out nicht -- dort zeigt der Vergleich mit der
    ungewichteten Variante, was die Korrektur anrichtet.
    """

    name = "wis-cross-collision"
    weighted = True


# Formeln, die alle Sample-Sets gemeinsam auswerten (Capture-Recapture).
# Getrennt von FORMULAS, weil sie eine andere Signatur haben und ein Thinning
# brauchen, das mehrere Sets liefert -- siehe sampling.thinning.ByWalkThinning.
SETS_FORMULAS: dict[str, type[SetsFormula]] = {
    "lincoln-petersen": LincolnPetersen,
    "chapman": ChapmanEstimator,
    "schnabel": SchnabelEstimator,
    "cross": CrossCollisionEstimator,
    "cross-wis": WISCrossCollisionEstimator,
}


# Nur die Formeln, die je Set rechnen: die generischen build()-Funktionen
# konstruieren daraus mit FORMULAS[name](margin=...). LincolnPetersen steht
# bewusst nicht drin -- es braucht mehrere Sets und wird direkt von
# estimators/methods/capture_recapture.py gebaut.
FORMULAS: dict[str, type[EstimationFormula]] = {
    "uis-collision": CollisionCountEstimator,
    "wis-col-katzir": WISCollisionEstimatorKatzir,
}
