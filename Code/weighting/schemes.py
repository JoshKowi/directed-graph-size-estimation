"""Konkrete Weighting-Schemata.

Schnittstelle:
    class UniformWeighting(WeightingScheme)        -- w_i = 1
    class InverseDegreeWeighting(WeightingScheme)  -- w_i = 1/deg(u_i),
                                                      needs_degree = True
    class DurwWeighting(WeightingScheme)           -- w_i = 1/(w + deg_Gu(u_i)),
                                                      needs_degree = True
    class InDegreeWeighting(WeightingScheme)       -- w_i = 1/d-(u_i),
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


class InDegreeWeighting(WeightingScheme):
    """Fuer NMMC-Stichproben mit dem Ziel pi(v) ~ d-(v) (sampling.nmmc).

    Bei diesem Ziel kuerzt sich d-(j) aus der Annahmewahrscheinlichkeit heraus,
    b_ij = d+(i)/d-(i) haengt nur noch an i -- der Grund, warum das Paper es
    als das deutlich schneller konvergierende Ziel misst. Die Stichprobe ist
    dafuer proportional zum Eingangsgrad verzerrt, und genau das korrigiert
    dieses Gewicht. Die Normierung kuerzt der Kollisionsschaetzer heraus
    (estimators.formulas).

    `Sample.degree` traegt bei NMMC den -- je nach sampling.indegree
    geschaetzten oder exakten -- **Eingangs**grad, nicht den Ausgangsgrad.
    Dieselbe Abkuerzung nutzt DURW fuer den G_u-Grad; der frozen dataclass
    Sample einen eigenen Slot dafuer zu geben, waere fuer alle anderen Sampler
    toter Ballast.

    Rechnerisch ist das Gewicht identisch mit InverseDegreeWeighting -- der
    Unterschied steckt allein darin, *welche* Groesse im Sample liegt.
    Absichern laesst sich das nicht durch Typen, sondern nur ueber den
    Konstruktionsweg: die Zuordnung Sampler <-> Gewichtung setzt
    estimators/methods/nmmc.py, so wie es DURW mit DurwWeighting haelt. Einen
    eigenen Namen bekommt es trotzdem, denn was verschiedene Groessen
    korrigiert, soll in der Ergebnis-CSV nicht gleich heissen.
    """

    name = "inv_in_degree"
    needs_degree = True

    def weights(self, samples: Sequence[Sample]) -> np.ndarray:
        # Wie bei den anderen Schemata laut melden statt still falsch rechnen.
        if any(s.degree is None for s in samples):
            raise ValueError(
                "InDegreeWeighting braucht Sample.degree (den Eingangsgrad), "
                "die Stichprobe wurde aber ohne gezogen. NmmcSampler liefert "
                "ihn immer -- kommt die Stichprobe von einem anderen Sampler, "
                "passt diese Gewichtung nicht."
            )
        # sampling.indegree gibt nie 0 zurueck; das max ist die zweite Schranke
        # fuer den Fall, dass die Stichprobe doch woanders herkommt.
        deg = np.array([max(s.degree, 1) for s in samples], dtype=float)
        return 1.0 / deg


class DurwJumpSetWeighting(WeightingScheme):
    """Fuer DURW mit einem Sprung, der V *nicht* abdeckt (oracles.name_list).

    **Abweichung vom Original.** Bei Ribeiro & Towsley ist der virtuelle Knoten
    sigma mit ganz V verbunden: der Sprung ist gleichverteilt ueber die
    Knotenmenge, und daraus folgt pi(v) ~ w + deg_Gu(v) -- die Annahme, unter
    der DurwWeighting korrekt ist. Zieht der Sprung dagegen aus einer externen
    Namensliste, ist sigma nur mit der Trefferteilmenge S verbunden. Das ist
    eine andere Kette mit einer anderen Stationaerverteilung:

        pi(v) ~ deg_Gu(v) + w * 1[v in S]

    Diese Klasse ist deren Kehrwert. Sie macht die Abweichung *nicht*
    rueckgaengig -- der Sprung erreicht weiterhin nur 6 bis 22 % der Knoten --,
    sondern bringt nur die Gewichtung wieder mit der tatsaechlichen Kette in
    Einklang. Mit S = V faellt sie exakt auf DurwWeighting zurueck, weil
    Sample.in_jump_set dann ueberall True ist.

    Warum das noetig ist: mit DurwWeighting bekommen Knoten *ausserhalb* von S
    einen um w zu grossen Nenner, sind also systematisch zu niedrig gewichtet
    -- und das betrifft keine Randgruppe. Gemessen auf gpt4o_io gerichtet
    liegen nur 58,0 % (top-q) bzw. 68,9 % (indeg) der Samples in S; der Rest
    wurde ueber Kanten erreicht. Die Folge war ein Schaetzer ohne Fixpunkt bei
    |V|: durw-indeg-gpt4_io__n1000000__b0__margin stieg mit wachsendem Budget
    monoton von 0,115 auf 1,850, statt zu konvergieren.

    Gilt nur fuer draw_burn_in = 0. Laeuft nach dem Treffer noch ein Burn-in,
    liefert der Sprung einen Knoten, der gar nicht mehr in S liegt -- die
    Sprungverteilung ist dann wieder unbekannt und diese Gewichtung ebenso
    falsch wie DurwWeighting. estimators/methods/durw.py weist die Kombination
    deshalb ab.
    """

    needs_degree = True

    def __init__(self, jump_weight: float = config.DURW_JUMP_WEIGHT) -> None:
        self.jump_weight = float(jump_weight)
        if self.jump_weight <= 0:
            raise ValueError(f"jump_weight muss > 0 sein, ist {self.jump_weight}")
        self.name = f"inv_deg_plus_w{self.jump_weight:g}_inS"

    def weights(self, samples: Sequence[Sample]) -> np.ndarray:
        if any(s.degree is None for s in samples):
            raise ValueError(
                "DurwJumpSetWeighting braucht Sample.degree (den Grad in G_u), "
                "die Stichprobe wurde aber ohne Gradabfrage gezogen."
            )
        deg = np.array([s.degree for s in samples], dtype=float)
        in_set = np.array([s.in_jump_set for s in samples], dtype=float)
        denom = deg + self.jump_weight * in_set
        if not np.all(denom > 0):
            # deg_Gu = 0 *und* ausserhalb von S hiesse: weder per Sprung noch
            # ueber eine Kante erreichbar -- der Knoten kann gar nicht im Trace
            # stehen. Tritt es doch auf, stimmt etwas anderes nicht (etwa ein
            # Burn-in, der Knoten ausserhalb von S liefert).
            raise ValueError(
                f"{int((denom <= 0).sum())} Samples haben deg_Gu = 0 und liegen "
                "ausserhalb der Sprungmenge -- so ein Knoten waere unerreichbar. "
                "Laeuft hier ein draw_burn_in > 0? Dann passt diese Gewichtung "
                "nicht (s. Docstring)."
            )
        return 1.0 / denom


class DurwSigmaWeighting(WeightingScheme):
    """Fuer DURW mit Sprung auf S u H (sampling.durw, history_jumps=True).

    **Abweichung vom Original.** Bei Ribeiro & Towsley ist der virtuelle Knoten
    sigma mit ganz V verbunden, jeder Knoten mit Gewicht w. Hier ist sigma mit
    der Vereinigung aus Sprungliste S und Historie H verbunden, mit Gewicht
    w * c(u), c(u) = m(u) + beta * 1[u in H]:

        m(u)   wie oft u in der Liste steht (Vielfachheit, 0 ausserhalb S)
        beta   Gewicht der Historie (history_weight)

    Die Kette bleibt damit ein Random Walk auf einem *ungerichteten*
    gewichteten Graphen -- jeder Knoten, auf dem sigma landen kann, kann auch
    springen, und umgekehrt. Nur dann ist sie reversibel, und nur dann gilt

        pi(u) ~ deg_Gu(u) + w * c(u).

    Diese Klasse ist deren Kehrwert: w_i = 1 / (deg_Gu + w * sigma_weight),
    mit sigma_weight = c(u), vom Sampler beim Erstbesuch eingefroren wie deg_Gu.

    Warum nicht DurwJumpSetWeighting (durwset-*): dort springt *jeder* Knoten,
    sigma landet aber nur auf S. Ein Knoten ausserhalb S hat damit eine Kante
    zu sigma, aber keine von sigma -- eine Einbahnstrasse. Die Kette ist nicht
    reversibel und hat *keine* geschlossene Stationaerverteilung; exakt
    nachgerechnet (Eigenvektor, 300 Knoten) weicht pi um 29 % von
    deg_Gu + w*1[S] ab. durwset-* verschiebt die Schaetzung deshalb nur um
    einen Faktor, statt sie zu korrigieren.

    Warum nicht "nur S springt": dann kann der Walk ausserhalb von S nicht
    heraus und haengt lange in Regionen fest, die er nur zu Fuss wieder
    verlaesst. Mit H kann jeder besuchte Knoten springen.

    Die Vielfachheit m(u) gehoert in Sprungregel *und* Gewicht -- bei top-q
    steht ein Knoten bis zu 40-mal in der Liste. Ohne sie weicht pi im Test um
    5 % ab. Fehlschlaege beim Ziehen beruehren weder das eine noch das andere:
    neu ziehen bis zum Treffer liefert exakt m(u)/sum(m).

    Mit S = V und m = 1 sowie beta -> 0 faellt das aufs Original zurueck.
    """

    needs_degree = True

    def __init__(self, jump_weight: float = config.DURW_JUMP_WEIGHT) -> None:
        self.jump_weight = float(jump_weight)
        if self.jump_weight <= 0:
            raise ValueError(f"jump_weight muss > 0 sein, ist {self.jump_weight}")
        self.name = f"inv_deg_plus_w{self.jump_weight:g}_sigma"

    def weights(self, samples: Sequence[Sample]) -> np.ndarray:
        if any(s.degree is None for s in samples):
            raise ValueError(
                "DurwSigmaWeighting braucht Sample.degree (den Grad in G_u).")
        deg = np.array([s.degree for s in samples], dtype=float)
        sig = np.array([s.sigma_weight for s in samples], dtype=float)
        denom = deg + self.jump_weight * sig
        if not np.all(denom > 0):
            raise ValueError(
                f"{int((denom <= 0).sum())} Samples mit deg_Gu = 0 und "
                "sigma_weight = 0 -- ein solcher Knoten waere unerreichbar. Mit "
                "history_jumps ist sigma_weight >= beta > 0; laeuft der Sampler "
                "vielleicht nicht in diesem Modus?")
        return 1.0 / denom
