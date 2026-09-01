"""Capture-Recapture (Lincoln-Petersen) als Pipeline: zwei Faenge, eine Formel.

    n_hat = |S1| * |S2| / |S1 geschnitten S2|

mit |S1|, |S2| als Anzahl *verschiedener* besuchter Knoten je Fang.

Das Verfahren schreibt nicht nur die Formel vor, sondern auch die Form der
Ziehung: es braucht zwei Stichproben mit eigenem Einstieg. Ein Halbieren
*derselben* Trajektorie taugt nicht -- die zweite Haelfte liefe dort weiter,
wo die erste aufgehoert hat, und die Ueberschneidung waere strukturell zu
gross. Deshalb sitzt der Unterschied zu den anderen Verfahren im Sampler
(`n_walks=2`) und im Thinning (`ByWalkThinning`), nicht in einer eigenen
Estimator-Klasse.

Die beiden Faenge teilen sich ein Oracle und damit den Cache: es ist derselbe
Client, der zweimal losgeschickt wird, und was er beim ersten Fang geholt hat,
muss er beim zweiten nicht erneut anfragen. Der zweite kommt mit seiner
Budget-Haelfte dadurch weiter. Die Schaetzung selbst ist davon unberuehrt --
gezaehlt werden Knoten, nicht Anfragen.

Kein `estimate_nested`: der Umschaltpunkt zwischen den Faengen liegt bei der
Haelfte des *Gesamtbudgets*. Ein Praefix bei Budget b waere deshalb nicht
derselbe Lauf wie ein eigenstaendiger b-Lauf, der bei b/2 umschaltete. Das
liegt am Verfahren, nicht an der Umsetzung -- der Runner erkennt es daran,
dass die Methode fehlt.

Schnittstelle:
    build(dead_end="restart", n_captures=2, ...) -> PipelineEstimator
"""

from __future__ import annotations

from estimators.formulas import LincolnPetersen
from estimators.pipeline import PipelineEstimator
from oracles.local_access import CrawlOracle
from sampling.dead_ends import DEAD_ENDS
from sampling.samplers import RandomWalkSampler
from sampling.thinning import ByWalkThinning
from weighting.schemes import UniformWeighting


def build(dead_end: str = "restart", n_captures: int = 2, n_seeds: int = 1,
          burn_in: int = 0) -> PipelineEstimator:
    est = PipelineEstimator(
        name=f"capture_recapture__{dead_end}",
        oracle_cls=CrawlOracle,
        sampler=RandomWalkSampler(dead_end=DEAD_ENDS[dead_end](), n_seeds=n_seeds,
                                  n_walks=n_captures, burn_in=burn_in),
        weighting=UniformWeighting(),   # Lincoln-Petersen zaehlt nur Knoten
        formula=LincolnPetersen(),
        thinning=ByWalkThinning(n_walks=n_captures),
    )
    # s. Modul-Docstring: der Umschaltpunkt haengt am Gesamtbudget, ein Praefix
    # waere deshalb nicht derselbe Lauf.
    est.supports_nested = False
    return est
