"""Capture-Recapture als Pipeline: k Faenge, eine Formel darueber.

    lincoln-petersen  n_hat = n1*n2/m                  zwei Faenge
    chapman           n_hat = (n1+1)(n2+1)/(m+1) - 1   zwei Faenge, immer definiert
    schnabel          n_hat = sum C_t*M_t / sum R_t    beliebig viele Faenge
    cross             Kollisionen zwischen den Faengen, beliebig viele
    cross-wis         dasselbe mit Gradkorrektur -- die einzige Variante, die
                      Gewichte nutzen kann (Begruendung in formulas.py)

mit n1, n2 als Anzahl *verschiedener* besuchter Knoten je Fang und m ihrer
Ueberschneidung. Die Formeln stehen in estimators/formulas.py.

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
    build(dead_end="restart", formula="lincoln-petersen", n_captures=2, ...)
        -> PipelineEstimator
"""

from __future__ import annotations

from estimators.formulas import SETS_FORMULAS
from estimators.pipeline import PipelineEstimator
from oracles.local_access import CrawlOracle
from sampling.dead_ends import DEAD_ENDS
from sampling.samplers import RandomWalkSampler
from sampling.thinning import ByWalkThinning
from weighting.schemes import InverseDegreeWeighting, UniformWeighting


def build(dead_end: str = "restart", formula: str = "lincoln-petersen",
          n_captures: int = 2, n_seeds: int = 1,
          burn_in: int = 0) -> PipelineEstimator:
    if formula in ("lincoln-petersen", "chapman") and n_captures != 2:
        raise ValueError(
            f"{formula!r} ist auf zwei Faenge festgelegt, n_captures="
            f"{n_captures} geht nur mit 'schnabel' oder 'cross'."
        )
    est = PipelineEstimator(
        name=f"capture-recapture__{dead_end}__{formula}",
        oracle_cls=CrawlOracle,
        sampler=RandomWalkSampler(dead_end=DEAD_ENDS[dead_end](), n_seeds=n_seeds,
                                  n_walks=n_captures, burn_in=burn_in),
        # Nur die Kollisions-Variante kann Gewichte nutzen; die mengenbasierten
        # Formeln (LP, Chapman, Schnabel) zaehlen Knoten und ignorieren sie.
        weighting=(InverseDegreeWeighting() if SETS_FORMULAS[formula].weighted
                   else UniformWeighting()),
        formula=SETS_FORMULAS[formula](),
        thinning=ByWalkThinning(n_walks=n_captures),
    )
    # s. Modul-Docstring: der Umschaltpunkt haengt am Gesamtbudget, ein Praefix
    # waere deshalb nicht derselbe Lauf.
    est.supports_nested = False
    return est
