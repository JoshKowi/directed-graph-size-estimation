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

`sampler="randomwalk"` (Default) crawlt real ab einem Einstiegsknoten
(`CrawlOracle` + `RandomWalkSampler`, `dead_end` greift). `sampler="uniform"`
ersetzt das durch unabhaengiges, gleichverteiltes Ziehen aus ganz V
(`UniformNodeOracle` + `UniformSampler`) -- nicht real umsetzbar, aber die
Referenz dafuer, wie Capture-Recapture ohne Grad- und Walk-Verzerrung
abschneidet. `dead_end` ist dabei ohne Wirkung (kein Walk, keine Sackgassen).
`sampler="durw"` crawlt ebenfalls, baut sich dabei aber den ungerichteten
Graphen G_u auf und springt gradproportional (`sampling.durw`); statt
`dead_end` greift dort `jump`. Jeder Fang baut sein eigenes G_u auf -- sonst
waeren die Faenge ueber die geteilte Historie abhaengig, und genau ihre
Unabhaengigkeit ist die Voraussetzung des Verfahrens.

Die Gewichtung haengt am Sampler, nicht nur an der Formel: `cross-wis`
korrigiert beim Random Walk mit 1/deg, bei DURW aber mit 1/(w + deg_Gu). Jeder
Zweig setzt sie deshalb selbst.

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
    build(dead_end="restart", formula="lincoln-petersen", n_captures=2,
          sampler="randomwalk", jump="uniform", ...) -> PipelineEstimator
"""

from __future__ import annotations

import config
from estimators.formulas import SETS_FORMULAS
from estimators.methods.durw import JUMP_ORACLES
from estimators.pipeline import PipelineEstimator
from oracles.global_access import UniformNodeOracle
from oracles.local_access import CrawlOracle
from sampling.dead_ends import DEAD_ENDS
from sampling.durw import DurwSampler
from sampling.jumps import JUMPS
from sampling.samplers import RandomWalkSampler, UniformSampler
from sampling.thinning import ByWalkThinning
from weighting.schemes import (DurwWeighting, InverseDegreeWeighting,
                               UniformWeighting)


def build(dead_end: str = "restart", formula: str = "lincoln-petersen",
          n_captures: int = 2, sampler: str = "randomwalk", n_seeds: int = 1,
          burn_in: int = 0, jump: str = "uniform",
          jump_weight: float = config.DURW_JUMP_WEIGHT) -> PipelineEstimator:
    if formula in ("lincoln-petersen", "chapman") and n_captures != 2:
        raise ValueError(
            f"{formula!r} ist auf zwei Faenge festgelegt, n_captures="
            f"{n_captures} geht nur mit 'schnabel' oder 'cross'."
        )
    # Nur die Kollisions-Variante kann Gewichte nutzen; die mengenbasierten
    # Formeln (LP, Chapman, Schnabel) zaehlen Knoten und ignorieren sie.
    weighted = SETS_FORMULAS[formula].weighted

    if sampler == "uniform":
        oracle_cls = UniformNodeOracle
        weighting = InverseDegreeWeighting() if weighted else UniformWeighting()
        # Ohne Gradgewichtung wird der Grad nicht gebraucht und deshalb auch
        # nicht abgefragt -- ein Sample kostet dann eine Einheit statt zwei.
        drawer = UniformSampler(n_walks=n_captures,
                                with_degree=weighting.needs_degree)
        tag = "uniform"
    elif sampler == "randomwalk":
        oracle_cls = CrawlOracle
        weighting = InverseDegreeWeighting() if weighted else UniformWeighting()
        drawer = RandomWalkSampler(dead_end=DEAD_ENDS[dead_end](), n_seeds=n_seeds,
                                   n_walks=n_captures, burn_in=burn_in)
        tag = dead_end
    elif sampler == "durw":
        oracle_cls = JUMP_ORACLES[jump]
        # 1/(w + deg_Gu), nicht 1/deg -- s. Modul-Docstring.
        weighting = DurwWeighting(jump_weight) if weighted else UniformWeighting()
        drawer = DurwSampler(jump=JUMPS[jump](), jump_weight=jump_weight,
                             n_seeds=n_seeds, n_walks=n_captures, burn_in=burn_in)
        tag = f"durw-{jump}"
    else:
        raise ValueError(
            f"Unbekannter sampler {sampler!r} -- 'randomwalk', 'durw' oder 'uniform'")

    est = PipelineEstimator(
        name=f"capture-recapture__{tag}__{formula}",
        oracle_cls=oracle_cls,
        sampler=drawer,
        weighting=weighting,
        formula=SETS_FORMULAS[formula](),
        thinning=ByWalkThinning(n_walks=n_captures),
    )
    # s. Modul-Docstring: der Umschaltpunkt haengt am Gesamtbudget, ein Praefix
    # waere deshalb nicht derselbe Lauf.
    est.supports_nested = False
    return est
