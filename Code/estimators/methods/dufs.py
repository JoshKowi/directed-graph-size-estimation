"""DUFS + Thinning + Collision Counting.

Das Gegenstueck zu estimators/methods/durw.py: derselbe Aufbau, nur laeuft
statt des einen DURW-Walkers das k-Walker-Frontier-Sampling aus sampling.dufs.
Stationaerverteilung, Sprungregel und Gewichtung sind identisch zu DURW --
verschieden ist allein die Autokorrelation der Sample-Folge, und genau die ist
das, was der Kollisionsschaetzer zu spueren bekommt.

Die austauschbaren Achsen sind dieselben wie bei DURW:

    jump        -- "uniform", eine Listenquelle (namelists.SOURCES) oder
                   "rand<P>" (sampling.jumps). Bei DUFS steuert sie *zwei*
                   Dinge: das Sprungziel und die k Startknoten.
    jump_weight -- w der Sprungregel (config.DURW_JUMP_WEIGHT)
    n_walkers   -- k (config.DUFS_WALKERS). k = 1 ist exakt DURW.
    thinning    -- "none" | "simple" | "shifted"    (sampling.thinning)
    margin      -- Safety Margin wie dort (estimators.formulas). 0 = aus.
                   Achtung: er streicht Paare mit kleinem Abstand *in der
                   Sample-Folge*; bei k > 1 stammen die aber ueberwiegend von
                   verschiedenen Walkern, er korrigiert also immer weniger.
    formula     -- "uis-collision" | "wis-col-katzir"

**Der eine echte Unterschied zu durw.build(): `no_jumps` behaelt das
Sprung-Oracle.** Bei DURW schaltet `no_jumps` auf das blosse CrawlOracle um --
ohne Sprung wird oracle.random_node() nie gebraucht. Bei DUFS gilt das nicht:
die k Startknoten kommen aus derselben Quelle wie die Spruenge, und k ueber
eine Namensliste verteilte Startpunkte sind der eigentliche Inhalt dieser
Variante. Daraus folgt auch die Kategorie: `dufs-<liste>__nojump` ist
REALIZABLE (die Liste ist externes Wissen), `dufs__uniform__nojump` dagegen
COMPARISON (gleichverteilt aus V zu ziehen setzt V voraus) -- waehrend
`durw-plain__nojump` ohne jede Sprungquelle auskommt und deshalb
uneingeschraenkt REALIZABLE ist. Vergeben wird die Kategorie wie ueblich erst
in estimators/__init__.py.

Nicht uebernommen aus DURW: `history_jumps` (durwhist-*) und
`jump_set_weighting` (durwset-*, widerlegt, s. weighting.DurwJumpSetWeighting).
Beide haben kein DUFS-Gegenstueck, weil es dafuer keine Altergebnisse zu
erhalten gibt.

Schnittstelle:
    build(jump, thinning, step, margin, formula, jump_weight, n_walkers, ...)
        -> PipelineEstimator
"""

from __future__ import annotations

from functools import partial

import numpy as np

import config
from estimators.formulas import FORMULAS
from estimators.methods.durw import jump_oracle
from estimators.pipeline import PipelineEstimator
from sampling.dufs import DufsSampler
from sampling.jumps import jump_strategy, subset_percent
from sampling.thinning import THINNINGS
from weighting.schemes import (DurwWeighting, InverseDegreeWeighting,
                               UniformWeighting)


def build(
    jump: str = "uniform",
    thinning: str = "none",
    step: int = 5,
    margin: int = 0,
    formula: str = "uis-collision",
    jump_weight: float = config.DURW_JUMP_WEIGHT,
    n_walkers: int = config.DUFS_WALKERS,
    burn_in: int = 0,
    draw_burn_in: int = config.DEFAULT_DRAW_BURN_IN,
    draw_limit: int | None = None,
    cost_miss: float = config.COST_DRAW_MISS,
    union_jumps: bool = False,
    no_jumps: bool = False,
    aggregate=np.median,
) -> PipelineEstimator:
    # `no_jumps`: reiner Random Walk auf G_u von k Startknoten aus. Anders als
    # bei durw.build() bleiben jump/draw_limit/draw_burn_in wirksam -- sie
    # steuern die Ziehung der Startknoten. Bedeutungslos sind nur w und die
    # Sprungziel-Strategie.
    if no_jumps:
        if union_jumps:
            raise ValueError(
                "no_jumps schliesst union_jumps aus -- ohne Sprung gibt es kein "
                "Sprungziel. Die Sprungart selbst bleibt wirksam, sie liefert "
                "die k Startknoten.")
        if jump_weight != config.DURW_JUMP_WEIGHT:
            raise ValueError(
                f"no_jumps und jump_weight={jump_weight} passen nicht zusammen: "
                "ohne Sprung geht w weder in die Absprungregel noch in die "
                "Auswahlgewichte ein.")
    # `union_jumps`: Sprung gleichverteilt auf S u H, Absprungregel und Gewicht
    # des Originals (sampling.dufs / sampling.durw). H ist bei DUFS die
    # Historie aller k Walker. Die Liste muss dafuer jeden Knoten nur einmal
    # enthalten -- das Oracle bekommt deshalb unique=True.
    if union_jumps:
        if jump == "uniform":
            raise ValueError(
                "union_jumps mit jump='uniform' ist das Original -- S = V, "
                "die Historie fuegt nichts hinzu. Ein eigener Eintrag waere "
                "ein Duplikat von wis-dufs__uniform__k<K>__margin.")
        if draw_burn_in:
            raise ValueError(
                f"union_jumps braucht draw_burn_in = 0, ist {draw_burn_in}: "
                "mit Burn-in laege das Sprungziel nicht mehr in S u H.")

    oracle_cls = jump_oracle(jump)
    if subset_percent(jump) is not None:
        # Die Zufallsteilmenge hat weder Nieten noch eine Listenlaenge, und
        # ein Burn-in nach dem Sprung ist dort nicht vorgesehen.
        if draw_burn_in or draw_limit:
            raise ValueError(f"jump={jump!r} kennt weder draw_burn_in noch "
                             "draw_limit -- der Anteil steckt im Namen.")
    elif jump != "uniform":
        oracle_cls = partial(oracle_cls, burn_in=draw_burn_in,
                             cost_miss=cost_miss, limit=draw_limit,
                             **({"unique": True} if union_jumps else {}))

    thin_cls = THINNINGS[thinning]
    thin = thin_cls() if thinning == "none" else thin_cls(step=step)
    if not FORMULAS[formula].weighted:
        weighting = UniformWeighting()
    elif no_jumps:
        # w -> 0: pi(v) ~ deg_Gu(v) ohne Sigma-Term. InverseDegreeWeighting
        # liest denselben Sample.degree (der bei DUFS wie bei DURW den G_u-Grad
        # traegt) und ist hier exakt richtig.
        weighting = InverseDegreeWeighting()
    else:
        weighting = DurwWeighting(jump_weight)

    # Kosmetischer Name fuer Direktaufrufe; estimators.build() ueberschreibt
    # ihn durch den Registry-Namen.
    name = (f"dufs__{formula}__{jump}__{thinning}__k{n_walkers}"
            + (f"__n{draw_limit}" if draw_limit else "")
            + ("__union" if union_jumps else "")
            + ("__nojump" if no_jumps else "")
            + (f"__w{jump_weight:g}"
               if jump_weight != config.DURW_JUMP_WEIGHT else "")
            + (f"__m{margin}" if margin else ""))

    est = PipelineEstimator(
        name=name,
        oracle_cls=oracle_cls,
        sampler=DufsSampler(jump=jump_strategy(jump), jump_weight=jump_weight,
                            n_walkers=n_walkers, burn_in=burn_in,
                            union_jumps=union_jumps, no_jumps=no_jumps),
        weighting=weighting,
        formula=FORMULAS[formula](margin=margin),
        thinning=thin,
        aggregate=aggregate,
    )
    # Die Ziehung haengt vom Gesamtbudget ab: die Zahl der Startknoten ist auf
    # config.DUFS_MAX_SEED_SHARE des Budgets gedeckelt (s.
    # DufsSampler._n_seeds), ein Lauf mit kleinerem Budget haette also weniger
    # Walker. Ein Praefix des grossen Laufs ist damit nicht derselbe Lauf --
    # dieselbe Lage wie bei capture_recapture. Nachgewiesen mit
    # check_nested.py: ohne diese Zeile weicht wis-dufs__uniform__k100__margin
    # beim kleinsten Budget ab (5 statt 32 besuchte Knoten).
    #
    # Der Deckel selbst ist nicht verhandelbar: ohne ihn liefe DUFS bei
    # kleinem Budget und grossem k gar keinen Schritt. Also lieber je Budget
    # ein eigener Lauf.
    est.supports_nested = False
    return est
