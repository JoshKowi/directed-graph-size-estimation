"""DURW + Thinning + Collision Counting.

Das Gegenstueck zu estimators/methods/random_walk_collision.py: derselbe
Aufbau, nur laeuft statt des einfachen Random Walks der DURW aus
sampling.durw. Er baut sich waehrend des Laufs einen ungerichteten Graphen G_u
und springt mit Wahrscheinlichkeit w/(w + deg_Gu(v)) -- dadurch ist seine
Stationaerverteilung auch auf den *gerichteten* Views geschlossen bekannt,
was fuer den einfachen Random Walk gerade nicht gilt.

Die austauschbaren Achsen:

    jump        -- "uniform", eine Listenquelle (namelists.SOURCES) oder
                   "rand<P>" (sampling.jumps); an die Stelle von `dead_end`
                   getreten. DURW braucht keine Sackgassen-Strategie, weil eine
                   Sackgasse nie absorbierend wird: ueber eine Kante erreicht,
                   traegt sie diese Kante als Rueckweg in ihrem G_u-Grad; per
                   Sprung erreicht, hat sie deg_Gu = 0 und springt zwingend
                   weiter (w/(w+0) = 1).
    jump_weight -- w der Sprungregel (config.DURW_JUMP_WEIGHT)
    thinning    -- "none" | "simple" | "shifted"    (sampling.thinning)
    margin      -- Safety Margin wie dort (estimators.formulas). 0 = aus.
    formula     -- "uis-collision" | "wis-col-katzir" | "ie2-xcol" | "ie2m-xcol"
    a_source    -- nur fuer die IE2-Formeln: aus welcher Nachbarschaft die
                   Menge A gebaut wird, "raw" (rohe Ausgangsnachbarn) oder
                   "gu" (eingefrorene G_u-Nachbarschaft). Siehe
                   sampling.observed.

"wis-col-katzir" gehoert mit DurwWeighting zusammen: der Walk zieht
proportional zu (w + deg_Gu), das Gewicht korrigiert genau das. Nicht mit
InverseDegreeWeighting verwechseln -- die passt zum einfachen Random Walk,
nicht zu DURW *mit Sprung*. "uis-collision" ignoriert die Gewichte und
unterstellt gleichverteilte Ziehungen; die Differenz ist der Preis der
Verzerrung, also das, was DURW ueberhaupt korrigiert.

`no_jumps` -- der Grenzfall w -> 0 (sampling.durw): reiner Random Walk auf
G_u, kein Sprung, keine Sprungoption anwendbar. Dort *ist* pi(v) ~ deg_Gu(v)
ohne Sigma-Term, und InverseDegreeWeighting (liest denselben `Sample.degree`)
ist dafuer genau richtig -- die Warnung oben gilt nur fuer w > 0.

Welches Oracle ein Lauf braucht, haengt an der Sprungart -- JUMP_ORACLES haelt
die Zuordnung. Die Kategorie (real umsetzbar oder nicht) folgt daraus, wird
aber wie im Repo ueblich erst in estimators/__init__.py vergeben.

Die IE2-Formeln (estimators.formulas.IE2SetEstimator) zaehlen Treffer gegen die
Vereinigung aller beobachteten Nachbarschaften statt Knoten-Kollisionen. Sie
brauchen dafuer, dass der Sampler die Nachbarlisten aufbewahrt; das Flag dafuer
wird aus `EstimationFormula.needs_neighbors` abgeleitet, genau wie `with_degree`
aus `WeightingScheme.needs_degree`. Am Walk aendert es nichts, an der
Gewichtung auch nicht -- IE2 ist `weighted`, bekommt also dasselbe
Weighting-Schema wie "wis-col-katzir".

Schnittstelle:
    JUMP_ORACLES: dict[str, type]
    jump_oracle(jump) -> type | partial     -- JUMP_ORACLES plus rand<P>
    build(jump, thinning, step, margin, formula, jump_weight, ...)
        -> PipelineEstimator
"""

from __future__ import annotations

from functools import partial

import numpy as np

import config
import namelists
from estimators.formulas import FORMULAS
from estimators.pipeline import PipelineEstimator
from oracles.local_access import CrawlOracle, JumpCrawlOracle
from oracles.name_list import NameListOracle
from oracles.random_subset import RandomSubsetOracle
from sampling.durw import DurwSampler
from sampling.jumps import jump_strategy, subset_percent
from sampling.thinning import THINNINGS
from weighting.schemes import (DurwJumpSetWeighting, DurwSigmaWeighting,
                               DurwWeighting, InverseDegreeWeighting,
                               UniformWeighting)

# Jede Sprungart braucht ein Oracle, das sie bedienen kann. Eine spaeter
# hinzukommende Sprungart, die ihr Ziel aus externen Daten simuliert, traegt
# hier ihr eigenes Oracle ein -- am Sampler aendert das nichts.
JUMP_ORACLES: dict[str, object] = {
    "uniform": JumpCrawlOracle,
}
# Die Listenquellen ziehen ueber NameListOracle. Sie brauchen keine Kenntnis
# von V -- deshalb sind sie in estimators/__init__.py REALIZABLE, waehrend
# "uniform" COMPARISON bleibt.
for _src in sorted(namelists.SOURCES):
    JUMP_ORACLES[_src] = partial(NameListOracle, source=_src)
del _src


def jump_oracle(jump: str):
    """Oracle zur Sprungart. "rand<P>" -- Sprung auf eine feste Zufalls-
    teilmenge mit P % der Knoten (oracles.random_subset) -- steht nicht in
    JUMP_ORACLES, weil P frei waehlbar ist."""
    p = subset_percent(jump)
    if p is not None:
        return partial(RandomSubsetOracle, percent=p)
    return JUMP_ORACLES[jump]


def build(
    jump: str = "uniform",
    thinning: str = "none",
    step: int = 5,
    margin: int = 0,
    formula: str = "uis-collision",
    a_source: str = "raw",
    jump_weight: float = config.DURW_JUMP_WEIGHT,
    n_seeds: int = 1,
    burn_in: int = 0,
    draw_burn_in: int = config.DEFAULT_DRAW_BURN_IN,
    draw_limit: int | None = None,
    cost_miss: float = config.COST_DRAW_MISS,
    jump_set_weighting: bool = False,
    history_jumps: bool = False,
    history_weight: float = 1.0,
    union_jumps: bool = False,
    union_seen: bool = False,
    no_jumps: bool = False,
    aggregate=np.median,
) -> PipelineEstimator:
    # `no_jumps`: der Grenzfall w -> 0, reiner Random Walk auf G_u (siehe
    # sampling.durw). Jede Sprung-Option ist dann bedeutungslos -- wer sie
    # trotzdem setzt, meint vermutlich etwas anderes als das, was passiert.
    if no_jumps and (jump != "uniform" or jump_weight != config.DURW_JUMP_WEIGHT
                     or history_jumps or union_jumps or union_seen
                     or jump_set_weighting or draw_burn_in or draw_limit):
        raise ValueError(
            "no_jumps schliesst jede Sprung-Option aus -- ohne Sprung haben "
            "jump/jump_weight/history_jumps/union_jumps/union_seen/"
            "jump_set_weighting/draw_burn_in/draw_limit keine Wirkung mehr.")
    # `union_jumps`: Sprung gleichverteilt auf S u H, Absprungregel und Gewicht
    # des Originals (sampling.durw). Die Liste muss dafuer jeden Knoten nur
    # einmal enthalten -- das Oracle bekommt deshalb unique=True.
    if union_jumps:
        if history_jumps or union_seen or jump_set_weighting:
            raise ValueError("union_jumps schliesst history_jumps, union_seen "
                             "und jump_set_weighting aus -- alle bringen ihre "
                             "eigene Vorstellung von H bzw. Gewichtung mit.")
        if jump == "uniform":
            raise ValueError(
                "union_jumps mit jump='uniform' ist das Original -- S = V, "
                "die Historie fuegt nichts hinzu. Ein eigener Eintrag waere "
                "ein Duplikat von wis-durw__uniform__margin.")
        if draw_burn_in:
            raise ValueError(
                f"union_jumps braucht draw_burn_in = 0, ist {draw_burn_in}: "
                "mit Burn-in laege das Sprungziel nicht mehr in S u H.")
    # `union_seen`: wie union_jumps, aber H sind alle je *gesehenen* Knoten
    # (auch nur als Nachbar beobachtete, noch nicht besuchte) statt nur der
    # tatsaechlich besuchten. Dieselbe Gewichtung, dieselben Randbedingungen.
    if union_seen:
        if history_jumps or jump_set_weighting:
            raise ValueError("union_seen schliesst history_jumps und "
                             "jump_set_weighting aus -- es bringt die "
                             "Gewichtung des Originals mit.")
        if jump == "uniform":
            raise ValueError(
                "union_seen mit jump='uniform' ist das Original -- S = V, "
                "gesehene Knoten fuegen nichts hinzu.")
        if draw_burn_in:
            raise ValueError(
                f"union_seen braucht draw_burn_in = 0, ist {draw_burn_in}: "
                "mit Burn-in laege das Sprungziel nicht mehr in S u H_gesehen.")
    # `history_jumps`: Sprung auf S u H, mit passender Sprungregel *und*
    # Gewichtung -- die korrekte Fassung dessen, was jump_set_weighting
    # versucht hat. Siehe weighting.DurwSigmaWeighting.
    if history_jumps:
        if jump_set_weighting:
            raise ValueError("history_jumps und jump_set_weighting schliessen "
                             "sich aus -- history_jumps bringt die eigene "
                             "Gewichtung mit.")
        if jump == "uniform":
            raise ValueError(
                "history_jumps ist fuer Listenquellen gedacht; beim "
                "gleichverteilten Sprung deckt der Sprung V ohnehin ab.")
        if draw_burn_in:
            raise ValueError(
                f"history_jumps braucht draw_burn_in = 0, ist {draw_burn_in}: "
                "ein Burn-in im Oracle liefert Knoten, die weder in S noch in H "
                "liegen muessen, und die Landeverteilung waere wieder unbekannt.")
    # `jump_set_weighting` ist die Abweichung vom Original: bei einem Sprung
    # aus einer Namensliste ist sigma nur mit der Trefferteilmenge S verbunden,
    # nicht mit ganz V, und pi(v) ~ deg_Gu(v) + w*1[v in S] statt
    # pi(v) ~ w + deg_Gu(v). Siehe weighting.DurwJumpSetWeighting.
    if jump_set_weighting:
        if jump == "uniform":
            raise ValueError(
                "jump_set_weighting mit jump='uniform' waere rechnerisch "
                "identisch zu DurwWeighting -- dort ist S = V. Ein eigener "
                "Eintrag dafuer waere ein Duplikat."
            )
        if draw_burn_in:
            raise ValueError(
                f"jump_set_weighting braucht draw_burn_in = 0, ist "
                f"{draw_burn_in}: laeuft nach dem Treffer noch ein Burn-in, "
                "liefert der Sprung einen Knoten ausserhalb von S, und die "
                "Sprungverteilung ist wieder unbekannt."
            )
    # `burn_in` verwirft die ersten Schritte des Walks, `draw_burn_in` die
    # ersten Schritte *nach jedem Sprung* -- zwei verschiedene Dinge, die nicht
    # verwechselt werden duerfen. Letzteres kennt nur das NameListOracle.
    if no_jumps:
        # Kein Sprung -> nie oracle.random_node(): das einfache CrawlOracle
        # reicht, es braucht kein privilegiertes Wissen ueber V. Deshalb ist
        # dieser Zweig REALIZABLE (s. estimators/__init__.py), staerker noch
        # als jump="uniform" (COMPARISON).
        oracle_cls = CrawlOracle
    else:
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
                                 # nur wenn gesetzt: sonst aendert sich der
                                 # Walk-Schluessel der vorhandenen Estimators
                                 **({"unique": True}
                                    if (union_jumps or union_seen) else {}))
    thin_cls = THINNINGS[thinning]
    thin = thin_cls() if thinning == "none" else thin_cls(step=step)
    # Nur die IE2-Formeln brauchen die Nachbarlisten -- und nur sie kennen
    # a_source. Ein a_source bei einer Formel, die ihn ignoriert, waere ein
    # stiller Tippfehler; deshalb hier laut.
    needs_nbrs = FORMULAS[formula].needs_neighbors
    if not needs_nbrs and a_source != "raw":
        raise ValueError(
            f"a_source={a_source!r} hat bei formula={formula!r} keine Wirkung "
            "-- A wird nur von den IE2-Formeln gebraucht.")
    _a_tag = f"__{a_source}" if needs_nbrs else ""
    if not FORMULAS[formula].weighted:
        weighting = UniformWeighting()
    elif no_jumps:
        # w -> 0: pi(v) ~ deg_Gu(v) ohne Sigma-Term -- InverseDegreeWeighting
        # liest denselben Sample.degree und ist hier exakt richtig (s.
        # Modul-Docstring).
        weighting = InverseDegreeWeighting()
    elif history_jumps:
        weighting = DurwSigmaWeighting(jump_weight)
    elif jump_set_weighting:
        weighting = DurwJumpSetWeighting(jump_weight)
    else:
        weighting = DurwWeighting(jump_weight)

    if no_jumps:
        # Ohne Sprung sind jump/w/hist/union bedeutungslos -- eigener,
        # schmalerer Name statt des Astes unten (der bleibt fuer w>0 exakt
        # wie bisher).
        name = (f"durw__{formula}{_a_tag}__nojump__{thinning}"
                + (f"__m{margin}" if margin else ""))
        sampler = DurwSampler(no_jumps=True, n_seeds=n_seeds, burn_in=burn_in,
                              collect_nbrs=needs_nbrs)
    else:
        # w nur dann im Namen, wenn es vom Default abweicht -- sonst hiessen
        # die Registry-Eintraege ohne w-Angabe anders als bisher. Fuer Laeufe
        # ueber die Registry ist der Name ohnehin kosmetisch (estimators.build()
        # ueberschreibt ihn), fuer Direktaufrufe aus einem Notebook nicht.
        name = (f"durw__{formula}{_a_tag}__{jump}__{thinning}"
                + (f"__n{draw_limit}" if draw_limit else "")
                + ("__inS" if jump_set_weighting else "")
                + (f"__hist{history_weight:g}" if history_jumps else "")
                + ("__union" if union_jumps else "")
                + ("__unionE" if union_seen else "")
                + (f"__w{jump_weight:g}" if jump_weight != config.DURW_JUMP_WEIGHT else "")
                + (f"__m{margin}" if margin else ""))
        sampler = DurwSampler(jump=jump_strategy(jump), jump_weight=jump_weight,
                              n_seeds=n_seeds, burn_in=burn_in,
                              history_jumps=history_jumps,
                              history_weight=history_weight,
                              union_jumps=union_jumps,
                              union_seen=union_seen,
                              collect_nbrs=needs_nbrs)

    return PipelineEstimator(
        name=name,
        oracle_cls=oracle_cls,
        sampler=sampler,
        weighting=weighting,
        formula=(FORMULAS[formula](margin=margin, a_source=a_source)
                 if needs_nbrs else FORMULAS[formula](margin=margin)),
        thinning=thin,
        aggregate=aggregate,
    )
