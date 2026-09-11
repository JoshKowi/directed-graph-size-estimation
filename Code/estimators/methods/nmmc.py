"""NMMC + Thinning + Collision Counting.

Das Gegenstueck zu estimators/methods/durw.py: derselbe Aufbau, nur laeuft
statt DURW der NMMC-Sampler aus sampling.nmmc. Beide loesen dasselbe Problem --
eine bekannte Verteilung auf einer *gerichteten* Sicht --, aber mit
entgegengesetztem Preis: DURW kauft sie mit einem gleichverteilten Sprung
(also mit Kenntnis von V), NMMC mit dem Eingangsgrad. Nur der laesst sich
schaetzen, ohne V zu kennen, und deshalb ist NMMC das erste Verfahren hier mit
bekannter Zielverteilung, das mit dem reinen CrawlOracle auskommt.

Die austauschbaren Achsen:

    indeg       -- "online" | "exact" | "cross-one" | "cross-online"
                   (sampling.indegree); an die Stelle von `dead_end` bzw.
                   `jump` getreten. Entscheidet ueber das Oracle und damit
                   ueber die Kategorie. Die cross-Varianten holen den
                   Eingangsgrad aus dem Partnergraphen (config.CROSS_GRAPHS)
                   und gibt es nur fuer die Graphen, die dort einen haben.
    target      -- "uniform" (pi = u) | "indeg" (pi ~ d-). Das Ziel bestimmt
                   die Annahmewahrscheinlichkeit *und* die noetige Gewichtung.
    alpha       -- Gedaechtnis der Umverteilung, w_k = k^alpha
                   (config.NMMC_ALPHA)
    c_update_p  -- p aus Algorithmus 2 (config.NMMC_C_UPDATE_P)
    n_agents    -- Zahl der Agenten (config.NMMC_AGENTS). Sie teilen sich
                   Budget, Cache und die d--Schaetzung, haben aber je eine
                   eigene Historie -- Details in sampling.nmmc.
    shared_c    -- laesst c_t ueber die Agenten stehen statt es je Agent auf 1
                   zurueckzusetzen. Vom Paper nicht gedeckt, aber die
                   Gegenprobe dazu.
    thinning    -- "none" | "simple" | "shifted"    (sampling.thinning)
    margin      -- Safety Margin wie dort (estimators.formulas). 0 = aus.
    formula     -- "uis-collision" | "wis-col-katzir"

Ziel und Formel gehoeren zusammen: `target="indeg"` zieht proportional zu d-,
"wis-col-katzir" mit InDegreeWeighting korrigiert genau das, und
"uis-collision" auf denselben Faengen zeigt, was die Verzerrung kostet -- das
Paar durw-plain/wis-durw noch einmal. `target="uniform"` zieht dagegen bereits
gleichverteilt und braucht gar keine Gewichtung; eine gewichtete Formel bekaeme
dort w_i == 1 und rechnete dasselbe wie "uis-collision". Diese Kombination
waere also kein Verfahren, sondern ein Irrtum -- build() lehnt sie ab.

Welches Oracle ein Lauf braucht, haengt an der d--Quelle -- INDEG_ORACLES haelt
die Zuordnung. Die Kategorie folgt daraus, wird aber wie im Repo ueblich erst in
estimators/__init__.py vergeben.

Schnittstelle:
    INDEG_ORACLES: dict[str, type]
    build(target, indeg, thinning, step, margin, formula, alpha, ...)
        -> PipelineEstimator
"""

from __future__ import annotations

import numpy as np

import config
from estimators.formulas import FORMULAS
from estimators.pipeline import PipelineEstimator
from oracles.local_access import (CrawlOracle, CrossInDegreeCrawlOracle,
                                  InDegreeCrawlOracle)
from sampling.indegree import IN_DEGREES
from sampling.nmmc import NmmcSampler
from sampling.thinning import THINNINGS
from weighting.schemes import InDegreeWeighting, UniformWeighting

# Jede d--Quelle braucht ein Oracle, das sie bedienen kann. "online" kommt mit
# dem reinen Crawl-Zugriff aus -- das ist der ganze Punkt des Verfahrens.
INDEG_ORACLES: dict[str, type] = {
    "online": CrawlOracle,
    "exact": InDegreeCrawlOracle,
    # Der Partnergraph ist externes Wissen, kein Zugriff auf V des
    # geschaetzten Graphen -- beide cross-Varianten bleiben real umsetzbar.
    "cross-one": CrossInDegreeCrawlOracle,
    "cross-online": CrossInDegreeCrawlOracle,
}


def build(
    target: str = "uniform",
    indeg: str = "online",
    thinning: str = "none",
    step: int = 5,
    margin: int = 0,
    formula: str = "uis-collision",
    alpha: float = config.NMMC_ALPHA,
    c_update_p: float = config.NMMC_C_UPDATE_P,
    n_seeds: int = 1,
    burn_in: int = 0,
    n_agents: int = 1,
    shared_c: bool = False,
    aggregate=np.median,
) -> PipelineEstimator:
    if target == "uniform" and FORMULAS[formula].weighted:
        raise ValueError(
            f"target='uniform' mit formula={formula!r} ist doppelt gemoppelt: "
            "die Stichprobe ist bereits gleichverteilt, die Gewichte waeren "
            "alle 1 und die Formel rechnete dasselbe wie 'uis-collision'. "
            "Gemeint ist vermutlich target='indeg'."
        )
    if shared_c and n_agents == 1:
        raise ValueError(
            "shared_c ist mit n_agents=1 bedeutungslos -- es gibt nur ein c_t. "
            "Entweder n_agents > 1 setzen oder shared_c weglassen.")
    thin_cls = THINNINGS[thinning]
    thin = thin_cls() if thinning == "none" else thin_cls(step=step)
    weighting = (InDegreeWeighting() if FORMULAS[formula].weighted
                 else UniformWeighting())

    est = PipelineEstimator(
        # alpha nur dann im Namen, wenn es vom Default abweicht -- wie das w
        # bei DURW. Fuer Laeufe ueber die Registry ist der Name ohnehin
        # kosmetisch (estimators.build() ueberschreibt ihn), fuer Direktaufrufe
        # aus einem Notebook nicht.
        name=f"nmmc__{target}__{formula}__{indeg}__{thinning}"
             + (f"__a{alpha:g}" if alpha != config.NMMC_ALPHA else "")
             + (f"__k{n_agents}" if n_agents != 1 else "")
             + ("__sharedc" if shared_c else "")
             + (f"__m{margin}" if margin else ""),
        oracle_cls=INDEG_ORACLES[indeg],
        sampler=NmmcSampler(target=target, indeg=IN_DEGREES[indeg](),
                            alpha=alpha, c_update_p=c_update_p,
                            n_seeds=n_seeds, burn_in=burn_in,
                            n_agents=n_agents, shared_c=shared_c),
        weighting=weighting,
        formula=FORMULAS[formula](margin=margin),
        thinning=thin,
        aggregate=aggregate,
    )
    # Mit mehreren Agenten liest der Sampler oracle.budget, um die Grenzen zu
    # setzen -- ein Praefix ist dann nicht mehr bitgleich mit einem
    # eigenstaendigen kuerzeren Lauf. Dasselbe Muster wie capture_recapture.
    est.supports_nested = (n_agents == 1)
    return est
