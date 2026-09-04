"""Konkrete Sampler.

Schnittstelle:
    class UniformSampler(Sampler)     -- braucht oracle.random_node()
    class RandomWalkSampler(Sampler)  -- braucht oracle.seed_nodes()/neighbors()
"""

from __future__ import annotations

import math

from oracles.base import BudgetExceeded
from sampling.base import Sample, Sampler
from sampling.dead_ends import DeadEndStrategy, RestartToStart


class UniformSampler(Sampler):
    """Unabhaengige, gleichverteilte Knoten (nur mit globalem Oracle moeglich).

    `n_walks` > 1 teilt die Ziehungen in ebenso viele Faenge auf (Budget
    gleichmaessig verteilt, letzter Fang bis zum Budgetende) -- die Form, die
    Capture-Recapture braucht (siehe estimators/methods/capture_recapture.py
    und sampling.thinning.ByWalkThinning). Bei unabhaengigen Ziehungen aendert
    das an der Verteilung nichts, nur an der Buchhaltung: Samples tragen den
    Fang-Index in Sample.walk, sonst liefen alle Faenge unter walk=0 zusammen
    und ByWalkThinning saehe nur einen.

    `with_degree` entscheidet, ob zu jedem gezogenen Knoten auch sein Grad
    abgefragt wird. Das ist eine *zweite* Anfrage nach aussen und kostet noch
    einmal so viel wie die Ziehung selbst (oracles.base._fetch): mit
    Gradabfrage kostet ein Sample zwei Einheiten, ohne eine. Anders als beim
    Random Walk faellt der Grad hier nicht nebenbei ab -- dort ist die
    Nachbarabfrage ohnehin noetig, um weiterzugehen, und liefert den Grad
    gratis mit.

    Gebraucht wird der Grad allein von weighting.InverseDegreeWeighting; die
    build()-Funktionen setzen das Flag deshalb aus `weighting.needs_degree`.
    Ohne Abfrage tragen die Samples `degree=None`.
    """

    name = "uniform"

    def __init__(self, n_walks: int = 1, with_degree: bool = True) -> None:
        self.n_walks = n_walks
        self.with_degree = with_degree

    def key(self) -> str:
        # with_degree gehoert in den Schluessel: es aendert die Kosten je
        # Sample und damit die Trajektorie bei gegebenem Budget. Ohne das
        # wuerde --share-walks Estimators mit und ohne Gradabfrage in einen
        # gemeinsamen Lauf stecken (s. estimators/pipeline.py).
        return f"{self.name}|walks{self.n_walks}|deg{int(self.with_degree)}"

    def sample(self, oracle) -> list[Sample]:
        samples: list[Sample] = []
        try:
            for walk in range(self.n_walks):
                # Letzter Fang bis zum Budgetende, s. RandomWalkSampler.sample().
                limit = (math.inf if walk == self.n_walks - 1
                         else oracle.budget * (walk + 1) / self.n_walks)
                while oracle.queries < limit:
                    u = oracle.random_node()
                    deg = oracle.degree(u) if self.with_degree else None
                    samples.append(Sample(u, deg, len(samples), walk))
                    oracle.mark()  # fuer Budget-Zwischenstaende, s. oracles.base
        except BudgetExceeded:
            return samples
        return samples


class RandomWalkSampler(Sampler):
    """Random Walk ueber Nachbarschaftsabfragen; liefert die volle Trajektorie.

    Das Aufteilen der Trajektorie in Sample-Sets uebernimmt sampling.thinning.

    `dead_end` bestimmt, wie es bei einem Knoten ohne ausgehende Kanten
    weitergeht (siehe sampling.dead_ends). `restart_prob` ist davon unabhaengig
    und teleportiert zusaetzlich mit fester Wahrscheinlichkeit zum Anfang.

    `n_walks` > 1 laesst mehrere Walks nacheinander laufen, jeder mit eigenem
    Einstieg; die Samples tragen den Index in `Sample.walk`. Das ist die Form,
    die Capture-Recapture braucht -- zwei Faenge, die nicht einer die
    Fortsetzung des anderen sind. Walk i endet, sobald er seinen Anteil am
    Budget (i+1)/n verbraucht hat; der letzte laeuft bis zum Budgetende. Bleibt
    ein Fang vorzeitig stehen, faellt sein Rest dem naechsten zu.

    `allow_self_loops=False` (Default) behandelt eine Kante auf den eigenen
    Knoten nicht als Weiterkommen. Das ist noetig, weil ein Knoten, dessen
    einzige Kante auf ihn selbst zeigt, den Walk sonst absorbiert -- in
    Slashdot0811 trifft das auf 6418 Knoten (8,3 %) zu. Mit dieser Einstellung
    ist ein solcher Knoten eine Sackgasse und die dead_end-Strategie greift.
    Die in Sample.degree berichtete Gradzahl bleibt der rohe Ausgangsgrad,
    schliesst also eine eventuelle Selbstkante mit ein.

    Hinweis: die Stationaerverteilung pi(u) ~ deg(u) gilt streng nur fuer
    ungerichtete, zusammenhaengende Graphen -- und jede Sackgassen-Strategie
    veraendert sie zusaetzlich. Genau das soll der Vergleich sichtbar machen.
    """

    def __init__(
        self,
        dead_end: DeadEndStrategy | None = None,
        n_seeds: int = 1,
        n_walks: int = 1,
        restart_prob: float = 0.0,
        burn_in: int = 0,
        allow_self_loops: bool = False,
    ) -> None:
        self.dead_end = dead_end or RestartToStart()
        self.n_seeds = n_seeds
        self.n_walks = n_walks
        self.restart_prob = restart_prob
        self.burn_in = burn_in
        self.allow_self_loops = allow_self_loops
        self.name = f"random_walk_{self.dead_end.name}"

    def key(self) -> str:
        """Alles, was den Walk steuert -- die Sackgassen-Strategie steckt
        bereits im Namen."""
        return (f"{self.name}|seeds{self.n_seeds}|walks{self.n_walks}"
                f"|burn{self.burn_in}|restart{self.restart_prob:g}"
                f"|loops{int(self.allow_self_loops)}")

    def _step(self, u, nbrs, rng):
        """Zufaelliger Nachbar != u, oder None wenn es keinen gibt.

        Der Normalfall kostet O(1); die gefilterte Liste wird nur gebaut, wenn
        die Selbstkante tatsaechlich gezogen wurde (Wahrscheinlichkeit 1/Grad).
        """
        v = nbrs[rng.randrange(len(nbrs))]
        if self.allow_self_loops or v != u:
            return v
        others = nbrs[nbrs != u]
        return others[rng.randrange(len(others))] if len(others) else None

    def sample(self, oracle) -> list[Sample]:
        trace: list[Sample] = []
        current: list[Sample] = []
        try:
            for walk in range(self.n_walks):
                # Der letzte Walk laeuft bis zum Budgetende. Damit ist der
                # Normalfall n_walks=1 exakt der alte Code: `while True`.
                limit = (math.inf if walk == self.n_walks - 1
                         else oracle.budget * (walk + 1) / self.n_walks)
                current = []
                seeds = oracle.seed_nodes(self.n_seeds)
                start = seeds[0]
                u = start
                path = [u]
                step = 0
                while oracle.queries < limit:
                    nbrs = oracle.neighbors(u)
                    if step >= self.burn_in:
                        current.append(Sample(u, len(nbrs), step, walk))
                        oracle.mark()  # fuer Budget-Zwischenstaende, s. oracles.base
                    step += 1

                    nxt = self._step(u, nbrs, oracle.rng) if len(nbrs) else None
                    if nxt is None:
                        # bewusst nur die Besuchsfolge *dieses* Walks: ein
                        # History-Sprung ueber Walk-Grenzen hinweg machte die
                        # Faenge voneinander abhaengig.
                        u = self.dead_end.next_node(oracle, path, current, start)
                    elif self.restart_prob and oracle.rng.random() < self.restart_prob:
                        path.clear()
                        u = start
                    else:
                        u = nxt
                    path.append(u)
                trace.extend(current)
                current = []
        except BudgetExceeded:
            pass
        trace.extend(current)   # der abgebrochene Walk zaehlt mit
        return trace
