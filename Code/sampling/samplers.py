"""Konkrete Sampler.

Schnittstelle:
    class UniformSampler(Sampler)     -- braucht oracle.random_node()
    class RandomWalkSampler(Sampler)  -- braucht oracle.seed_nodes()/neighbors()
"""

from __future__ import annotations

from oracles.base import BudgetExceeded
from sampling.base import Sample, Sampler
from sampling.dead_ends import DeadEndStrategy, RestartToStart


class UniformSampler(Sampler):
    """Unabhaengige, gleichverteilte Knoten (nur mit globalem Oracle moeglich)."""

    name = "uniform"

    def sample(self, oracle) -> list[Sample]:
        samples: list[Sample] = []
        try:
            while True:
                u = oracle.random_node()
                samples.append(Sample(u, oracle.degree(u), len(samples)))
        except BudgetExceeded:
            return samples


class RandomWalkSampler(Sampler):
    """Random Walk ueber Nachbarschaftsabfragen; liefert die volle Trajektorie.

    Das Aufteilen der Trajektorie in Sample-Sets uebernimmt sampling.thinning.

    `dead_end` bestimmt, wie es bei einem Knoten ohne ausgehende Kanten
    weitergeht (siehe sampling.dead_ends). `restart_prob` ist davon unabhaengig
    und teleportiert zusaetzlich mit fester Wahrscheinlichkeit zum Anfang.

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
        restart_prob: float = 0.0,
        burn_in: int = 0,
        allow_self_loops: bool = False,
    ) -> None:
        self.dead_end = dead_end or RestartToStart()
        self.n_seeds = n_seeds
        self.restart_prob = restart_prob
        self.burn_in = burn_in
        self.allow_self_loops = allow_self_loops
        self.name = f"random_walk_{self.dead_end.name}"

    def _step(self, u, nbrs, rng):
        """Zufaelliger Nachbar != u, oder None wenn es keinen gibt.

        Der Normalfall kostet O(1); die gefilterte Liste wird nur gebaut, wenn
        die Selbstkante tatsaechlich gezogen wurde (Wahrscheinlichkeit 1/Grad).
        """
        v = nbrs[rng.randrange(len(nbrs))]
        if self.allow_self_loops or v != u:
            return v
        others = [x for x in nbrs if x != u]
        return others[rng.randrange(len(others))] if others else None

    def sample(self, oracle) -> list[Sample]:
        trace: list[Sample] = []
        try:
            seeds = oracle.seed_nodes(self.n_seeds)
            start = seeds[0]
            u = start
            path = [u]
            step = 0
            while True:
                nbrs = oracle.neighbors(u)
                if step >= self.burn_in:
                    trace.append(Sample(u, len(nbrs), step))
                step += 1

                nxt = self._step(u, nbrs, oracle.rng) if nbrs else None
                if nxt is None:
                    u = self.dead_end.next_node(oracle, path, trace, start)
                elif self.restart_prob and oracle.rng.random() < self.restart_prob:
                    path.clear()
                    u = start
                else:
                    u = nxt
                path.append(u)
        except BudgetExceeded:
            return trace
