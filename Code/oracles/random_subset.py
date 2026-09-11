"""Oracle, dessen Zufallssprung nur eine zufaellige Teilmenge S von V trifft.

Zweck: die Sprungvarianten fuer externe Namenslisten (durwset-*, durwhist-*,
durwunion-*, siehe sampling.durw) auf *jedem* Graphen testen -- auch dort, wo es keine
Liste gibt (Slashdot0811, wiki-topcats haben numerische Knotennamen) --, und
auf den GPT-Graphen unabhaengig davon, *welche* Knoten eine echte Liste
trifft. Die echten Listen treffen eine gradverzerrte Teilmenge; hier ist S
gleichverteilt gezogen, und nur ihre relative Groesse ist einstellbar. Damit
lassen sich zwei Effekte trennen: "der Sprung erreicht nur einen Teil von V"
und "dieser Teil ist verzerrt".

    percent   |S| = round(percent/100 * |V|), mindestens 1

S steht fuer ein ganzes Experiment fest: gezogen aus dem Seed des Laufs
(run_experiment.py --seed), dem Graphennamen und sonst nichts. Alle
Wiederholungen eines Seeds sehen also *dasselbe* S -- wie ein Crawler, der
eine feste Liste besitzt --, ein neuer Seed zieht ein neues. Die View geht
nicht ein: directed und undirected teilen sich die Knoten-IDs und damit S,
der Vergleich bleibt gepaart (derselbe Grundsatz wie in experiment.runner).
Die Teilmengen verschiedener Anteile sind *geschachtelt*: S(10 %) ist ein
Praefix derselben Permutation wie S(50 %), steckt also darin.

Den Seed setzt experiment.runner vor dem Fork (set_experiment_seed); wer das
Oracle ausserhalb des Runners baut, bekommt config.DEFAULT_SEED.

Ein Sprung trifft immer (S ist eine Teilmenge von V, es gibt keine Nieten)
und kostet COST_RANDOM_NODE wie jede andere Knotenziehung. Die Einstiege
kommen ebenfalls aus S -- wie bei oracles.name_list: ein Crawler, der nur S
kennt, startet auch dort.

Kategorie: Vergleich. S gleichverteilt aus V zu ziehen setzt voraus, V zu
kennen; das Oracle simuliert eine Liste, es ist keine.

Mit percent = 100 ist S = V und der Sprung genau der gleichverteilte des
Papers -- die Gegenprobe: durwset-rand100 und durw-rand100 sind dann das
Original-DURW (bis auf den Einstieg, der aus V statt aus config.SEED_NODES
kommt).

Schnittstelle:
    set_experiment_seed(seed), experiment_seed() -> int
    subset_size(n, percent) -> int
    jump_set(graph, percent, seed) -> (ids, mask)
    class RandomSubsetOracle(CrawlOracle) -- random_node(), in_jump_set(),
        jump_multiplicity(), list_mass(), seed_nodes(), list_length(),
        list_entry()
"""

from __future__ import annotations

import zlib

import numpy as np

import config
from oracles.local_access import CrawlOracle

_SEED: int | None = None

# Eine Permutation je (Graph, Seed); die Teilmengen sind ihre Praefixe. Vor
# dem Fork gebaut (prepare), erben die Kindprozesse sie per Copy-on-Write.
_PERMS: dict[tuple[str, int, int], np.ndarray] = {}
# (ids, mask) je (Graph, Seed, Anteil) -- die Maske als bool-Array wie in
# oracles.name_list, der Test ist ein Array-Zugriff.
_SETS: dict[tuple[str, int, int, float], tuple[np.ndarray, np.ndarray]] = {}


def set_experiment_seed(seed: int) -> None:
    """Seed, aus dem S gezogen wird. experiment.runner ruft das vor dem Fork."""
    global _SEED
    _SEED = int(seed)


def experiment_seed() -> int:
    return config.DEFAULT_SEED if _SEED is None else _SEED


def subset_size(n: int, percent: float) -> int:
    return max(1, min(n, int(round(n * percent / 100))))


def _permutation(graph, seed: int) -> np.ndarray:
    key = (graph.name, graph.n_nodes, seed)
    perm = _PERMS.get(key)
    if perm is None:
        # Eigener Strom, unabhaengig vom random.Random der Laeufe: S darf
        # nicht an Estimator, Walk oder Laufnummer haengen.
        rng = np.random.default_rng(zlib.crc32(f"{seed}|{graph.name}".encode()))
        perm = rng.permutation(graph.n_nodes).astype(np.int64)
        _PERMS[key] = perm
    return perm


def jump_set(graph, percent: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(Knoten-IDs von S, bool-Maske ueber V) fuer diesen Anteil und Seed."""
    key = (graph.name, graph.n_nodes, seed, float(percent))
    out = _SETS.get(key)
    if out is None:
        ids = _permutation(graph, seed)[:subset_size(graph.n_nodes, percent)]
        mask = np.zeros(graph.n_nodes, dtype=bool)
        mask[ids] = True
        out = (ids, mask)
        _SETS[key] = out
    return out


class RandomSubsetOracle(CrawlOracle):
    """CrawlOracle plus Sprung gleichverteilt auf eine feste Zufallsteilmenge S."""

    @classmethod
    def prepare(cls, graph) -> None:
        # Nur die Permutation: der Anteil steckt in den partial-Keywords, die
        # der Runner abstreift (s. oracles.name_list.NameListOracle.prepare).
        # Die Masken daraus sind billig und entstehen je Prozess.
        _permutation(graph, experiment_seed())

    def __init__(self, *args, percent: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not 0 < percent <= 100:
            raise ValueError(f"percent muss in (0, 100] liegen, ist {percent}")
        self.percent = float(percent)
        self.subset_seed = experiment_seed()
        self._ids, self._mask = jump_set(self.graph, self.percent, self.subset_seed)
        self._k = len(self._ids)

    def random_node(self):
        u = int(self._ids[self.rng.randrange(self._k)])
        self._charge(u, self.cost_random_node)
        self.n_random_node += 1
        return u

    def list_length(self) -> int:
        return self._k

    def list_entry(self, i: int):
        """Position i der "Liste" S -- trifft immer (s. NameListOracle)."""
        u = int(self._ids[i])
        self._charge(u, self.cost_random_node)
        self.n_random_node += 1
        return u

    def in_jump_set(self, u) -> bool:
        return bool(self._mask[u])

    def jump_multiplicity(self, u) -> int:
        """1 auf S, 0 sonst -- jede "Listenposition" trifft genau einen Knoten."""
        return int(self._mask[u])

    def list_mass(self) -> int:
        return self._k

    def seed_nodes(self, k: int = 1) -> list:
        return [self.random_node() for _ in range(k)]
