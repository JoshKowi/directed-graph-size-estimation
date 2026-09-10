"""Oracle, das seine Knoten aus einer externen Namensliste zieht.

Der Zufallssprung von DURW und das gleichverteilte Ziehen von UniformNodeOracle
setzen beide Kenntnis der Knotenmenge V voraus -- genau das, was geschaetzt
werden soll. Dieses Oracle ersetzt die Kenntnis durch *externes Wissen*: eine
Liste plausibler Namen (Wikipedia-Titel, Top-Wikidata-Entitaeten, siehe
namelists). Gezogen wird eine Position der Liste; steht der Name im Graphen,
ist der Knoten gefunden, sonst kostet der Fehlschlag und es geht weiter. Das
ist Rejection Sampling mit einem Rahmen, den man wirklich hat.

Damit ist ein darauf gebautes Verfahren real umsetzbar -- die Kategorie wird
aber wie immer erst in estimators/__init__.py vergeben.

**Was das Oracle *nicht* leistet.** Die Liste deckt den Graphen nicht ab: fuer
gpt4_io gegen enwiki sind es 37,9 % der Knoten, fuer gpt4o_io 22,7 % (gemessen
mit title_overlap). Die Ziehung ist also gleichverteilt auf der *Trefferteilmenge*,
nicht auf V, und diese Teilmenge ist nicht zufaellig: getroffene Knoten haben
deutlich hoeheren Grad als verfehlte (gpt4o_io/enwiki 10,70 gegen 3,55; unter
den verfehlten sind 60,5 % Sackgassen, unter den getroffenen 29,1 %). Wer das
als gleichverteilte Ziehung aus V behandelt, schaetzt die Groesse der
Teilmenge, nicht |V| -- derselbe Fallstrick, den
oracles.global_access.DegWeightedIndependentOracle im Docstring beschreibt.

`burn_in` ist die Gegenmassnahme: nach einem Treffer erst n Schritte ueber
Ausgangskanten laufen, dann den Knoten liefern. Das fuehrt von der Verzerrung
der Liste weg -- und die eines Random-Walk-Schritts ein. Wo das Optimum liegt,
ist eine empirische Frage, deshalb ist n einstellbar und 0 der Default.

Kosten:
    Treffer      COST_RANDOM_NODE   (wie jede andere Knotenziehung)
    Fehlschlag   COST_DRAW_MISS     (einstellbar, muss > 0 sein)
    Burn-in      COST_NEIGHBORS bzw. COST_CACHE_HIT je Schritt, wie ueberall

Der Index (Listenposition -> Knoten-ID oder -1) wird von build_name_index.py
vorab gebaut; zur Laufzeit wird kein einziger String angefasst.

Schnittstelle:
    load_index(graph_name, source) -> np.ndarray
    jump_set_mask(graph, source, limit) -> np.ndarray
    index_meta(graph_name, source) -> dict
    available(graph_name) -> list[str]
    class NameListOracle(Oracle)  -- random_node(), neighbors(), degree(),
                                     seed_nodes()
"""

from __future__ import annotations

import json

import numpy as np

import config
from oracles.base import Oracle

MISS = -1

# Modulweiter Cache: derselbe Index wird von allen Estimators eines Laufs
# benutzt. experiment.runner waermt ihn vor dem Fork vor, danach erben die
# Kindprozesse das Array per Copy-on-Write, statt es je Prozess neu zu laden
# (77 MB bei enwiki -- achtmal waeren es 600 MB fuer nichts).
_CACHE: dict[tuple[str, str], np.ndarray] = {}

# Zugehoerigkeitsmasken je (Graph, Quelle, Laenge). Als bool-Array ueber alle
# Knoten statt als Menge von IDs: bei gpt4_io sind das 6,5 MB gegen ein
# Vielfaches fuer ein Python-set, und der Test ist ein Array-Zugriff.
_MASKS: dict[tuple[str, str, int | None], np.ndarray] = {}


def _path(graph_name: str, source: str):
    return config.NAME_INDEX_DIR / f"{graph_name}__{source}.npy"


def load_index(graph_name: str, source: str) -> np.ndarray:
    key = (graph_name, source)
    arr = _CACHE.get(key)
    if arr is None:
        p = _path(graph_name, source)
        if not p.exists():
            raise FileNotFoundError(
                f"Kein Namensindex fuer Graph {graph_name!r} und Quelle "
                f"{source!r} unter {p}. Erst bauen:\n"
                f"    python build_name_index.py --graphs {graph_name} "
                f"--sources {source}"
            )
        arr = np.load(p, mmap_mode=None)
        _CACHE[key] = arr
    return arr


def jump_set_mask(graph, source: str, limit: int | None = None) -> np.ndarray:
    """bool-Maske ueber alle Knoten: erreicht der Sprung diesen Knoten?

    Aus dem *geschnittenen* Index gebaut -- eine kuerzere Liste erreicht
    weniger Knoten, die Maske haengt also an `limit`.
    """
    key = (graph.name, source, limit)
    mask = _MASKS.get(key)
    if mask is None:
        idx = load_index(graph.name, source)
        if limit is not None:
            idx = idx[:limit]
        mask = np.zeros(graph.n_nodes, dtype=bool)
        hit = idx[idx != MISS]
        if hit.size:
            mask[hit] = True
        _MASKS[key] = mask
    return mask


def index_meta(graph_name: str, source: str) -> dict:
    p = _path(graph_name, source).with_suffix(".json")
    return json.loads(p.read_text()) if p.exists() else {}


def available(graph_name: str) -> list[str]:
    """Quellen, fuer die ein Index dieses Graphen auf der Platte liegt."""
    if not config.NAME_INDEX_DIR.exists():
        return []
    pre = f"{graph_name}__"
    return sorted(f.stem[len(pre):] for f in config.NAME_INDEX_DIR.glob(f"{pre}*.npy"))


class NameListOracle(Oracle):
    """Ziehung aus einer externen Namensliste, mit Rejection und Burn-in.

    `limit` nimmt nur die ersten n Eintraege der Liste. Sinnvoll, weil die
    Quellen nach Relevanz sortiert sind (top-q nach QRank, die In-Grad-Listen
    nach Eingangsgrad): ein Praefix ist dort "die n prominentesten Entitaeten".
    Der Index ist positionsbasiert, das Abschneiden also ein Slice -- es braucht
    je (Graph, Quelle) trotzdem nur *einen* Index, nicht einen je Laenge.

    Laengere Liste heisst mehr erreichbare Knoten *und* mehr Fehlschlaege; wo
    der Handel kippt, zeigen die Kurven aus plotting/name_coverage.py.
    """

    @classmethod
    def prepare(cls, graph) -> None:
        """Indizes vor dem Fork laden, damit die Kindprozesse sie per
        Copy-on-Write erben statt jeder fuer sich (enwiki: 77 MB, achtmal
        waeren es 600 MB fuer nichts).

        Der Hook bekommt nur den Graphen, nicht die Quelle -- die steckt in den
        partial-Keywords, die experiment.runner beim Aufruf abstreift. Deshalb
        wird geladen, was fuer diesen Graphen *vorhanden* ist. Wer alle drei
        Indizes gebaut hat und nur einen benutzt, zahlt einmalig ein paar
        hundert MB im Elternprozess; das ist billiger als die Alternative.
        """
        for source in available(graph.name):
            load_index(graph.name, source)
            # Die volle Maske (limit=None); geschnittene Varianten entstehen
            # beim ersten Zugriff und sind dann klein genug, um sie je Prozess
            # zu bauen.
            jump_set_mask(graph, source)

    def __init__(self, *args, source: str, limit: int | None = None,
                 burn_in: int = config.DEFAULT_DRAW_BURN_IN,
                 cost_miss: float = config.COST_DRAW_MISS, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if cost_miss <= 0:
            # Ohne Preis dreht die Ziehschleife auf einem Index ohne Treffer
            # endlos -- dasselbe Argument, mit dem oracles.base einen Preis > 0
            # fuer den Cache-Treffer verlangt.
            raise ValueError(
                f"cost_miss muss > 0 sein, ist {cost_miss}: sonst terminiert "
                "die Ziehschleife nicht, wenn die Liste den Graphen nicht trifft."
            )
        self.source = source
        self.burn_in = int(burn_in)
        self.cost_miss = float(cost_miss)
        self.n_draw_miss = 0
        self._index = load_index(self.graph.name, source)
        if limit is not None:
            if limit > len(self._index):
                raise ValueError(
                    f"limit={limit:,} ist groesser als der Index "
                    f"{self.graph.name}/{source} mit {len(self._index):,} "
                    "Eintraegen -- laut statt still die volle Liste zu nehmen, "
                    "sonst waere ein Lauf mit zu grossem limit nicht von einem "
                    "ohne limit zu unterscheiden.".replace(",", " ")
                )
            # Ein Slice auf einem numpy-Array ist ein *View*: kein zusaetzlicher
            # Speicher, und das gecachte Array bleibt das, was prepare() vor dem
            # Fork geladen hat (Copy-on-Write bleibt heil).
            self._index = self._index[:limit]
        self.limit = limit
        self._n = len(self._index)
        self._mask = jump_set_mask(self.graph, source, limit)
        if self._n == 0 or not bool((self._index != MISS).any()):
            # Passiert z.B. bei Graphen mit numerischen Knotennamen
            # (Slashdot0811, wiki-topcats): dort gibt es nichts abzugleichen.
            # Ohne diese Meldung wuerde der Lauf sein Budget still in
            # Fehlschlaegen verbrennen und eine leere Stichprobe liefern.
            raise ValueError(
                f"Der Namensindex {self.graph.name}/{source} enthaelt keinen "
                "einzigen Treffer -- aus dieser Liste laesst sich kein Knoten "
                "ziehen. Hat der Graph ueberhaupt Namen als Knoten?"
            )

    # -- Zugriffe ---------------------------------------------------------
    def random_node(self):
        """Ein Knoten aus der Liste. Fehlschlaege kosten und werden gezaehlt."""
        while True:
            u = int(self._index[self.rng.randrange(self._n)])
            if u != MISS:
                self._charge(u, self.cost_random_node)
                self.n_random_node += 1
                break
            # Kein Knoten, also auch kein Besuch: node=None, damit der
            # Fehlschlag nicht in die Besuchszaehler laeuft.
            self._charge(None, self.cost_miss)
            self.n_draw_miss += 1

        for _ in range(self.burn_in):
            nbrs = self._fetch(u)
            if not len(nbrs):
                break       # Sackgasse: hier endet der Burn-in, der Knoten zaehlt
            u = int(nbrs[self.rng.randrange(len(nbrs))])
        return u

    def neighbors(self, u) -> tuple:
        return self._fetch(u)

    def degree(self, u) -> int:
        return len(self._fetch(u))

    def in_jump_set(self, u) -> bool:
        """Liegt der Knoten in der Trefferteilmenge S?

        Anders als beim gleichverteilten Sprung ist das hier *nicht* fuer alle
        Knoten wahr: die Liste deckt den Graphen nur zu 6 bis 22 % ab. Knoten
        ausserhalb von S sind allein ueber Kanten erreichbar und haben deshalb
        eine andere Stationaerwahrscheinlichkeit -- s.
        weighting.DurwJumpSetWeighting.
        """
        return bool(self._mask[u])

    def seed_nodes(self, k: int = 1) -> list:
        """Einstiege kommen aus derselben Liste -- ein Crawler, der nur sie
        hat, startet auch dort."""
        return [self.random_node() for _ in range(k)]

    # -- Buchhaltung ------------------------------------------------------
    def cost(self) -> dict[str, int]:
        c = super().cost()
        c["n_draw_miss"] = self.n_draw_miss
        return c
