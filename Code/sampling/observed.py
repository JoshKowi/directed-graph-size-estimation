"""Beobachtete Nachbarschaften -- der Kanal Sampler -> Formel fuer IE2.

Die Schaetzformeln (estimators.formulas) sehen normalerweise nur
`(Samples, Gewichte)`: der Graph wird dort nicht mehr befragt, es wird nur noch
gerechnet. IE2 (Induced Edges, zweite Variante) braucht mehr -- es zaehlt nicht
Knoten-Kollisionen `s_i == s_j`, sondern Kreuzkollisionen gegen

    A = Vereinigung der Nachbarschaften N(s') aller gezogenen Knoten.

Dieses Modul ist der Kanal dafuer und die gesamte Rechenmechanik dahinter. Die
Formeln selbst stehen in estimators.formulas (IE2SetEstimator,
IE2MultisetEstimator), die Herleitung der Schaetzer dort.

**A kostet kein Budget.** DurwSampler fragt jeden Knoten beim Erstbesuch
ohnehin ab (`oracle.neighbors(u)`) und baut daraus seinen ungerichteten Graphen
G_u. Beides wurde bezahlt und danach weggeworfen; hier wird es nur aufbewahrt.
Zwei Quellen, weil es zwei verschiedene Nachbarschaften sind:

    raw  -- die rohen Ausgangsnachbarn, genau die Antwort des Oracles. Naeher an
            N(s) im Paper und deutlich groesser.
    gu   -- die eingefrorene G_u-Nachbarschaft (`adj`). Konsistent mit dem
            Gewicht w(v) = deg_Gu(v), aber systematisch entleert: Kanten auf
            bereits besuchte Knoten verwirft G_u (sampling.durw), damit kein
            besuchter Knoten je seinen Grad aendert.

Welche der beiden besser ist, ist eine empirische Frage und deshalb eine Achse
(`a_source`), keine Festlegung.

Gemessen auf Slashdot0811 (4000 Queries) sind die beiden *Mengen* fast immer
identisch -- und das hat einen Grund: ein Knoten, den G_u aus `adj[u]`
verwirft, war bereits besucht, hat also einen Vorgaenger, und in dessen
`adj` steht er. Er ist damit ohnehin in A. Was sich unterscheidet, ist die
*Kantenmenge*: 294 928 gegen 294 534 beobachtete Kanten auf der gerichteten
Sicht. Das aendert nicht |A|, aber die Zeugenindizes und damit alles, was der
Margin ausschliesst -- und bei der Multiset-Form auch deg(s_j). Umgekehrt kann
`gu` auf einer *gerichteten* Sicht Knoten enthalten, die `raw` nicht hat (5 von
44 389 im Test): G_u symmetrisiert, ein Vorgaenger ohne eingehende Kante von
einem besuchten Knoten steht nur dort. Die Achse ist also kein Duplikat, aber
der Unterschied liegt nicht dort, wo man ihn zuerst vermutet.

**Speicher.** `raw` haelt die Antworten des Oracles als *Views* in die
CSR-Kantenliste (graphs.graph.neighbors gibt einen Slice zurueck) -- ein dict
mit einem Slice je verschiedenem besuchten Knoten, ohne eine einzige kopierte
Kante. `gu` muss kopieren, weil G_u eine andere Kantenmenge ist, tut das aber
als int32-Array statt als Python-Liste (4 statt 36 Byte je Kante).
`NeighborIndex` legt beim Bauen zwei dichte int32-Arrays ueber den beobachteten
Knoten-ID-Raum an (8 Byte je Knoten, auf dem groessten Graphen hier also
~144 MB) und gibt sie danach wieder frei; was bleibt, ist O(n + |A|).

Schnittstelle:
    class ObservedNeighborhoods
        .raw, .gu -- dict[int, np.ndarray]  (Knoten -> Nachbarn)
        .nbrs(source) -> dict
        .index(samples, source) -> NeighborIndex      (memoisiert je Sample-Set)
    class NeighborIndex
        .n, .a_size, .in_a, .deg
        .far_self(m)  -> np.ndarray[bool]   1{es gibt einen fernen Zeugen fuer s_i}
        .a_far(m)     -> np.ndarray[int]    |A_i^fern| je Sample
        .mult()       -> np.ndarray[int]    Vielfachheit von s_i in A (Multiset)
        .m_win(m)     -> np.ndarray[int]    davon im Fenster |j-i| <= m
        .d_win(m)     -> np.ndarray[int]    Summe deg(s_j) im Fenster
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Ein Knoten ohne aufgezeichnete Nachbarschaft -- kommt nur vor, wenn eine
# Stichprobe von einem Sampler stammt, der gar nichts mitschreibt; die Formeln
# weisen das vorher laut ab.
_EMPTY = np.empty(0, dtype=np.int32)

A_SOURCES = ("raw", "gu")


def _contains(sorted_keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Mengenzugehoerigkeit vektorisiert: `query in sorted_keys`, elementweise.

    np.isin waere dasselbe, baut aber je Aufruf neu -- hier wird gegen *ein*
    sortiertes Array sehr oft gefragt (einmal je Fensterabstand d), deshalb
    direkt searchsorted.
    """
    if sorted_keys.size == 0 or query.size == 0:
        return np.zeros(query.size, dtype=bool)
    pos = np.searchsorted(sorted_keys, query)
    np.clip(pos, 0, sorted_keys.size - 1, out=pos)
    return sorted_keys[pos] == query


class NeighborIndex:
    """Zeugen-Index einer Trajektorie: wer liegt in A, und ab/bis wann.

    "Zeuge" von v ist ein Sample-Index j mit v in N(s_j). Fuer die
    SafetyMargin-Rechnung wird staendig gefragt: *gibt es* einen Zeugen von v
    ausserhalb des Fensters [i-m, i+m]? Dafuer genuegen der kleinste und der
    groesste Zeugenindex:

        alle Zeugen von v liegen in [minw[v], maxw[v]]

    also gibt es einen Zeugen ausserhalb des Fensters **genau dann**, wenn
    minw[v] < i-m oder maxw[v] > i+m. Kein Suchen, kein Naehern -- und der
    Grund, warum die Set-Form ohne Doppelsumme auskommt.

    Gebaut wird alles in einem Durchlauf ueber die *verschiedenen* gesampelten
    Knoten. Das ist dieselbe Groessenordnung (Summe der Grade der verschiedenen
    besuchten Knoten), die der Sampler beim Aufbau von G_u ohnehin schon einmal
    bezahlt hat -- nicht die Summe ueber die Trajektorie mit Vielfachheit.

    Die Indizes sind Positionen *innerhalb des uebergebenen Sample-Sets*, nicht
    innerhalb der Gesamttrajektorie. Mit einem Thinning, das die Trajektorie
    zerschneidet, bezieht sich der Margin also auf Abstaende im Set -- genau
    wie bei estimators.formulas._collisions.
    """

    def __init__(self, samples, nbrs: dict) -> None:
        n = len(samples)
        self.n = n
        self.nodes = np.fromiter((s.node for s in samples), dtype=np.int64,
                                 count=n)
        if n == 0:
            self.a_size = 0
            self.in_a = np.zeros(0, dtype=bool)
            self.deg = np.zeros(0, dtype=np.int64)
            self._minw_s = self._maxw_s = np.zeros(0, dtype=np.int64)
            self._am = self._aM = np.zeros(0, dtype=np.int64)
            self._pos = np.zeros(0, dtype=np.int64)
            self._mult = None
            self._keys = None
            return

        # --- verschiedene gesampelte Knoten, mit erstem/letztem Auftreten ----
        # Stabil sortiert stehen die Originalindizes innerhalb einer Gruppe
        # aufsteigend: der erste ist das kleinste, der letzte das groesste
        # Auftreten. Das ersetzt zwei Durchlaeufe mit dicts.
        order = np.argsort(self.nodes, kind="stable")
        sn = self.nodes[order]
        starts = np.flatnonzero(np.r_[True, sn[1:] != sn[:-1]])
        ends = np.r_[starts[1:], n]
        self.uniq = sn[starts]                       # sortiert
        first = order[starts].astype(np.int64)
        last = order[ends - 1].astype(np.int64)
        self.counts = (ends - starts).astype(np.int64)
        self._pos = np.searchsorted(self.uniq, self.nodes)

        missing = [int(u) for u in self.uniq if int(u) not in nbrs]
        if missing:
            raise ValueError(
                f"{len(missing)} gesampelte Knoten haben keine aufgezeichnete "
                f"Nachbarschaft (z. B. {missing[:3]}). Die Stichprobe kommt von "
                "einem Sampler, der ohne collect_nbrs=True gelaufen ist -- IE2 "
                "braucht die Nachbarlisten (s. sampling.observed)."
            )
        nb = [np.asarray(nbrs[int(u)], dtype=np.int64) for u in self.uniq]
        self.deg = np.array([a.size for a in nb], dtype=np.int64)[self._pos]

        # --- minw/maxw ueber dichte Arrays -----------------------------------
        # Beide Werte sind ein Minimum bzw. Maximum ueber alle Zeugen. Statt
        # np.minimum.at (ungepuffert und langsam) wird die *Reihenfolge*
        # ausgenutzt: schreibt man die Knoten nach `last` aufsteigend, gewinnt
        # am Ende der groesste Wert; nach `first` absteigend der kleinste.
        # Beides ist eine gepufferte Fancy-Index-Zuweisung und damit schnell.
        nmax = int(max(self.uniq[-1],
                       max((int(a.max()) for a in nb if a.size), default=0))) + 1
        minw = np.full(nmax, n, dtype=np.int32)      # n = "kein Zeuge"
        maxw = np.full(nmax, -1, dtype=np.int32)
        for i in np.argsort(last, kind="stable"):
            maxw[nb[i]] = last[i]
        for i in np.argsort(first, kind="stable")[::-1]:
            minw[nb[i]] = first[i]

        in_a_all = maxw >= 0
        self.a_size = int(np.count_nonzero(in_a_all))
        # Was danach noch gebraucht wird, ist O(n + |A|) -- die dichten Arrays
        # duerfen weg, sobald daraus gelesen wurde.
        self.in_a = in_a_all[self.nodes]
        self._minw_s = minw[self.nodes].astype(np.int64)
        self._maxw_s = maxw[self.nodes].astype(np.int64)
        self._am = minw[in_a_all].astype(np.int64)
        self._aM = maxw[in_a_all].astype(np.int64)
        del minw, maxw, in_a_all

        self._nb = nb
        self._mult = None
        self._keys = None

    # ------------------------------------------------------------------ Set
    def far_self(self, margin: int) -> np.ndarray:
        """1{es gibt einen Zeugen von s_i ausserhalb [i-m, i+m]}, je Sample.

        Exakt, s. Klassen-Docstring. Knoten ausserhalb von A tragen die
        Sentinels (minw = n, maxw = -1) und fallen damit von selbst heraus.
        """
        i = np.arange(self.n, dtype=np.int64)
        m = int(margin)
        return (self._minw_s < i - m) | (self._maxw_s > i + m)

    def a_far(self, margin: int) -> np.ndarray:
        """|A_i^fern| je Sample: A ohne die Knoten, deren Zeugen *alle* im
        Fenster [i-m, i+m] liegen.

        Ein Knoten v ist fuer i ausgeschlossen, wenn minw[v] >= i-m und
        maxw[v] <= i+m. Beide Bedingungen sind in i monoton, aber
        gegenlaeufig -- zusammen ergeben sie ein *Intervall* von i, fuer das v
        ausgeschlossen ist:

            ausgeschlossen fuer  max(maxw-m, 0) <= i < minw+m+1

        Damit wird aus n Bereichsabfragen eine Differenzenreihe: Intervallenden
        einsammeln, kumulativ summieren. O(n + |A|), ohne Fenwick-Baum und ohne
        jede Schleife.
        """
        n, m = self.n, int(margin)
        if self.a_size == 0:
            return np.zeros(n, dtype=np.int64)
        ins = np.clip(self._aM - m, 0, n)
        out = np.clip(self._am + m + 1, 0, n)
        keep = ins < out                      # leere Intervalle weglassen
        diff = (np.bincount(ins[keep], minlength=n + 1).astype(np.int64)
                - np.bincount(out[keep], minlength=n + 1))
        near = np.cumsum(diff)[:n]
        return self.a_size - near

    # ------------------------------------------------------------- Multiset
    def mult(self) -> np.ndarray:
        """Vielfachheit von s_i im Multiset A: #{j : s_i in N(s_j)}.

        Gebraucht wird das nur fuer *gesampelte* Knoten -- deshalb werden die
        Paare (u, v) von vornherein auf gesampelte v eingeschraenkt. Das ist
        der Grund, warum hier kein dichtes Array ueber ganz V noetig ist: die
        Zahl der Kanten zwischen gesampelten Knoten ist um Groessenordnungen
        kleiner als die Summe aller beobachteten Grade.
        """
        if self._mult is None:
            self._build_pairs()
        return self._mult[self._pos]

    def m_win(self, margin: int) -> np.ndarray:
        """Von `mult()` der Anteil mit |j - i| <= m.

        j = i traegt nie bei (kein Knoten ist sein eigener Nachbar, Schlingen
        entfernt graphs.graph._simplify beim Laden), wird aber mitgezaehlt,
        damit das Fenster hier dasselbe ist wie in `d_win` -- Zaehler und
        Nenner muessen dieselben Paare auslassen.

        Je Abstand d eine vektorisierte Mengenabfrage ueber alle Samples: O(m)
        numpy-Aufrufe, nicht O(n*m) Python-Schritte. Der Aufwand waechst
        trotzdem linear in m -- fuer grosse m ist die Set-Form die richtige
        Wahl, die kennt kein m im Aufwand.
        """
        if self._keys is None:
            self._build_pairs()
        n, m = self.n, int(margin)
        out = np.zeros(n, dtype=np.int64)
        p, u = self._pos, self.uniq.size
        for d in range(1, min(m, n - 1) + 1):
            # j = i - d: Zeuge ist s_{i-d}, Ziel s_i
            out[d:] += _contains(self._keys, p[:-d] * u + p[d:])
            # j = i + d
            out[:-d] += _contains(self._keys, p[d:] * u + p[:-d])
        return out

    def d_win(self, margin: int) -> np.ndarray:
        """Summe der Grade im Fenster |j - i| <= m (inklusive j = i)."""
        n, m = self.n, int(margin)
        cs = np.r_[0, np.cumsum(self.deg)]
        i = np.arange(n, dtype=np.int64)
        return cs[np.clip(i + m + 1, 0, n)] - cs[np.clip(i - m, 0, n)]

    def _build_pairs(self) -> None:
        """Kanten (u, v) mit v gesampelt -- in Indizes von `uniq`.

        Daraus kommen beide Multiset-Groessen: die Vielfachheit (Summe der
        Besuchszahlen der Zeugen) und die Schluesselmenge fuer die
        Fensterabfragen. Der Schluessel u*|uniq| + v ist eindeutig, weil beide
        Indizes kleiner als |uniq| sind.
        """
        u = self.uniq.size
        keys, mult = [], np.zeros(u, dtype=np.int64)
        for i, a in enumerate(self._nb):
            if a.size == 0:
                continue
            pos = np.searchsorted(self.uniq, a)
            np.clip(pos, 0, u - 1, out=pos)
            hit = self.uniq[pos] == a
            if not hit.any():
                continue
            pv = pos[hit]
            mult[pv] += self.counts[i]
            keys.append(np.full(pv.size, i, dtype=np.int64) * u + pv)
        self._mult = mult
        self._keys = (np.sort(np.concatenate(keys)) if keys
                      else np.zeros(0, dtype=np.int64))


@dataclass
class ObservedNeighborhoods:
    """Was der Sampler unterwegs gesehen hat, aufbewahrt fuer IE2.

    Je verschiedenem besuchtem Knoten eine Nachbarliste, zwei Quellen (s.
    Modul-Docstring). Der Index wird je Sample-Set memoisiert: in einer
    geteilten Walk-Gruppe (pipeline.estimate_group) werten mehrere IE2-Formeln
    dieselbe Trajektorie aus, und der Index ist das Teuerste daran.
    """

    raw: dict
    gu: dict
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def nbrs(self, source: str) -> dict:
        if source not in A_SOURCES:
            raise ValueError(f"a_source muss aus {A_SOURCES} sein, ist {source!r}")
        return self.raw if source == "raw" else self.gu

    def index(self, samples, source: str) -> NeighborIndex:
        # Schluessel ist die Identitaet des Sample-Sets. Die starke Referenz im
        # Cache haelt das Objekt am Leben, damit id() nicht recycelt werden
        # kann -- ohne sie waere ein Treffer nicht verlaesslich derselbe.
        key = (source, id(samples))
        hit = self._cache.get(key)
        if hit is not None and hit[0] is samples:
            return hit[1]
        idx = NeighborIndex(samples, self.nbrs(source))
        self._cache[key] = (samples, idx)
        return idx
