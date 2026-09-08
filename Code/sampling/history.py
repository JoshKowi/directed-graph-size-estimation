"""Gewichtete Besuchshistorie mu_t -- die Umverteilung von NMMC.

Bei NMMC (sampling.nmmc) wird ein abgelehnter Zug nicht wiederholt, sondern die
Kette wird absorbiert und sofort auf einen *bereits besuchten* Knoten
umverteilt. Gezogen wird aus der gewichteten Besuchsfolge

    mu_t = sum_k eta_t(k) delta_{Z_k},   eta_t(k) = w_k / sum_{k'<=t} w_{k'}

mit w_k = k^alpha. Genau diese Historienabhaengigkeit macht den Prozess
nicht-markovsch und laesst mu_t gegen die quasi-stationaere Verteilung
konvergieren (Benaim/Cloez).

**Ein Slot je verschiedenem Knoten, nicht je Schritt.** Das ist exakt, keine
Zusammenfassung: mu_t ist ein Mass auf Knoten, also

    P(X = v) = (sum_{k: Z_k = v} w_k) / sum_k w_k

-- es haengt nur an der Gewichtssumme je Knoten. Es ist zugleich das einzig
Moegliche: eine Umverteilung landet immer auf einem schon geholten Knoten und
kostet damit nur COST_CACHE_HIT, ein Lauf macht also bis zu 50 Schritte je
Budget-Einheit (config.COST_CACHE_HIT). *Verschiedene* Knoten kann er dagegen
hoechstens budget/COST_NEIGHBORS viele sehen, weil ein Erstbesuch den vollen
Preis kostet. Auf gpt-4o-io bei Budget 10 % sind das rund 19 Mio. Schritte
gegen 0,19 Mio. Slots -- die Trajektorie zu halten waere hundertmal so teuer.

Datenstruktur ist ein Fenwick-Baum (Binary Indexed Tree): add() und draw() in
O(log m). Ein dict Knoten -> Slot haelt die Zuordnung, ein Python-`list` die
Teilsummen -- bewusst keine numpy-Array, weil skalares numpy-Indexing in einer
Python-Schleife am Boxing rund dreimal so lange braucht.

Zahlenbereich: bei 46 Mio. Schritten und alpha = 10 ist sum_k k^10 rund 10^83,
float64 traegt bis 1.8e308. Dass fruehe Knoten dabei in den Teilsummen
untergehen, ist kein Rundungsfehler, sondern die Aussage: ihre
Wahrscheinlichkeit *ist* 10^-76. alpha ist eine Gedaechtnislaenge, kein
numerisches Risiko.

Schnittstelle:
    class WeightedHistory
        .add(node, w)      -- Besuch mit Gewicht w verbuchen
        .draw(rng) -> node -- Knoten ~ mu_t, genau ein rng-Aufruf
        .total, len()      -- Gesamtgewicht, Zahl verschiedener Knoten
"""

from __future__ import annotations


class WeightedHistory:
    """Ziehen aus mu_t in O(log m) ueber einen Fenwick-Baum."""

    __slots__ = ("_slot", "_nodes", "_tree", "_cap", "total")

    def __init__(self) -> None:
        self._slot: dict = {}        # Knoten -> 0-basierter Slot
        self._nodes: list = []       # Slot -> Knoten
        self._tree: list[float] = [0.0]   # 1-basiert, [0] ungenutzt
        self._cap = 0                # Kapazitaet, immer Zweierpotenz
        self.total = 0.0

    def __len__(self) -> int:
        return len(self._nodes)

    def add(self, node, w: float) -> None:
        """Einen Besuch verbuchen. Bekannte Knoten sammeln nur Gewicht an."""
        i = self._slot.get(node)
        if i is None:
            i = len(self._nodes)
            self._slot[node] = i
            self._nodes.append(node)
            if i + 1 > self._cap:
                new_cap = self._cap * 2 if self._cap else 16
                self._tree.extend([0.0] * (new_cap - self._cap))
                # Nur der neue Wurzelknoten deckt (0, new_cap] ab und erbt
                # deshalb die bisherige Gesamtsumme. Jedes andere neue j hat
                # lsb(j) < cap, sein Bereich liegt komplett oberhalb der bisher
                # belegten Slots und ist noch leer.
                self._tree[new_cap] = self.total
                self._cap = new_cap
        self.total += w
        tree, cap = self._tree, self._cap
        j = i + 1
        while j <= cap:   # bis zur Kapazitaet, nicht bis zur Belegung
            tree[j] += w
            j += j & -j

    def draw(self, rng):
        """Knoten ~ mu_t. Verbraucht genau einen rng.random()-Aufruf."""
        r = rng.random() * self.total
        tree = self._tree
        idx, mask = 0, self._cap
        while mask:
            nxt = idx + mask
            if tree[nxt] <= r:
                r -= tree[nxt]
                idx = nxt
            mask >>= 1
        # idx == len(self._nodes) kann nur durch Rundung entstehen (r liegt
        # dann numerisch genau auf der Gesamtsumme); der letzte Slot ist dann
        # die richtige Antwort.
        size = len(self._nodes)
        return self._nodes[idx if idx < size else size - 1]
