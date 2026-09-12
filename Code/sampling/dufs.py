"""DUFS -- Directed Unbiased Frontier Sampling (Murai, Ribeiro, Towsley).

DUFS ist DURW mit k Walkern statt einem. Alles, was DURW auf den gerichteten
Views ueberhaupt erst rechtfertigt, bleibt unveraendert:

  * derselbe waehrend des Laufs aufgebaute ungerichtete Graph G_u, mit
    derselben Regel "beim Erstbesuch einfrieren" (sampling.durw, Abschnitt 1),
  * dieselbe Absprungregel w/(w + deg_Gu(v)),
  * dieselbe Stationaerverteilung pi(v) ~ w + deg_Gu(v) und damit dasselbe
    Gewicht 1/(w + deg_Gu) aus weighting.DurwWeighting.

Der einzige Unterschied ist, *welcher* Knoten in jedem Schritt an der Reihe
ist: statt der einen aktuellen Position gibt es k davon (die Liste L), und
gezogen wird eine davon mit Wahrscheinlichkeit proportional zu ihrem
Sprunggewicht w + deg_Gu. Genau diese gradproportionale Auswahl ist der Grund,
warum die Folge der emittierten Knoten weiterhin pi-verteilt ist -- das ist die
Frontier-Sampling-Aussage von Ribeiro & Towsley (2010), hier auf den
DURW-Graphen G_u samt Sprungkante uebertragen.

**Wozu das gut ist.** Nicht die Verteilung aendert sich, sondern die
Autokorrelation. Ein einzelner Walker verfaengt sich in einer Region; seine
Kollisionen sind dann lokal statt global, und der Kollisionsschaetzer
(estimators.formulas.WISCollisionEstimatorKatzir) unterschaetzt |V|, weil er
Kollisionen als Hinweis auf eine kleine Knotenmenge liest. Bei DUFS stammen
aufeinanderfolgende Samples aus k verschiedenen Regionen. Der Safety Margin
(`margin`) greift deshalb bei DUFS anders als bei DURW: er streicht Paare mit
kleinem Abstand *in der Sample-Folge*, und diese Paare stammen bei k > 1
ueberwiegend gar nicht mehr vom selben Walker. Er schadet nicht, aber er
korrigiert immer weniger, je groesser k wird.

**k = 1 ist DURW.** Und zwar bitgleich, nicht nur der Verteilung nach: bei
einem einzigen Walker entfaellt die Auswahl, es wird dafuer auch kein
Zufallswert verbraucht, und die Reihenfolge der Buchungen bleibt dieselbe (s.
"Wann die Nachbarn geholt werden"). check_dufs.py prueft das.

**Wann die Nachbarn geholt werden.** DURW fragt die Nachbarn der *aktuellen*
Position am Schleifenkopf ab. DUFS fragt sie stattdessen bei der *Ankunft* ab,
also unmittelbar nachdem ein Walker gezogen oder gesprungen ist. Das ist kein
kosmetischer Unterschied, sondern noetig: das Auswahlgewicht eines Walkers ist
w + deg_Gu seiner Position, und deg_Gu steht erst fest, wenn die
Nachbarschaft geholt und in G_u eingefroren ist. Wuerde erst bei der Auswahl
gefragt, muesste der Baum mit einem geschaetzten Gewicht arbeiten und die
Auswahl waere verzerrt.

Fuer die Buchhaltung aendert sich dadurch nichts. Je Iteration wird genau eine
Nachbarabfrage bezahlt, nur eben am Ende statt am Anfang -- die Abfrage, die
DURW am Kopf von Iteration n macht, macht DUFS am Fuss von Iteration n-1. Die
Seeds werden vor der Schleife angefragt, genau wie DURW seinen Seed in
Iteration 0 anfragt. Die Folge aus Buchungen und oracle.mark() ist damit
identisch, und nur deshalb ist die k=1-Gleichheit bitgenau.

**Die k Startknoten** kommen aus oracle.seed_nodes(k) und damit aus derselben
Quelle wie die Spruenge: bei NameListOracle aus der Namensliste, bei
RandomSubsetOracle aus der Teilmenge S, bei JumpCrawlOracle gleichverteilt aus
V. Das ist der Grund, warum DUFS ohne Sprung (`no_jumps`) *nicht* mit dem
blossen CrawlOracle auskommt, DURW ohne Sprung aber schon: k ueber eine Liste
verteilte Startpunkte sind der eigentliche Inhalt dieser Variante.

**`no_jumps` (w -> 0).** Reiner Random Walk auf G_u, von k Startknoten aus.
Anders als bei DURW ist das hier keine Randnotiz: die dortige Einschraenkung
"pi ~ deg_Gu gilt nur in der schwach zusammenhaengenden Komponente des Seeds"
faellt mit k Startknoten aus einer Liste deutlich milder aus, weil viele
Komponenten getroffen werden. Ein Walker auf einem Knoten mit deg_Gu = 0
(nur der Startknoten kann das sein, s. sampling.durw) bekommt Gewicht 0 und
wird nie wieder gezogen; der Fang laeuft mit den uebrigen weiter, statt wie
bei DURW zu enden. Sind alle Walker tot, ist der Fang zu Ende.

Was hier -- wie bei DURW -- *nicht* vorkommt: eine Sackgassen-Strategie (mit
Sprung wird eine Sackgasse nie absorbierend) und `history_jumps`.

Schnittstelle:
    class DufsSampler(Sampler)  -- braucht oracle.seed_nodes()/neighbors()
                                   und, je nach Sprungart, oracle.random_node()
"""

from __future__ import annotations

import math

import config
from oracles.base import BudgetExceeded
from sampling.base import Sample, Sampler
from sampling.durw import union_target
from sampling.jumps import JumpStrategy, UniformJump


class _WeightTree:
    """Fenwick-Baum ueber die k Walker-Gewichte: ziehen und aendern in
    O(log k).

    Die naive Auswahl waere eine Summe ueber alle k Gewichte je Schritt. Bei
    k = 1000 und einigen 100 000 Schritten sind das Milliarden von
    Python-Operationen -- der Baum ist hier kein Feinschliff, sondern die
    Bedingung dafuer, dass grosse k ueberhaupt laufen.

    Je Schritt aendert sich genau ein Gewicht (das des gezogenen Walkers, der
    weitergezogen ist); alle anderen stehen fest, weil deg_Gu beim Erstbesuch
    eingefroren wird.
    """

    def __init__(self, weights: list[float]) -> None:
        self.n = len(weights)
        self.w = [float(x) for x in weights]
        # 1-basiert, tree[0] bleibt ungenutzt
        self.tree = [0.0] * (self.n + 1)
        for i in range(1, self.n + 1):
            self.tree[i] += self.w[i - 1]
            j = i + (i & -i)
            if j <= self.n:
                self.tree[j] += self.tree[i]
        self.total = math.fsum(self.w)
        # groesste Zweierpotenz <= n, Startschrittweite der Abstiegssuche
        self._step0 = 1 << max(0, self.n.bit_length() - 1)

    def update(self, i: int, value: float) -> None:
        delta = value - self.w[i]
        if delta == 0.0:
            return
        self.w[i] = value
        self.total += delta
        j = i + 1
        while j <= self.n:
            self.tree[j] += delta
            j += j & -j

    def find(self, x: float) -> int:
        """Kleinster Index, dessen Praefixsumme x echt uebersteigt.

        Ein Walker mit Gewicht 0 laesst die Praefixsumme unveraendert und kann
        deshalb nie das Ergebnis sein -- genau das braucht `no_jumps`, wo tote
        Walker auf 0 gesetzt werden.
        """
        pos = 0
        step = self._step0
        while step:
            nxt = pos + step
            if nxt <= self.n and self.tree[nxt] <= x:
                pos = nxt
                x -= self.tree[pos]
            step >>= 1
        if pos >= self.n:
            # Nur durch Rundungsdrift in `total` erreichbar: auf den letzten
            # Walker mit Gewicht > 0 zurueckfallen, statt einen toten zu ziehen.
            pos = self.n - 1
            while pos > 0 and self.w[pos] <= 0.0:
                pos -= 1
        return pos


class DufsSampler(Sampler):
    """DUFS ueber Nachbarschaftsabfragen plus Spruenge; liefert die volle
    Trajektorie. Das Aufteilen in Sample-Sets uebernimmt sampling.thinning.

    `n_walkers` ist das k. Es ist *keine* Zahl von Faengen -- alle k Walker
    laufen gleichzeitig auf demselben G_u und liefern eine einzige
    Trajektorie. Die Faenge fuer Capture-Recapture sind weiterhin `n_walks`,
    und G_u wird je Fang neu aufgebaut (Begruendung s. sampling.durw).

    `jump_weight` ist das w der Sprungregel, mit derselben Bedeutung wie bei
    DURW. Bei `no_jumps=True` ist es bedeutungslos und geht auch nicht in die
    Auswahlgewichte ein -- dort ist das Gewicht eines Walkers schlicht sein
    deg_Gu.
    """

    def __init__(
        self,
        jump: JumpStrategy | None = None,
        jump_weight: float = config.DURW_JUMP_WEIGHT,
        n_walkers: int = config.DUFS_WALKERS,
        n_walks: int = 1,
        burn_in: int = 0,
        union_jumps: bool = False,
        no_jumps: bool = False,
    ) -> None:
        self.jump = jump or UniformJump()
        self.jump_weight = float(jump_weight)
        if self.jump_weight <= 0:
            # Wie bei DURW: w = 0 ist nicht der Grenzfall, sondern `no_jumps`.
            raise ValueError(f"jump_weight muss > 0 sein, ist {self.jump_weight}")
        self.n_walkers = int(n_walkers)
        if self.n_walkers < 1:
            raise ValueError(f"n_walkers muss >= 1 sein, ist {self.n_walkers}")
        self.n_walks = n_walks
        self.burn_in = burn_in
        self.union_jumps = bool(union_jumps)
        self.no_jumps = bool(no_jumps)
        if self.no_jumps and self.union_jumps:
            raise ValueError(
                "no_jumps schliesst union_jumps aus -- ohne Sprung gibt es kein "
                "Sprungziel zu waehlen. Die Sprungart bleibt trotzdem noetig, "
                "sie liefert die k Startknoten.")
        self.name = f"dufs_{self.jump.name}"

    def key(self) -> str:
        """Alles, was den Walk steuert -- die Sprungart steckt im Namen."""
        base = (f"{self.name}|w{self.jump_weight:g}|k{self.n_walkers}"
                f"|walks{self.n_walks}|burn{self.burn_in}")
        return (base + ("|union" if self.union_jumps else "")
                + ("|nojump" if self.no_jumps else ""))

    def _n_seeds(self, oracle) -> int:
        """k, gedeckelt auf config.DUFS_MAX_SEED_SHARE des Budgets.

        Ohne Deckel liefe DUFS bei kleinem Budget und grossem k gar keinen
        Schritt mehr -- das ganze Budget ginge fuer die Startknoten drauf und
        der Kollisionsschaetzer haette nichts zu zaehlen. Bei einer Namensliste
        ist der Deckel nur eine Naeherung: die Nieten kommen obendrauf und sind
        nicht vorhersagbar, deshalb faengt sample() das Budgetende beim Seeding
        zusaetzlich ab.
        """
        cost = oracle.cost_random_node
        if cost <= 0:
            return self.n_walkers
        room = int(oracle.budget * config.DUFS_MAX_SEED_SHARE / cost)
        return max(1, min(self.n_walkers, room))

    def sample(self, oracle) -> list[Sample]:
        # Auswahlgewicht: mit Sprung w + deg_Gu (die Sprungkante zaehlt mit),
        # ohne Sprung nur deg_Gu.
        w = self.jump_weight
        w_sel = 0.0 if self.no_jumps else w
        trace: list[Sample] = []
        current: list[Sample] = []
        try:
            for walk in range(self.n_walks):
                limit = (math.inf if walk == self.n_walks - 1
                         else oracle.budget * (walk + 1) / self.n_walks)
                current = []
                # G_u dieses Fangs -- Bedeutung und Invarianten wie bei DURW,
                # nur von allen k Walkern gemeinsam gefuellt.
                adj: dict[int, list[int]] = {}
                back: dict[int, list[int]] = {}
                # Nur fuer union_jumps: besuchte Knoten ausserhalb von S. Das
                # ist H, und H ist die Historie *aller* Walker -- ein Knoten
                # gehoert dazu, weil er besucht wurde, nicht weil ein
                # bestimmter Walker dort war.
                outside: list[int] = []

                def arrive(v: int) -> int:
                    """Ankunft auf v: Nachbarn holen, ggf. G_u einfrieren,
                    deg_Gu zurueckgeben. Der Erstbesuch-Block ist der von DURW
                    -- nur Kanten auf noch unbesuchte Knoten, damit kein
                    besuchter Knoten je seinen Grad aendert."""
                    out = oracle.neighbors(v)
                    if v not in adj:
                        fresh = [int(x) for x in out if int(x) not in adj]
                        adj[v] = fresh + back.pop(v, [])
                        for x in fresh:
                            back.setdefault(x, []).append(v)
                        if self.union_jumps and not oracle.in_jump_set(v):
                            outside.append(v)
                    return len(adj[v])

                # -- Startknoten ------------------------------------------
                # Einzeln gezogen, damit ein Budgetende mittendrin (Nieten
                # einer Namensliste sind nicht vorhersagbar) den Lauf nicht
                # verwirft, sondern mit den bisherigen Walkern weiterlaufen
                # laesst. Bei k = 1 ist das genau oracle.seed_nodes(1) von
                # DURW -- gleiche Ziehung, gleicher Zufallsstrom.
                L: list[int] = []
                weights: list[float] = []
                seeded_out = False
                try:
                    for _ in range(self._n_seeds(oracle)):
                        v = int(oracle.seed_nodes(1)[0])
                        L.append(v)
                        weights.append(w_sel + arrive(v))
                except BudgetExceeded:
                    seeded_out = True
                if not L:
                    raise BudgetExceeded("Budget vor dem ersten Startknoten erschoepft")

                k = len(L)
                tree = _WeightTree(weights)
                jumped = [False] * k   # kein Startknoten ist ein Sprungziel
                step = 0
                while not seeded_out and oracle.queries < limit:
                    if tree.total <= 0.0:
                        # Nur bei no_jumps erreichbar: alle Walker sitzen auf
                        # deg_Gu = 0 fest.
                        break
                    # Bei k = 1 gibt es nichts zu ziehen -- und es wird auch
                    # kein Zufallswert verbraucht, sonst waere der Strom
                    # gegenueber DURW verschoben (s. Modul-Docstring).
                    i = 0 if k == 1 else tree.find(oracle.rng.random() * tree.total)
                    u = L[i]
                    # Kein neues neighbors() hier: adj[u] steht seit der
                    # Ankunft fest, die Abfrage ist dort bezahlt worden.
                    nbrs = adj[u]

                    if step >= self.burn_in:
                        current.append(Sample(u, len(nbrs), step, walk,
                                              oracle.in_jump_set(u), jumped[i],
                                              walker=i))
                        oracle.mark()
                    step += 1

                    if self.no_jumps:
                        if not nbrs:
                            # Dieser Walker steckt fest -- Gewicht 0, er wird
                            # nie wieder gezogen. Die uebrigen laufen weiter.
                            tree.update(i, 0.0)
                            continue
                        jumped[i] = False
                        v_new = nbrs[oracle.rng.randrange(len(nbrs))]
                    else:
                        # Absprungregel des Originals; bei deg 0 ist
                        # w/(w+0) = 1, der Sprung also erzwungen.
                        hit = oracle.rng.random() < w / (w + len(nbrs))
                        jumped[i] = hit
                        if hit and self.union_jumps:
                            v_new = int(union_target(oracle, outside))
                        elif hit:
                            v_new = int(self.jump.next_node(oracle))
                        else:
                            v_new = nbrs[oracle.rng.randrange(len(nbrs))]

                    L[i] = v_new
                    # arrive() zuerst: erst danach steht deg_Gu(v_new) fest,
                    # und nur ein exaktes Gewicht macht die Auswahl unverzerrt.
                    tree.update(i, w_sel + arrive(v_new))
                trace.extend(current)
                current = []
        except BudgetExceeded:
            pass
        trace.extend(current)   # der abgebrochene Fang zaehlt mit
        return trace
