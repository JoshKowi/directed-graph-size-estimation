"""DURW -- Directed Unbiased Random Walk (Ribeiro & Towsley).

Der Random Walk in sampling.samplers laeuft auf den gerichteten Views auf einem
Graphen, auf dem er gar nicht laufen duerfte: pi(u) ~ deg(u) gilt nur
ungerichtet und zusammenhaengend, eingehende Kanten sind unbeobachtbar, und
jede Sackgassen-Strategie verschiebt die Verteilung noch einmal. DURW baut sich
deshalb *waehrend des Laufs* einen ungerichteten Graphen G_u, auf dem die
Theorie wieder traegt.

Zwei Zutaten:

1. Rueckwaerts begehbare Kanten. Jede beobachtete Ausgangskante u -> v wird
   gemerkt; landet der Walk spaeter auf v, darf er sie rueckwaerts nach u
   nehmen. Aber nur, solange v noch *unbesucht* ist: Kanten auf bereits
   besuchte Knoten werden verworfen. Damit steht der Grad eines Knotens in
   G_u in dem Moment fest, in dem er zum ersten Mal besucht wird, und aendert
   sich nie wieder -- genau das braucht die Gewichtung, denn sonst haenge sie
   von Kanten ab, die der Walk erst spaeter sieht.

2. Gradproportionale Spruenge. Mit Wahrscheinlichkeit w/(w + deg_Gu(v))
   springt der Walk auf einen zufaellig gezogenen Knoten (sampling.jumps).
   Das entspricht einer Kante mit Gewicht w zu einem virtuellen Knoten sigma,
   der mit allen Knoten verbunden ist. Auf diesem gewichteten Graphen ist

       pi(v) = (w + deg_Gu(v)) / (vol(V) + w|V|)

   -- bis auf die unbekannte Normierung bekannt, sobald v besucht ist. Genau
   diese Groesse setzt weighting.DurwWeighting als 1/(w + deg) ein; die
   Normierung kuerzt der Kollisionsschaetzer heraus.

Was hier *nicht* vorkommt:

    dead_end -- bei deg_Gu(v) = 0 ist w/(w+0) = 1, der Walk springt
                zwangslaeufig. Sackgassen sind bei DURW kein Sonderfall,
                sondern der Grenzfall der Sprungregel. Ein eigener Zweig dafuer
                wuerde nur den Zufallsstrom verschieben.
    allow_self_loops -- graphs.graph._simplify() entfernt Schlingen bereits
                beim Laden, in G_u kann keine entstehen.

Wichtig fuer alles, was danach kommt: `Sample.degree` traegt hier den Grad in
G_u, *nicht* den Ausgangsgrad wie bei RandomWalkSampler. InverseDegreeWeighting
passt damit nicht zu DURW -- die richtige Gewichtung ist DurwWeighting.

Schnittstelle:
    class DurwSampler(Sampler)  -- braucht oracle.seed_nodes()/neighbors()
                                   und, je nach Sprungart, oracle.random_node()
"""

from __future__ import annotations

import math

import config
from oracles.base import BudgetExceeded
from sampling.base import Sample, Sampler
from sampling.jumps import JumpStrategy, UniformJump


class DurwSampler(Sampler):
    """DURW ueber Nachbarschaftsabfragen plus Spruenge; liefert die volle
    Trajektorie. Das Aufteilen in Sample-Sets uebernimmt sampling.thinning.

    `jump_weight` ist das w der Sprungregel. Groesseres w heisst: haeufiger
    springen, also weniger Autokorrelation und bessere Abdeckung, aber mehr
    Budget fuer Spruenge statt fuer Schritte (ein Sprung kostet
    COST_RANDOM_NODE, ein Wiederbesuch nur COST_CACHE_HIT). w -> 0 ergibt
    einen reinen Random Walk auf G_u, w -> unendlich gleichverteiltes Ziehen.

    `n_walks` > 1 laesst mehrere Faenge nacheinander laufen -- die Form, die
    Capture-Recapture braucht (siehe sampling.samplers.RandomWalkSampler zur
    Budget-Aufteilung). G_u wird dabei je Fang *neu* aufgebaut: sonst erbte
    der zweite Fang die eingefrorenen Grade des ersten und die beiden Faenge
    waeren ueber diese Historie voneinander abhaengig. Jeder Fang ist so fuer
    sich ein gueltiger DURW-Lauf mit eigenem, gueltigem pi.
    """

    def __init__(
        self,
        jump: JumpStrategy | None = None,
        jump_weight: float = config.DURW_JUMP_WEIGHT,
        n_seeds: int = 1,
        n_walks: int = 1,
        burn_in: int = 0,
    ) -> None:
        self.jump = jump or UniformJump()
        self.jump_weight = float(jump_weight)
        if self.jump_weight <= 0:
            # w = 0 kappt die Sprungkante: der Walk sitzt in der ersten
            # Sackgasse fest, und pi ~ deg_Gu waere auf einem unzusammen-
            # haengenden G_u ohnehin nicht mehr die Stationaerverteilung.
            raise ValueError(f"jump_weight muss > 0 sein, ist {self.jump_weight}")
        self.n_seeds = n_seeds
        self.n_walks = n_walks
        self.burn_in = burn_in
        self.name = f"durw_{self.jump.name}"

    def key(self) -> str:
        """Alles, was den Walk steuert -- die Sprungart steckt im Namen."""
        return (f"{self.name}|w{self.jump_weight:g}|seeds{self.n_seeds}"
                f"|walks{self.n_walks}|burn{self.burn_in}")

    def sample(self, oracle) -> list[Sample]:
        w = self.jump_weight
        trace: list[Sample] = []
        current: list[Sample] = []
        try:
            for walk in range(self.n_walks):
                # Der letzte Walk laeuft bis zum Budgetende -- wie bei
                # RandomWalkSampler, damit n_walks=1 der einfache Fall bleibt.
                limit = (math.inf if walk == self.n_walks - 1
                         else oracle.budget * (walk + 1) / self.n_walks)
                current = []
                # G_u dieses Fangs:
                #   adj  -- eingefrorene Nachbarschaft *besuchter* Knoten.
                #           Zugleich die Knotenmenge V(i): u in adj <=> besucht.
                #   back -- beobachtete Kanten auf noch *unbesuchte* Knoten,
                #           also E(i) eingeschraenkt auf offene Endpunkte.
                adj: dict[int, list[int]] = {}
                back: dict[int, list[int]] = {}
                u = int(oracle.seed_nodes(self.n_seeds)[0])
                step = 0
                while oracle.queries < limit:
                    # Auch beim Wiederbesuch gefragt: der Cache-Treffer kostet
                    # (oracles.base), sonst liefe ein Walk in bekanntem Gebiet
                    # gratis weiter. Die Antwort selbst braucht nur der
                    # Erstbesuch -- danach zaehlt die eingefrorene Liste.
                    out = oracle.neighbors(u)
                    if u not in adj:
                        # N'(u): nur Kanten auf noch unbesuchte Knoten. Kanten
                        # auf besuchte Knoten fallen weg, damit kein besuchter
                        # Knoten je seinen Grad aendert.
                        fresh = [int(v) for v in out if int(v) not in adj]
                        # back[u] kann nach diesem Pop nicht mehr wachsen: neue
                        # Eintraege entstehen nur fuer unbesuchte Knoten, und u
                        # steht ab jetzt in adj.
                        adj[u] = fresh + back.pop(u, [])
                        for v in fresh:
                            back.setdefault(v, []).append(u)
                    nbrs = adj[u]

                    if step >= self.burn_in:
                        current.append(Sample(u, len(nbrs), step, walk))
                        oracle.mark()  # fuer Budget-Zwischenstaende, s. oracles.base
                    step += 1

                    # Bei deg 0 ist w/(w+0) = 1 -- der Sprung ist dann sicher,
                    # ohne dass es einen eigenen Zweig braucht.
                    if oracle.rng.random() < w / (w + len(nbrs)):
                        u = int(self.jump.next_node(oracle))
                    else:
                        u = nbrs[oracle.rng.randrange(len(nbrs))]
                trace.extend(current)
                current = []
        except BudgetExceeded:
            pass
        trace.extend(current)   # der abgebrochene Fang zaehlt mit
        return trace
