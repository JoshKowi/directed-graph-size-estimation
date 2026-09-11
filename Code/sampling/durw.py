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

    dead_end -- eine Sackgasse hat bei DURW immer einen Zug, sie wird nie
                absorbierend. Wurde sie ueber eine Kante erreicht, steht genau
                diese Kante in ihrem G_u-Grad (sie wurde aufgezeichnet, als der
                Vorgaenger besucht wurde) -- der Walk geht mit
                deg_Gu/(w+deg_Gu) zurueck. Nur wer per Sprung auf einer
                Sackgasse landet, hat deg_Gu = 0, und dann ist w/(w+0) = 1, der
                Sprung also erzwungen. Beides faellt aus der Sprungregel selbst;
                ein eigener Zweig dafuer wuerde nur den Zufallsstrom
                verschieben.

                Nicht verwechseln: "Sackgasse in G_d" (kein Ausgangsgrad) und
                "deg_Gu = 0" sind verschiedene Dinge. Auf gpt4o_io gerichtet
                haben 51 % der besuchten Sackgassen deg_Gu >= 1, auf gpt4_io
                70 %; ihr mittlerer G_u-Grad ist 0,75 bzw. 1,22. Sie springen
                also haeufig (im Mittel 73 % bzw. 60 %), aber nicht immer.
    allow_self_loops -- graphs.graph._simplify() entfernt Schlingen bereits
                beim Laden, in G_u kann keine entstehen.

**Abweichung vom Original, wenn der Sprung aus einer Liste kommt.** Bei Ribeiro
& Towsley ist der virtuelle Knoten sigma mit *ganz V* verbunden -- der Sprung
ist gleichverteilt ueber die Knotenmenge, und genau das macht
pi(v) ~ w + deg_Gu(v) moeglich. Zieht der Sprung dagegen aus einer externen
Namensliste (oracles.name_list), ist sigma nur mit der Trefferteilmenge S
verbunden, und die Stationaerverteilung ist eine andere:

    pi(v) ~ deg_Gu(v) + w * 1[v in S]

Das ist eine *Variante*, kein Nachbau des Papers. Der Sampler selbst bleibt
davon unberuehrt -- er merkt sich je Sample nur, ob der Knoten in S liegt
(`Sample.in_jump_set`); welche Gewichtung daraus folgt, entscheidet
weighting.DurwWeighting (Original, S = V) bzw.
weighting.DurwJumpSetWeighting (Variante).

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
        history_jumps: bool = False,
        history_weight: float = 1.0,
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
        # Sprung auf S u H statt nur auf die Sprungmenge der Liste -- s.
        # Modul-Docstring, Abschnitt "Sprung auf S u H".
        self.history_jumps = bool(history_jumps)
        self.history_weight = float(history_weight)
        if self.history_jumps and self.history_weight <= 0:
            raise ValueError(
                f"history_weight muss > 0 sein, ist {self.history_weight}: bei 0 "
                "koennten Knoten ausserhalb von S nicht mehr springen.")
        self.name = f"durw_{self.jump.name}"

    def key(self) -> str:
        """Alles, was den Walk steuert -- die Sprungart steckt im Namen."""
        base = (f"{self.name}|w{self.jump_weight:g}|seeds{self.n_seeds}"
                f"|walks{self.n_walks}|burn{self.burn_in}")
        # Nur wenn eingeschaltet -- sonst bleibt der Schluessel der alten
        # Estimators unveraendert und ihre Walk-Gruppen stimmen weiter.
        return base + (f"|hist{self.history_weight:g}" if self.history_jumps else "")

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
                # Die *verschiedenen* besuchten Knoten, in Besuchsreihenfolge --
                # nur fuer history_jumps. Gezogen wird gleichverteilt daraus,
                # nicht aus der Trajektorie mit Vielfachheit: sonst aenderte sich
                # das sigma-Gewicht eines Knotens bei jedem Wiederbesuch und
                # froere nicht mehr ein.
                visited: list[int] = []
                u = int(oracle.seed_nodes(self.n_seeds)[0])
                step = 0
                jumped = False   # der Seed selbst ist kein Sprungziel
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
                        if self.history_jumps:
                            visited.append(u)
                    nbrs = adj[u]

                    if step >= self.burn_in:
                        # in_jump_set nur fuer die Gewichtung eines
                        # listenbasierten Sprungs relevant; beim
                        # gleichverteilten Sprung ist es immer True (s.
                        # oracles.base.Oracle.in_jump_set).
                        # sigma_weight: im alten Modus 1.0 (Original-DURW), im
                        # neuen m(u) + beta. Per Keyword, damit die Position
                        # der anderen Felder nicht davon abhaengt.
                        if self.history_jumps:
                            sigma = oracle.jump_multiplicity(u) + self.history_weight
                        else:
                            sigma = 1.0
                        current.append(Sample(u, len(nbrs), step, walk,
                                              oracle.in_jump_set(u), jumped,
                                              sigma_weight=sigma))
                        oracle.mark()  # fuer Budget-Zwischenstaende, s. oracles.base
                    step += 1

                    # Bei deg 0 ist w/(w+0) = 1 -- der Sprung ist dann sicher,
                    # ohne dass es einen eigenen Zweig braucht.
                    if not self.history_jumps:
                        # Original-DURW: jeder Knoten springt mit w/(w + deg).
                        jumped = oracle.rng.random() < w / (w + len(nbrs))
                        if jumped:
                            u = int(self.jump.next_node(oracle))
                        else:
                            u = nbrs[oracle.rng.randrange(len(nbrs))]
                    else:
                        # Sprung auf S u H. u ist immer schon besucht, liegt also
                        # in H: c(u) = m(u) + beta > 0, jeder Knoten kann springen.
                        c = oracle.jump_multiplicity(u) + self.history_weight
                        jumped = oracle.rng.random() < w * c / (w * c + len(nbrs))
                        if jumped:
                            # Landung ~ c: mit Wahrscheinlichkeit
                            # sum(m) / (sum(m) + beta*|H|) aus der Liste, sonst
                            # gleichverteilt aus der Historie. Die Mischung ist
                            # exakt c(v)/sum(c) -- nachgerechnet.
                            mass = oracle.list_mass()
                            hist = self.history_weight * len(visited)
                            if oracle.rng.random() < mass / (mass + hist):
                                u = int(self.jump.next_node(oracle))
                            else:
                                # aus dem eigenen Gedaechtnis: kostet nichts; die
                                # Nachbarabfrage bei Ankunft ist ein Cache-Treffer
                                u = visited[oracle.rng.randrange(len(visited))]
                        else:
                            u = nbrs[oracle.rng.randrange(len(nbrs))]
                trace.extend(current)
                current = []
        except BudgetExceeded:
            pass
        trace.extend(current)   # der abgebrochene Fang zaehlt mit
        return trace
