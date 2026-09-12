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

    no_jumps -- w -> 0, ganz ohne Sprung (s. `no_jumps` unten). Ein ueber eine
                echte Kante erreichter Knoten hat deg_Gu >= 1 immer schon: die
                Kante, ueber die der Walk gerade gekommen ist, wurde beim
                Besuch des Vorgaengers als back-Eintrag hinterlegt. deg_Gu = 0
                kann deshalb nur der allererste Knoten eines Fangs sein (kein
                Vorgaenger, keine back-Kante) -- und auch nur, wenn er zugleich
                ein echter Sink ist (deg_out = 0). Genau das Gegenstueck zur
                Randbedingung in sampling.dead_ends fuer die undirected View
                ("nur ein voellig isolierter Seed kann dort eine Sackgasse
                sein").
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

**`union_jumps` (durwunion-*): Sprung gleichverteilt auf S u H.** Die Liste S
enthaelt jeden Knoten nur einmal (oracles.name_list.unique_index) und wird
waehrend des Laufs um jeden besuchten Knoten ergaenzt, der nicht darin steht
(H = verschiedene besuchte Knoten). Ein Sprung zieht eine Position dieser
erweiterten Liste gleichverteilt; ist es eine Niete, wird die *ganze* Ziehung
wiederholt -- ueber Liste und Historie, nicht nur ueber die Liste. Damit landet
der Sprung exakt gleichverteilt auf S u H, ohne dass die Trefferzahl der
Liste bekannt sein muss.

Absprungregel w/(w + deg_Gu) und Gewicht 1/(w + deg_Gu) bleiben die des
Originals: sigma hat zu jedem Knoten in S u H eine Kante w, und jeder
besuchte Knoten liegt in H. Das ist die kleinstmoegliche Abweichung vom Paper
-- sigma ist mit S u H statt mit V verbunden -- und faellt bei S = V exakt auf
das Original zurueck. Anders als history_jumps zaehlt ein Knoten in S n H nur
einmal; sein sigma-Gewicht aendert sich also nie, nur Knoten ausserhalb von S
bekommen ihre sigma-Kante beim Erstbesuch. Grenze: Knoten, die weder in S
liegen noch von dort ueber Kanten erreichbar sind, besucht der Walk nie; die
Schaetzung laeuft dann gegen deren Komplement, nicht gegen |V|.

Wichtig fuer alles, was danach kommt: `Sample.degree` traegt hier den Grad in
G_u, *nicht* den Ausgangsgrad wie bei RandomWalkSampler. InverseDegreeWeighting
passt damit nicht zu DURW -- die richtige Gewichtung ist DurwWeighting.

**`collect_nbrs` (fuer IE2).** Beide Nachbarschaften, die hier entstehen,
werden normalerweise am Ende weggeworfen: die rohe Antwort des Oracles (`out`)
und die eingefrorene G_u-Nachbarschaft (`adj`). Mit `collect_nbrs=True` bleiben
sie als `self.observed` erhalten -- die Datengrundlage fuer
estimators.formulas.IE2SetEstimator, das Treffer gegen die Vereinigung aller
Nachbarschaften zaehlt statt Knoten-Kollisionen. Am Walk aendert das Flag
nichts: es wird kein Knoten zusaetzlich abgefragt und keine Zufallszahl
zusaetzlich gezogen, die Trajektorie ist bitgleich. Siehe sampling.observed.

Schnittstelle:
    class DurwSampler(Sampler)  -- braucht oracle.seed_nodes()/neighbors()
                                   und, je nach Sprungart, oracle.random_node()
        .observed                  -- nur mit collect_nbrs=True, s. dort
"""

from __future__ import annotations

import math

import config
from oracles.base import BudgetExceeded
import numpy as np

from sampling.base import Sample, Sampler
from sampling.jumps import JumpStrategy, UniformJump
from sampling.observed import ObservedNeighborhoods


def union_target(oracle, outside: list[int]) -> int:
    """Sprungziel gleichverteilt auf S u H.

    Die erweiterte Liste ist: die Positionen der Liste (Nieten
    eingeschlossen), dahinter die besuchten Knoten ausserhalb von S. Eine
    Position wird gleichverteilt gezogen; bei einer Niete beginnt die
    *ganze* Ziehung neu. Jeder Knoten aus S u H steht genau einmal darin,
    das Ziel ist also gleichverteilt darauf -- und dafuer muss niemand
    wissen, wie viele Listennamen treffen.

    Die Liste kostet wie jede Ziehung (Treffer COST_RANDOM_NODE, Niete
    COST_DRAW_MISS); ein Knoten aus der Historie ist eigenes Wissen und
    kostet nichts, die Nachbarabfrage bei Ankunft ist ein Cache-Treffer.

    Modul-Funktion statt Methode, weil sampling.dufs sie unveraendert
    mitbenutzt: H ist dort die Historie *aller* Walker, an der Ziehung selbst
    aendert das nichts.
    """
    n_list = oracle.list_length()
    while True:
        i = oracle.rng.randrange(n_list + len(outside))
        if i >= n_list:
            return outside[i - n_list]
        u = oracle.list_entry(i)
        if u is not None:
            return u


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

    `no_jumps=True` ist der Grenzfall w -> 0: ein reiner Random Walk auf G_u,
    ganz ohne Sprung. Begruendung s. Modul-Docstring, Abschnitt "no_jumps" --
    kurz: jeder ueber eine echte Kante erreichte Knoten hat deg_Gu >= 1 schon
    von der Ankunftskante her, `len(nbrs) == 0` kann also nur am allerersten
    Knoten eines Fangs auftreten (kein Vorgaenger), und auch dort nur, wenn er
    zugleich ein echter Sink ist -- dann endet dieser Fang dort, statt an
    einer 0/0-Division zu scheitern. Einschraenkung: pi(v) ~ deg_Gu(v) gilt
    nur innerhalb der schwach zusammenhaengenden Komponente des Seeds, denn
    ohne Sprung erreicht der Walk keine andere Komponente je (vgl.
    diagnose_walk.diagnose_connectivity -- auf den hier verwendeten Graphen
    deckt die groesste WCC praktisch alles ab).
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
        union_jumps: bool = False,
        no_jumps: bool = False,
        collect_nbrs: bool = False,
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
        # Sprung gleichverteilt auf S u H -- s. Modul-Docstring, `union_jumps`.
        self.union_jumps = bool(union_jumps)
        if self.union_jumps and self.history_jumps:
            raise ValueError("union_jumps und history_jumps schliessen sich aus.")
        # Ganz ohne Sprung -- s. Modul-Docstring, Abschnitt "no_jumps". Schliesst
        # jede Sprungziel-Strategie aus, es gibt ja keinen Sprung mehr.
        self.no_jumps = bool(no_jumps)
        if self.no_jumps and (self.history_jumps or self.union_jumps):
            raise ValueError(
                "no_jumps schliesst history_jumps und union_jumps aus -- beides "
                "sind Sprungziel-Strategien, ohne Sprung bedeutungslos.")
        # Nachbarschaften aufbewahren statt verwerfen -- s. Modul-Docstring.
        self.collect_nbrs = bool(collect_nbrs)
        if self.collect_nbrs and self.n_walks > 1:
            # G_u wird je Fang neu aufgebaut (s. Klassen-Docstring), die
            # Nachbarschaften mehrerer Faenge sind also verschiedene Graphen.
            # Ein Index ueber die zusammengeklebte Trajektorie waere still
            # falsch: Zeugenindizes aus Fang 1 gelten fuer Fang 2 nicht.
            raise ValueError(
                f"collect_nbrs braucht n_walks = 1, ist {self.n_walks}: G_u "
                "wird je Fang neu aufgebaut, eine gemeinsame Nachbarschaft "
                "ueber alle Faenge gibt es nicht.")
        self.name = f"durw_{self.jump.name}"

    def key(self) -> str:
        """Alles, was den Walk steuert -- die Sprungart steckt im Namen."""
        base = (f"{self.name}|w{self.jump_weight:g}|seeds{self.n_seeds}"
                f"|walks{self.n_walks}|burn{self.burn_in}")
        # Nur wenn eingeschaltet -- sonst bleibt der Schluessel der alten
        # Estimators unveraendert und ihre Walk-Gruppen stimmen weiter.
        # `collect_nbrs` steht im Schluessel, obwohl es die Trajektorie *nicht*
        # aendert. Grund: pipeline.estimate_group laesst den Walk vom ersten
        # Estimator der Gruppe laufen. Duerfte eine Gruppe aufzeichnende und
        # nicht aufzeichnende Varianten mischen, bekaeme IE2 je nach Reihenfolge
        # ein observed = None. Der Schluessel trennt die Gruppen deshalb.
        return (base + (f"|hist{self.history_weight:g}" if self.history_jumps else "")
                + ("|union" if self.union_jumps else "")
                + ("|nojump" if self.no_jumps else "")
                + ("|nbrs" if self.collect_nbrs else ""))

    def sample(self, oracle) -> list[Sample]:
        w = self.jump_weight
        trace: list[Sample] = []
        current: list[Sample] = []
        # Nur mit collect_nbrs belegt (s. Modul-Docstring). `raw` haelt die
        # Antwort des Oracles als Slice-View in die CSR-Kantenliste -- keine
        # kopierte Kante, kein zusaetzlicher Speicher.
        raw: dict[int, object] = {}
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
                # Nur fuer union_jumps: besuchte Knoten *ausserhalb* von S, in
                # Besuchsreihenfolge -- der Teil der erweiterten Liste, der
                # nicht schon in der Liste steht.
                outside: list[int] = []
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
                        if self.collect_nbrs:
                            raw[u] = out
                        for v in fresh:
                            back.setdefault(v, []).append(u)
                        if self.history_jumps:
                            visited.append(u)
                        if self.union_jumps and not oracle.in_jump_set(u):
                            outside.append(u)
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

                    if self.no_jumps:
                        # len(nbrs) == 0 kann hier nur der allererste Knoten
                        # dieses Fangs sein (kein Vorgaenger, keine back-
                        # Kante) und zugleich ein echter Sink -- s.
                        # Klassen-Docstring. Dann ist der Fang hier zu Ende,
                        # statt an w/(w+0) zu scheitern.
                        if not nbrs:
                            break
                        jumped = False
                        u = nbrs[oracle.rng.randrange(len(nbrs))]
                    # Bei deg 0 ist w/(w+0) = 1 -- der Sprung ist dann sicher,
                    # ohne dass es einen eigenen Zweig braucht.
                    elif not self.history_jumps:
                        # Original-DURW: jeder Knoten springt mit w/(w + deg).
                        jumped = oracle.rng.random() < w / (w + len(nbrs))
                        # union_jumps aendert nur das *Ziel*: die Absprungregel
                        # bleibt die des Originals (jeder besuchte Knoten liegt
                        # in H, hat also genau eine sigma-Kante w).
                        if jumped and self.union_jumps:
                            u = int(union_target(oracle, outside))
                        elif jumped:
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
        if self.collect_nbrs:
            # `adj` als int32-Arrays statt Python-Listen: 4 statt 36 Byte je
            # Kante, und die Formeln rechnen ohnehin mit numpy. `adj` ist hier
            # der Stand des *letzten* Fangs -- mit n_walks > 1 abgewiesen.
            self.observed = ObservedNeighborhoods(
                raw=raw,
                gu={u: np.fromiter(vs, dtype=np.int32, count=len(vs))
                    for u, vs in adj.items()},
            )
        return trace
