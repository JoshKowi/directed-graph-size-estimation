"""NMMC -- Non-Markovian Monte Carlo (Lee, Kang & Eun 2019).

Der einfache Random Walk hat auf einer gerichteten Sicht keine bekannte
Stationaerverteilung, und DURW (sampling.durw) erkauft sich eine nur mit einem
gleichverteilten Sprung -- also mit genau der Kenntnis von V, die hier
geschaetzt werden soll. NMMC schliesst diese Luecke: es kommt ohne jede
Ziehung aus V aus, weil es ausschliesslich auf **bereits besuchte** Knoten
umverteilt.

Aufbau. Vorschlagskette Q ist der Simple Random Walk auf den Ausgangskanten,
Q_ij = 1/d+(i). Daraus wird eine *transiente* Kette P auf N + {0} gebaut: der
Zug i -> j wird mit gamma_ij angenommen, sonst wird die Kette im kuenstlichen
Zustand 0 absorbiert. Auf die Absorption zu warten waere aussichtslos (die
Ueberlebenswahrscheinlichkeit faellt exponentiell); stattdessen wird sie sofort
durch eine Umverteilung auf die gewichtete eigene Historie ersetzt
(sampling.history). Der Prozess ist dadurch nicht mehr markovsch, und seine
empirische Verteilung mu_t konvergiert gegen die *quasi-stationaere*
Verteilung von P.

Warum das die Zielverteilung trifft: mit

    b_ij = (pi(j)/pi(i)) * d+(i)/d-(j),   gamma_ij = min(1, b_ij/c)

rechnet sich in zwei Zeilen nach, dass pi linker Eigenvektor von P ist --

    (pi P)_j = sum_{i->j} pi_i * (1/d+(i)) * (pi_j d+(i)) / (pi_i d-(j) c)
             = d-(j) * pi_j / (d-(j) c)
             = pi_j / c

also pi P = (1/c) pi. Nach Perron-Frobenius ist pi damit *die* QSD. Alle
Groessen sind lokal: d+(i) faellt bei der Nachbarabfrage ab, d-(j) liefert
sampling.indegree (geschaetzt oder exakt).

Zwei Ziele sind implementiert:

    uniform -- pi = u, dann b_ij = d+(i)/d-(j). Die Samples sind direkt
               gleichverteilt, es braucht keine Gewichtung.
    indeg   -- pi ~ d-, dann kuerzt sich d-(j) heraus und b_ij = d+(i)/d-(i)
               haengt nur noch an i. Das Paper misst dieses Ziel als deutlich
               schneller konvergierend; die Verzerrung korrigiert
               weighting.InDegreeWeighting.

`c` ist global (c >= max b_ij) und unbekannt, wird also als c_t mitgelernt
(Algorithmus 2 des Papers): mit Wahrscheinlichkeit p zieht ein Schritt c_t auf
das gerade gesehene b_ij hoch, sonst laesst er es stehen -- dann wird
gamma = 1 und der Zug sicher angenommen. Kleines p ist kein Schoenheitsfehler,
sondern noetig: ein einziges grosses b_ij wuerde sonst die Annahmequote aller
folgenden Schritte druecken, der Walk verteilte nur noch zwischen bekannten
Knoten um und diffundierte nicht mehr.

Was daran nicht exakt ist -- gehoert in jede Interpretation der Kurven:

**Solange c_t noch waechst, ist die QSD noch nicht pi.** Die Herleitung oben
setzt gamma_ij = b_ij/c ohne Kappung voraus; unterwegs greift aber min(1, .).
Mit c_0 = 1 und p = 0,01 koennen kleine Budgets enden, bevor c_t oben ist. Die
Verzerrung ist damit systematisch budgetabhaengig -- die Kurve ueber dem Budget
mischt echtes Signal mit dem Einschwingen von c_t. Genestete Budgets bleiben
davon unberuehrt exakt (s.u.).

**Sackgassen brechen die Irreduzibilitaet.** Bei d+(i) = 0 ist Q_srw gar nicht
definiert; hier wird das als Absorption mit Wahrscheinlichkeit 1 behandelt, mit
anschliessender Umverteilung. Das ist die NMMC-eigene Mechanik und kein
Fremdkoerper -- anders als bei DURW braucht es dafuer aber einen eigenen Zweig,
weil dort der Sprung aus der Sprungregel selbst faellt.

    Was gilt: die Sackgasse wird besucht, bemustert und traegt QSD-Masse
    nu_d = theta^-1 * sum_{i->d} nu_i P_id > 0, sobald ein Vorgaenger Masse
    hat. Sackgassen fallen also *nicht* aus der Stichprobe -- bei bis zu 53 %
    Sackgassen (gpt4o_io) waere alles andere fatal.

    Was nicht gilt: Irreduzibilitaet auf N. Eine Sackgasse ist eine Nullzeile
    in P; allgemein ist der Perron-Wert das Maximum ueber die starken
    Zusammenhangskomponenten der Kondensation, und der linke Perron-Vektor lebt
    auf der dominanten Komponente plus deren Vorwaerts-Abschluss. Zusammen mit
    mu_0 = delta_{Z_0} heisst das: mu_t konvergiert gegen die QSD von P
    eingeschraenkt auf die vom Startknoten aus erreichbare Menge. **NMMC
    schaetzt |supp mu|, nicht |V|** -- dieselbe Lage wie bei wis-katzir__indep,
    das auf gpt-4o-io nur die 46,35 % Knoten mit ausgehenden Kanten schaetzt
    (README, "Entwurfsentscheidungen", Punkt 3). Mit dahin sind Eindeutigkeit
    der QSD bei Gleichstand zweier Perron-Werte und jede Aussage ueber die
    Konvergenzrate.

Nicht uebernommen wurde die Vorschlagskette aus 6.4 des Papers, die
Sackgassen ueber einen Sprung auf die Einstiegsmenge S aufloest: dort bekommt
*jeder* Knoten eine Sprungkante nach S, damit ist |S_j| nicht mehr d-(j),
sondern haengt an |V| -- die Groesse, die geschaetzt werden soll. Die
Absorption bleibt damit die einzige lokal bestimmbare Antwort.

**Mehrere Agenten.** Das Paper faehrt in jeder Simulation 100 bis 10^4
Agenten; ein einzelner kommt dort nicht vor. Was dabei geteilt wird, legt es an
drei Stellen getrennt fest:

    Historie mu_t / Umverteilung -- NICHT geteilt, je Agent eigen ("each agent
        maintains its own historical empirical distribution" , Abschnitt 5). Die
        Historien werden dort nur fuer die TVD-Messung vereinigt, nie fuer die
        Umverteilung: ein Agent springt ausschliesslich auf eigene Besuche.
    Cache -- geteilt ("The local cache can also be easily shared among multiple
        crawling agents", 6.5). Hier faellt das von selbst an, weil alle Agenten
        auf demselben Oracle laufen.
    Online-Schaetzung von d- -- geteilt ("All the agents share the estimate of
        the in-degree of each node", 6.3). Deshalb liegen `state` und
        `expanded` ausserhalb der Agentenschleife.

Abweichend vom Paper teilen sich die Agenten hier das **Budget**: dort laeuft
jeder Agent volle t Schritte und die Kosten fallen nur ueber den Cache
zusammen. Die Aufteilung folgt dem Muster von sampling.durw bei n_walks > 1.

Was das bringt und warum, gemessen auf Slashdot0811 gerichtet bei Budget 20 %
mit Ziel pi ~ d-: 1 Agent 0,407 -- 10: 0,500 -- 50: 0,630 -- 200: 0,771. Die
Abdeckung steigt dabei nur um 15 %, die Schaetzung um 89 %. Der Gewinn kommt
also kaum aus mehr gesehenen Knoten, sondern daraus, dass **Scheinkollisionen
wegfallen**: ein festsitzender Einzelagent besucht dieselben paar tausend Knoten
tausendfach, und genau diese Wiederholungen zaehlt der Kollisionsschaetzer
(estimators.formulas) als Treffer und drueckt |V| nach unten.

Unbequem dabei: ein neuer Agent setzt c_t auf 1 zurueck, also in die Phase mit
hoher Annahmequote. Kuerzere Laeufe heissen damit, dass c_t dem wahren c noch
ferner bleibt -- die realisierte QSD ist *weiter* von pi entfernt, nicht naeher.
Die Schaetzung wird besser, obwohl die Garantie schlechter erfuellt ist. Wer
das Gegenteil messen will, setzt `shared_c=True`.

Nach oben begrenzt ist K durch die Zahl der Einstiegsknoten: alle Agenten
starten an config.SEED_NODES, bei Slashdot fuenf. Bei K = 1000 macht jeder
Agent nur rund 42 Schritte, bevor der naechste wieder an einem dieser fuenf
beginnt -- 8,2 % aller Samples landen dann in deren 1-Hop-Umkreis (20 Knoten),
und die Scheinkollisionen kommen durch die Hintertuer zurueck. Auf Slashdot
liegt das Optimum bei K ~ 100.

Randeffekt auf den Safety Margin: estimators.formulas._collisions vergleicht
Listenpositionen, nicht Sample.step. An den K Nahtstellen zwischen den Agenten
werden dadurch je m Paare ausgelassen, die in Wahrheit unabhaengig sind -- bei
K = 1000 und m = 10 rund 10^4 von ~10^10 Paaren. Konservativ und
vernachlaessigbar.

Wichtig fuer alles, was danach kommt: `Sample.degree` traegt hier den
**Eingangs**grad (ggf. geschaetzt), nicht den Ausgangsgrad wie bei
RandomWalkSampler und nicht den G_u-Grad wie bei DurwSampler. Die passende
Gewichtung ist weighting.InDegreeWeighting.

Schnittstelle:
    class NmmcSampler(Sampler)  -- braucht oracle.seed_nodes()/neighbors(),
                                   je nach indeg zusaetzlich in_degree()
"""

from __future__ import annotations

import math

import config
from oracles.base import BudgetExceeded
from sampling.base import Sample, Sampler
from sampling.history import WeightedHistory
from sampling.indegree import IN_DEGREES, InDegreeSource

TARGETS = ("uniform", "indeg")


class NmmcSampler(Sampler):
    """NMMC ueber Nachbarschaftsabfragen plus Umverteilung auf die Historie;
    liefert die volle Trajektorie. Das Aufteilen in Sample-Sets uebernimmt
    sampling.thinning.

    `alpha` ist das Gedaechtnis der Umverteilung (w_k = k^alpha, s.
    sampling.history): 0 zieht gleichverteilt aus der Besuchsfolge mit
    Vielfachheit -- der Fall, den sampling.dead_ends.HistoryJump abbildet --,
    grosses alpha fast nur zuletzt Besuchtes.

    `c_update_p` ist das p aus Algorithmus 2 (s. Modul-Docstring).

    `n_agents` ist die Zahl der Agenten aus dem Paper (s. Modul-Docstring).
    Sie teilen sich Budget, Cache und die d--Schaetzung, haben aber je eine
    eigene Historie -- und, sofern `shared_c` nicht gesetzt ist, ein eigenes
    c_t.

    Bei n_agents > 1 liest der Sampler `oracle.budget`, um die Grenzen zu
    setzen. Ein Praefix des Laufs ist damit nicht mehr bitgleich mit einem
    eigenstaendigen Lauf bei kleinerem Budget, und
    estimators/methods/nmmc.py setzt deshalb supports_nested = (n_agents == 1)
    -- dasselbe Muster wie estimators/methods/capture_recapture.py. Bei
    n_agents = 1 bleibt die Schleife exakt die alte (`limit` ist dann
    math.inf), und die Eigenschaft bleibt erhalten.
    """

    def __init__(
        self,
        target: str = "uniform",
        indeg: InDegreeSource | None = None,
        alpha: float = config.NMMC_ALPHA,
        c_update_p: float = config.NMMC_C_UPDATE_P,
        n_seeds: int = 1,
        burn_in: int = 0,
        n_agents: int = 1,
        shared_c: bool = False,
    ) -> None:
        if target not in TARGETS:
            raise ValueError(f"target muss aus {TARGETS} sein, ist {target!r}")
        self.target = target
        self.indeg = indeg or IN_DEGREES["online"]()
        self.alpha = float(alpha)
        self.c_update_p = float(c_update_p)
        if not 0.0 < self.c_update_p <= 1.0:
            # p = 0 hiesse: c_t bleibt bei c_0 = 1 und erreicht das wahre c
            # nie. Der Walk laeuft dann zwar (gamma = min(1, b) ist weiterhin
            # < 1 moeglich), aber ohne jede Konvergenzgarantie -- das Paper
            # misst genau das und findet die TVD stehenbleibend.
            raise ValueError(
                f"c_update_p muss in (0, 1] liegen, ist {self.c_update_p}. "
                "Bei 0 erreicht c_t das wahre c nie und die QSD ist nicht mehr "
                "die Zielverteilung."
            )
        self.n_seeds = n_seeds
        self.burn_in = burn_in
        self.n_agents = int(n_agents)
        if self.n_agents < 1:
            raise ValueError(f"n_agents muss >= 1 sein, ist {self.n_agents}")
        self.shared_c = bool(shared_c)
        self.name = f"nmmc_{self.target}_{self.indeg.name}"

    def key(self) -> str:
        """Alles, was den Walk steuert -- Ziel und d--Quelle stecken im Namen."""
        return (f"{self.name}|a{self.alpha:g}|p{self.c_update_p:g}"
                f"|seeds{self.n_seeds}|burn{self.burn_in}"
                f"|k{self.n_agents}|sc{int(self.shared_c)}")

    def sample(self, oracle, stats: dict | None = None) -> list[Sample]:
        """Die Trajektorie. `stats` ist rein diagnostisch (nmmc_trace.py).

        Was der Lauf innen tut -- Annahmequote, Umverteilungen, wie weit c_t
        gekommen ist -- steht in keiner Ergebnisspalte: die Pipeline reicht vom
        Sampler nichts als die Samples weiter (estimators.pipeline.evaluate baut
        `extra` allein aus der Formel). Statt dafuer die Pipeline zu biegen,
        nimmt der Sampler hier ein optionales dict entgegen und fuellt es. Ohne
        Argument -- also auf dem Produktivweg -- kostet das je Schritt einen
        `is not None`-Vergleich.
        """
        if stats is not None:
            stats.update(steps=0, accepted=0, redistributed=0, dead_ends=0,
                         proposals=0, dinhat_one=0, gamma_sum=0.0, c_final=1.0,
                         agents=self.n_agents, agents_eff=0)
        alpha, p = self.alpha, self.c_update_p
        uniform_target = self.target == "uniform"
        indeg, rng = self.indeg, oracle.rng
        trace: list[Sample] = []
        current: list[Sample] = []
        # Unter diesem Anteil kann ein Agent nichts mehr ausrichten: sein
        # Einstieg kostet schon COST_RANDOM_NODE. Ohne die Kappung ginge bei
        # grossem K das ganze Budget in Seed-Ziehungen. Wie viele Agenten
        # wirklich liefen, steht als n_random_node in der Ergebnis-CSV.
        min_per_agent = (oracle.cost_random_node
                         + config.NMMC_MIN_STEPS_PER_AGENT * oracle.cost_neighbors)
        k = max(1, min(self.n_agents, int(oracle.budget // max(min_per_agent, 1e-9))))
        if stats is not None:
            stats["agents_eff"] = k
        try:
            # Geteilt ueber alle Agenten -- so verlangt es das Paper fuer die
            # d--Schaetzung (6.3). `expanded` muss mit: sonst zaehlte der
            # zweite Agent dieselben Kanten noch einmal als Beleg.
            # Frisch je *Lauf*, denn eine Instanz haelt keinen Zustand, sonst
            # liefe der zweite Lauf desselben Objekts vorgewaermt (s.
            # sampling.indegree).
            state = indeg.start(oracle)
            expanded: set = set()
            c = 1.0                       # c_0 = 1 (Algorithmus 2)
            for agent in range(k):
                # Der letzte Agent laeuft bis zum Budgetende -- damit ist
                # k = 1 exakt der Ein-Agenten-Fall von vorher.
                limit = (math.inf if agent == k - 1
                         else oracle.budget * (agent + 1) / k)
                current = []
                hist = WeightedHistory()   # eigene Historie je Agent (Abschnitt 5)
                if not self.shared_c:
                    c = 1.0                # eigenes c_t je Agent
                u = int(oracle.seed_nodes(self.n_seeds)[0])
                step = 0                   # eigene Zeitachse: w_k = (step+1)^alpha
                # Kein eigenes Schritt-Limit: jeder Durchlauf fragt neighbors()
                # und zahlt mindestens COST_CACHE_HIT, das Budget beendet den
                # Lauf also von selbst (s. oracles.base).
                while oracle.queries < limit:
                    # Auch beim Wiederbesuch gefragt: der Cache-Treffer kostet,
                    # sonst liefe ein Walk in bekanntem Gebiet gratis weiter -- und
                    # bei NMMC ist das der Regelfall, nicht die Ausnahme.
                    out = oracle.neighbors(u)
                    if u not in expanded:
                        # Nur beim Erstbesuch: sonst zaehlte jeder Wiederbesuch
                        # dieselben Kanten noch einmal als Beleg.
                        expanded.add(u)
                        indeg.observe(state, out)
                    d_in_u = indeg.get(state, oracle, u)

                    if step >= self.burn_in:
                        # degree traegt hier den EINGANGSgrad, s. Modul-Docstring
                        current.append(Sample(u, d_in_u, step, agent))
                        oracle.mark()  # fuer Budget-Zwischenstaende, s. oracles.base
                    # Die Historie bekommt jeden Schritt, auch den Burn-in: genau
                    # den rechnet w_k = k^alpha spaeter ohnehin klein.
                    hist.add(u, float(step + 1) ** alpha)
                    step += 1

                    if len(out) == 0:
                        # Sackgasse: Absorption mit Wahrscheinlichkeit 1.
                        if stats is not None:
                            stats["steps"] += 1
                            stats["dead_ends"] += 1
                        u = hist.draw(rng)
                        continue

                    v = int(out[rng.randrange(len(out))])
                    d_in_v = indeg.get(state, oracle, v) if uniform_target else d_in_u
                    b = len(out) / d_in_v
                    if rng.random() < p:
                        c = b if b > c else c
                    # gamma = min(1, b/c): mit b >= c ist r*c < b sicher erfuellt,
                    # das min braucht deshalb keinen eigenen Zweig.
                    take = rng.random() * c < b
                    if stats is not None:
                        stats["steps"] += 1
                        stats["proposals"] += 1
                        # d- == 1 heisst: der Nenner von b traegt keine Information.
                        # Ist das der Regelfall, ist b = d+(i) und die Kette wird zu
                        # P = A/c -- Korollar 3.3, die QSD ist dann die
                        # Eigenvektorzentralitaet (s. sampling.indegree).
                        stats["dinhat_one"] += (d_in_v == 1)
                        stats["gamma_sum"] += b / c if b < c else 1.0
                        stats["accepted" if take else "redistributed"] += 1
                        stats["c_final"] = max(stats["c_final"], c)
                    u = v if take else hist.draw(rng)
                trace.extend(current)
                current = []
        except BudgetExceeded:
            pass
        # Der abgebrochene Agent zaehlt mit -- wie bei sampling.durw.
        trace.extend(current)
        return trace
