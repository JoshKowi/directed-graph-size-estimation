"""Zentrale Konfiguration: Pfade und Default-Parameter des Experiments.

Schnittstelle:
    ROOT, ADJACENCIES_DIR, RESULTS_DIR, PLOTS_DIR, ADDITIONALS_DIR, NAME_INDEX_DIR
    DEFAULT_BUDGETS, DEFAULT_N_RUNS, DEFAULT_SEED, DEFAULT_BUDGET_METRIC, DEFAULT_VIEWS
    GRAPH_LABELS, graph_label(name), GRAPH_ALIASES, resolve_graph(name)
    unique_path(path)
    SEED_NODES, seed_nodes(graph), start_slug(node)
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Originale Adjazenzlisten (.pkl) -- werden direkt verwendet.
ADJACENCIES_DIR = ROOT / "adjacencies"

# Ergebnisse (eine CSV je Graph) und Plots.
RESULTS_DIR = ROOT / "data" / "results"
PLOTS_DIR = ROOT / "data" / "plots"

# Externe Namenslisten (Wikipedia-Titeldumps, Top-Wikidata-Entitaeten) -- die
# Rohdaten, aus denen sich eine Ziehung ohne Kenntnis von V speisen laesst.
# Beschrieben werden sie in namelists.SOURCES.
ADDITIONALS_DIR = ROOT / "additionals"

# Vorab gebaute Namensindizes: je (Graph, Quelle) ein int32-Array, das jeder
# Position der Liste eine Knoten-ID oder -1 zuordnet (build_name_index.py).
# Abgeleitet und jederzeit neu baubar, deshalb nicht versioniert.
NAME_INDEX_DIR = ROOT / "data" / "name_index"

# --- Anzeigenamen der Graphen ------------------------------------------
# Der technische Name ist und bleibt der Dateiname der Adjazenzliste: er steht
# in jedem Pfad, in jeder CSV-Spalte `graph` und in jedem CLI-Aufruf. Fuer
# Grafiken und READMEs ist er aber nichtssagend -- "adjacency_list_uni" sagt
# niemandem, dass es der Wissensgraph von GPT-4 ist.
#
# Deshalb hier eine reine Anzeigeschicht: Dateiname -> Beschriftung. Nichts
# davon beruehrt Dateinamen, Ergebnisspalten oder CLI-Argumente, es aendert
# nur, was im Bild steht. Wer einen Graphen umbenennen will, aendert genau
# diese eine Zeile; fehlt ein Eintrag, wird der Dateiname selbst benutzt.
#
# Die Kurzbeschreibungen stammen aus adjacencies/README.txt.
GRAPH_LABELS = {
    "Slashdot0811": "Slashdot (Nov 2008)",
    "adjacency_list_uni": "GPT-4 knowledge graph (with literals)",
    "gpt4_io": "GPT-4 knowledge graph (instances only)",
    "gpt4o_adj_from_dataset": "GPT-4o knowledge graph (with literals)",
    "gpt4o_io": "GPT-4o knowledge graph (instances only)",
    "wiki-topcats": "Wikipedia (top categories)",
}


def unique_path(path):
    """Freier Dateiname: `x.png`, sonst `x-2.png`, `x-3.png`, ...

    Erzeugte Bilder und Ergebnisse ueberschreiben nichts mehr. Ein zweiter
    Lauf mit anderen Parametern soll den ersten nicht stillschweigend
    ersetzen -- was einmal auf der Platte liegt, bleibt liegen, und was
    verglichen werden soll, liegt nebeneinander.
    """
    if not path.exists():
        return path
    for i in range(2, 10000):
        candidate = path.with_name(f"{path.stem}-{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Zu viele Varianten von {path}")


def graph_label(name: str) -> str:
    """Beschriftung eines Graphen; faellt auf den Dateinamen zurueck."""
    return GRAPH_LABELS.get(name, name)


# Kuerzel fuer die Kommandozeile: --graphs gpt-4o-io statt --graphs gpt4o_io.
# Wieder nur eine Eingabeschicht -- aufgeloest wird sofort auf den Dateinamen,
# gespeichert und beschriftet wird nie das Kuerzel.
GRAPH_ALIASES = {
    "slashdot": "Slashdot0811",
    "gpt-4": "adjacency_list_uni",
    "gpt-4-io": "gpt4_io",
    "gpt-4o": "gpt4o_adj_from_dataset",
    "gpt-4o-io": "gpt4o_io",
    "wiki": "wiki-topcats",
}


def resolve_graph(name: str) -> str:
    """Kuerzel -> Dateiname. Unbekanntes bleibt unveraendert (auch der
    Dateiname selbst funktioniert also weiterhin).

    `-` und `_` sind austauschbar: `gpt4o-io`, `gpt-4o-io` und `gpt4o_io`
    meinen alle dieselbe Basis. Sonst plottet `--graphs gpt4_io gpt4o-io`
    stillschweigend nur den ersten Graphen, weil der zweite Name auf keine
    Ergebnisspalte trifft.
    """
    key = name.strip().lower()
    if key in GRAPH_ALIASES:
        return GRAPH_ALIASES[key]
    flat = key.replace("-", "").replace("_", "")
    for alias, target in GRAPH_ALIASES.items():
        if alias.replace("-", "").replace("_", "") == flat:
            return target
    for target in GRAPH_LABELS:
        if target.lower().replace("-", "").replace("_", "") == flat:
            return target
    return name


# --- Feste Einstiegsknoten ---------------------------------------------
# Ein realer Crawler startet nicht bei einem gleichverteilt gezogenen Knoten
# (dafuer muesste er V schon kennen), sondern bei ein paar bekannten. Diese
# Listen sind dieses "bekannte" Wissen -- einmal festgelegt, nicht je Lauf neu
# gezogen, damit alle Laeufe und alle Graphen von derselben Stelle starten.
#
# Slashdot: fuenf mit random.Random(42) aus den 70 898 Knoten mit ausgehenden
# Kanten gezogene Knoten. Ihre kleinen Grade (1 bis 6) sind kein Versehen,
# sondern das, was gleichverteiltes Ziehen in einem schwanzlastigen Graphen
# liefert.
#
# GPT-Graphen: fuenf Entitaeten verschiedener Art, jede in *beiden* Basen als
# Schluessel vorhanden (Ausgangsgrad gpt4_io / gpt4o_io):
#   Vannevar Bush             34 /  26   die Saat-Entitaet beider Erhebungen
#   Isaac Newton              38 /  79   Wissenschaftler
#   United States of America  81 /  92   Land; von den Varianten die einzige
#                                        mit aehnlichem Grad in beiden Basen
#                                        ("United States" 62/913, "USA" 81/781)
#   Kurashiki                 33 /  37   mittelgrosse japanische Stadt
#   Katsushika Hokusai        25 /  80   Kuenstler; bewusst nicht Yoshitomo
#                                        Nara (13/148884) -- der ist in
#                                        gpt4o_io ein Ausreisser, siehe README
GPT_SEED_NODES = [
    "Vannevar Bush",
    "Isaac Newton",
    "United States of America",
    "Kurashiki",
    "Katsushika Hokusai",
]

SEED_NODES = {
    "Slashdot0811": [3285, 14758, 30177, 33136, 37446],
    "adjacency_list_uni": GPT_SEED_NODES,
    "gpt4_io": GPT_SEED_NODES,
    "gpt4o_adj_from_dataset": GPT_SEED_NODES,
    "gpt4o_io": GPT_SEED_NODES,
}


def seed_nodes(graph: str) -> list:
    """Einstiegsknoten eines Graphen; leer, wenn keine hinterlegt sind.

    Der *erste* Eintrag ist der Default-Einstieg von run_experiment.py -- bei
    den GPT-Basen also "Vannevar Bush", die Saat-Entitaet beider Erhebungen.
    """
    return list(SEED_NODES.get(resolve_graph(graph), ()))


def start_slug(node) -> str:
    """Dateinamens-Baustein fuer einen Einstiegsknoten: "Isaac Newton" ->
    "isaac-newton". Nur fuer Namen, die Ergebnisse voneinander trennen."""
    keep = [c.lower() if c.isalnum() else "-" for c in str(node)]
    return "-".join("".join(keep).split("-")).strip("-") or "start"


# --- Safety Margin ------------------------------------------------------
# Mindestabstand im Walk, ab dem zwei Samples als Kollision zaehlen duerfen.
# Aufeinanderfolgende Schritte eines Random Walks sind stark korreliert: u_i
# und u_{i+1} sind Nachbarn, u_i und u_{i+2} oft derselbe Knoten. Solche
# Treffer sagen nichts ueber |V|, verkleinern die Schaetzung aber systematisch.
#
# Anders als Thinning kostet der Margin fast keine Samples -- ausgelassen
# werden nur Paare (m*k von C(k,2), bei k=100 000 und m=10 also 0,02 %),
# nicht Ziehungen. Der Wert sollte in der Groessenordnung der
# Autokorrelationslaenge des Walks liegen; ueber die Estimator-Namen
# (rw-plain__restart__margin20) ist er je Lauf ueberschreibbar.
SAFETY_MARGIN = 10

# Zahl der Faenge fuer die Schnabel-Variante von Capture-Recapture. Mehr Faenge
# heisst: jeder einzelne ist kleiner (das Budget wird geteilt), dafuer gibt es
# mehr Wiederfang-Information. Wo das Optimum liegt, ist eine empirische Frage
# -- ueber den Estimator-Namen (capture-recapture__restart__schnabel8) ist der
# Wert je Lauf ueberschreibbar.
DEFAULT_CAPTURES = 4

# Sprunggewicht w von DURW (sampling.durw): der Walk springt mit
# Wahrscheinlichkeit w/(w + deg_Gu(v)). Groesseres w heisst haeufiger springen
# -- weniger Autokorrelation und bessere Abdeckung, dafuer geht mehr Budget in
# Spruenge (COST_RANDOM_NODE) statt in Schritte (COST_CACHE_HIT beim
# Wiederbesuch). w -> 0 ergibt einen reinen Random Walk auf G_u, w -> unendlich
# gleichverteiltes Ziehen. Ribeiro & Towsley untersuchen w zwischen 0,1 und 10.
DURW_JUMP_WEIGHT = 1.0

# w-Werte, fuer die eigene Registry-Eintraege entstehen (estimators/__init__.py:
# wis-durw__<jump>__w<W>__margin). Logarithmisch gestuft und genau sieben Stueck
# -- plot_results.py stellt hoechstens acht Estimators je Bild dar.
#
# Der Sweep ist noetig, weil DURW_JUMP_WEIGHT kein guter Universalwert ist: wie
# oft DURW springt, haengt an der Struktur des gerichteten Graphen. Auf gpt4o_io
# (53 % Sackgassen) springt es bei w=1 schon zu 28 % und trifft |V|; auf gpt4_io
# (10 % Sackgassen) nur zu 15 % und schaetzt um die Haelfte zu klein. Erst
# groesseres w holt das auf (gemessen bei Budget 0,01: w=1 -> 0,178,
# w=100 -> 0,683 bei 94 % Spruengen).
DURW_JUMP_WEIGHTS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)

# --- NMMC (sampling.nmmc) ----------------------------------------------
# alpha der Gewichtsfolge w_k = k^alpha, mit der die Umverteilung ihre eigene
# Historie gewichtet. Es ist eine *Gedaechtnislaenge*: alpha = 0 zieht
# gleichverteilt aus der bisherigen Besuchsfolge (mit Vielfachheit -- der Fall,
# den sampling.dead_ends.HistoryJump abbildet), grosses alpha zieht fast nur
# zuletzt besuchte Knoten. Mehr Gewicht auf zuletzt Besuchtem heisst schnellere
# Diffusion, aber auch weniger Rueckgriff auf die alte Historie, die die
# Zielverteilung erst aufbaut -- das Paper findet je nach Graph und Ziel-QSD
# ein anderes Optimum zwischen beidem.
NMMC_ALPHA = 1.0

# alpha-Werte, fuer die eigene Registry-Eintraege entstehen
# (estimators/__init__.py: nmmc-uni__<indeg>__a<A>__margin). Genau vier Stueck:
# zusammen mit den vier Eintraegen im Thinning-Slot trifft
# "--match nmmc-uni__online__" damit exakt die acht Kurven, die
# plotting.style.color_for hergibt.
NMMC_ALPHAS = (0.0, 1.0, 3.0, 10.0)

# p aus Algorithmus 2 des Papers: mit dieser Wahrscheinlichkeit zieht der Walk
# die laufende Normierung c_t auf das gerade gesehene b_ij hoch, sonst laesst er
# sie stehen (und nimmt den Zug dann sicher an, weil gamma = min(1, b/c) = 1
# wird). Kleines p verlangsamt das Wachstum von c_t bewusst: sonst zieht ein
# einziges grosses b_ij die Annahmequote fuer alle folgenden Schritte nach
# unten, der Walk verteilt nur noch zwischen bekannten Knoten um und diffundiert
# nicht mehr. Das Paper misst p = 0,01 als besten Wert; erst p = 0 waere falsch,
# dort erreicht c_t das wahre c nie und die Konvergenzgarantie faellt.
NMMC_C_UPDATE_P = 0.01

# Agentenzahlen, fuer die eigene Registry-Eintraege entstehen
# (estimators/__init__.py: wis-nmmc__<indeg>__k<K>__margin). Das Paper faehrt
# in *jeder* Simulation 100 bis 10^4 Agenten -- ein einzelner Agent kommt dort
# nicht vor. Anders als dort teilen sich die Agenten hier das Budget; geteilt
# sind ausserdem Cache und die Online-Schaetzung des Eingangsgrades, eigen
# bleibt je Agent die Historie (siehe sampling.nmmc).
#
# Gemessen auf Slashdot0811 gerichtet bei Budget 20 %, Ziel pi ~ d-:
# 1 Agent 0,407 -- 10: 0,500 -- 50: 0,630 -- 200: 0,771. Der Gewinn kommt
# kaum aus der Abdeckung (+15 %), sondern daraus, dass Scheinkollisionen
# wegfallen: ein festsitzender Einzelagent besucht dieselben paar tausend
# Knoten tausendfach, und der Kollisionsschaetzer zaehlt das als Treffer.
NMMC_AGENTS = (1, 10, 100, 1000)

# Untergrenze fuer die Aufteilung: so viele volle Nachbarabfragen muss ein
# Agent zusaetzlich zu seinem Einstieg noch bezahlen koennen. Darunter ginge
# das ganze Budget in Seed-Ziehungen -- auf Slashdot waeren es bei Budget
# 0,1 % (77 Einheiten) und K = 1000 gerade 0,077 Einheiten je Agent. Der
# Sampler kappt K deshalb; wie viele Agenten wirklich liefen, steht als
# n_random_node in der Ergebnis-CSV (s. sampling.nmmc).
NMMC_MIN_STEPS_PER_AGENT = 5

# Partnergraph fuer die kreuzweise In-Grad-Schaetzung
# (sampling.indegree.CrossInDegree): fuer einen Lauf auf gpt4_io liefert
# gpt4o_io die Eingangsgrade und umgekehrt. Die beiden GPT-Basen teilen sich
# den Schluesselraum -- es sind Entitaetsnamen, keine IDs -- und beschreiben
# dieselbe Welt aus zwei Erhebungen. Damit ist der Partner *externes* Wissen
# ueber die Entitaeten, aber keine Kenntnis der Knotenmenge des geschaetzten
# Graphen: dieselbe Begruendung, die auch die Namenslisten real umsetzbar
# macht.
#
# Grenze des Arguments: die Paare sind sich aehnlich, weil sie aus verwandten
# Modellen stammen. Der Partner ist deshalb ein *guter* Prior, aber kein
# Beleg, dass beliebiges Fremdwissen so gut traegt.
CROSS_GRAPHS = {
    "gpt4_io": "gpt4o_io",
    "gpt4o_io": "gpt4_io",
    "adjacency_list_uni": "gpt4o_adj_from_dataset",
    "gpt4o_adj_from_dataset": "adjacency_list_uni",
}


def cross_graph(name: str) -> str | None:
    """Partnergraph, oder None wenn keiner hinterlegt ist."""
    return CROSS_GRAPHS.get(resolve_graph(name))


# Vorab gebaute In-Grad-Indizes: je (Graph, Partner) ein int32-Array, das zu
# jeder Knoten-ID den Eingangsgrad derselben Entitaet im Partnergraphen haelt
# (-1 = dort nicht vorhanden). Abgeleitet und jederzeit neu baubar
# (build_indeg_index.py), deshalb nicht versioniert.
INDEG_INDEX_DIR = ROOT / "data" / "indeg_index"


# Budgets relativ zur wahren Graph-Groesse |V|, z.B. 0.001 == 0.1 %.
# Fuer grosse Graphen siehe DEFAULT_BUDGETS_LARGE weiter unten.
DEFAULT_BUDGETS = (0.001, 0.005, 0.01, 0.05, 0.10, 0.20)

# Wiederholungen je (Estimator, Budget).
DEFAULT_N_RUNS = 10

# Besuchs-CSV (nur mit --visits): ab dieser Groesse faengt die naechste
# Teildatei an -- <graph>__...visits.csv, dann ...visits.2.csv, .3.csv, ...
# Verhindert die eine unbegrenzt wachsende Datei und das Neu-Einlesen mehrerer
# GB je Lauf (siehe experiment/results.py: append_visits).
VISITS_MAX_BYTES = 2_000_000_000        # ~2 GB

# Ab dieser Knotenzahl gilt ein Graph als gross: dort faellt das 20-%-Budget
# weg. Grund ist reine Rechenzeit -- auf Slashdot0811 entfallen 63 % aller
# Walk-Schritte allein auf dieses eine Budget, und die Kosten skalieren mit
# |V|. Mit --budgets laesst es sich jederzeit wieder anfordern.
LARGE_GRAPH_NODES = 1_000_000
DEFAULT_BUDGETS_LARGE = (0.001, 0.005, 0.01, 0.05, 0.10)

# Prozesse fuer die (Budget, Estimator, Lauf)-Schleife. Der Graph wird dabei
# nicht kopiert (siehe experiment.runner), zusaetzlicher Speicher faellt also
# kaum an. 1 = sequentiell.
DEFAULT_N_JOBS = 8

DEFAULT_SEED = 42

# Kantensichten, auf denen jeder Estimator laufen soll (siehe graphs.views).
# "undirected" symmetrisiert den Graphen und kostet zusaetzlichen Speicher.
DEFAULT_VIEWS = ("directed", "undirected")

# --- Kostenmodell des Oracles ------------------------------------------
# Drei Zugriffsarten, jede mit eigenem Preis in "Query-Einheiten":
#   random_node -- "gib mir einen zufaelligen Knoten"
#   neighbors   -- "gib mir die Nachbarn von u" (erster Zugriff auf u)
#   cache_hit   -- dieselbe Frage nochmal, aus dem eigenen Cache beantwortet
# Die ersten beiden sind in der Praxis zwei verschiedene Anfragen nach aussen
# und nicht zwingend gleich teuer. Sind die Einstiegsknoten fest bekannt, ist
# der Zufallszugriff faktisch gratis -- dann COST_RANDOM_NODE = 0 setzen.
COST_RANDOM_NODE = 1
COST_NEIGHBORS = 1

# Preis eines Fehlschlags beim Ziehen aus einer externen Namensliste
# (oracles.name_list): ein Name, den die Liste kennt, den der Graph aber nicht
# enthaelt. Solche Zuege sind der Preis dafuer, V *nicht* zu kennen -- die Liste
# passt nie genau auf den Graphen. Bei einer Trefferquote von 13 % (gpt4_io
# gegen enwiki) kostet ein Sample 1 + rund 6,5 * COST_DRAW_MISS.
#
# Muss > 0 sein: bei 0 dreht die Ziehschleife auf einem Index ohne Treffer
# endlos -- dasselbe Argument wie bei COST_CACHE_HIT weiter unten.
COST_DRAW_MISS = 1.0

# Schritte nach einem Treffer, bevor der Knoten in die Stichprobe darf
# (oracles.name_list). 0 = der gezogene Knoten selbst ist das Sample. Gedacht
# gegen die Verzerrung der Liste: getroffene Knoten haben deutlich hoeheren
# Grad als verfehlte (gpt4o_io/enwiki: 10,70 gegen 3,55), ein paar Schritte
# sollen davon wegfuehren. Sie fuehren dafuer die Verzerrung eines
# Random-Walk-Schritts ein -- wo das Optimum liegt, ist eine empirische Frage.
DEFAULT_DRAW_BURN_IN = 0

# Burn-in-Werte, fuer die eigene Registry-Eintraege entstehen
# (namelist-<quelle>__b<n>, durw-<quelle>__b<n>__margin). Klein gehalten: jeder
# Schritt kostet eine Nachbarabfrage, und nach wenigen Schritten dominiert
# ohnehin die Verteilung des Walks statt die der Liste.
DRAW_BURN_INS = (0, 1, 2, 5, 10)

# Listenlaengen, fuer die eigene Registry-Eintraege entstehen
# (durw-<quelle>__n<N>__b<B>__margin). Die Quellen sind nach Relevanz sortiert
# -- top-q nach QRank, die In-Grad-Listen nach Eingangsgrad -- ein Praefix ist
# dort also "die n prominentesten Entitaeten".
#
# Der Namensindex ist positionsbasiert, das Abschneiden damit ein Slice: es
# braucht je (Graph, Quelle) *einen* Index, nicht einen je Laenge.
DRAW_LIMITS = (1_000, 10_000, 100_000, 1_000_000)

# Anteile |S|/|V| in Prozent, fuer die der Sprung auf eine Zufallsteilmenge
# eigene Registry-Eintraege bekommt (durw-/durwset-/durwhist-rand<P>__b0__margin,
# oracles.random_subset). Andere Anteile loest estimators.build() zur Laufzeit
# auf. 100 ist die Gegenprobe: S = V, also der gleichverteilte Sprung des Papers.
JUMP_SUBSET_PERCENTS = (1, 5, 10, 25, 50, 100)

# Ein Cache-Treffer ist billig, aber nicht gratis: ein realer Crawler haelt die
# einmal geholte Nachbarschaft, muss sie aber weiterhin nachschlagen. Der Preis
# ist der einzige Regler fuer ein sonst unloesbares Problem: bei Preis 0 laeuft
# ein Walk, der sich in einer kleinen, bereits bekannten Region verfaengt,
# beliebig lange gratis weiter und sammelt beliebig viele wertlose,
# hochkorrelierte Samples. Mit einem Preis > 0 terminiert das Budget jeden Lauf
# von selbst, und jeder Estimator gibt seine 100 % aus -- erst dadurch ist
# "genutztes Budget" ueberhaupt eine vergleichbare Groesse.
#
# Die Decke fuer einen vollstaendig gecachten Walk ist budget / COST_CACHE_HIT,
# hier also 50 x Budget. Der Wert bestimmt damit die Groessenordnung der
# Schaetzung fuer verfangene Walks mit und gehoert in jede Ergebnisdarstellung.
# Gemessen auf Slashdot0811 gerichtet, Budget 3868, dead_end="history":
#
#   COST_CACHE_HIT   Schritte   Schaetzung/|V|
#   0.02               81 728   0.00009
#   0.05               34 057   0.00020
#   0.20               10 223   0.00079
#   1.00                3 867   0.00807   (== kein Cache-Rabatt)
COST_CACHE_HIT = 0.02

# Was das Budget begrenzt. Zulaessig ist nur noch "queries": die bezahlten,
# gewichteten Kosten des Modells oben. Weil jeder Zugriff einen Preis > 0 hat,
# terminiert diese Metrik jeden Lauf von selbst -- ein globales Aufruf-Limit
# gibt es deshalb nicht mehr.
#
# "unique_nodes" ist als *Limit* entfallen: die Zahl waechst bei einem Walk in
# bereits bekanntem Gebiet gar nicht mehr, der Lauf wuerde nie enden. Als
# Statistik steht sie weiterhin in der Ergebnis-CSV (`unique_nodes_used`).
DEFAULT_BUDGET_METRIC = "queries"
