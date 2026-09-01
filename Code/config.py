"""Zentrale Konfiguration: Pfade und Default-Parameter des Experiments.

Schnittstelle:
    ROOT, ADJACENCIES_DIR, RESULTS_DIR, PLOTS_DIR
    DEFAULT_BUDGETS, DEFAULT_N_RUNS, DEFAULT_SEED, DEFAULT_BUDGET_METRIC, DEFAULT_VIEWS
    GRAPH_LABELS, graph_label(name), GRAPH_ALIASES, resolve_graph(name)
    SEED_NODES, seed_nodes(graph), start_slug(node)
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Originale Adjazenzlisten (.pkl) -- werden direkt verwendet.
ADJACENCIES_DIR = ROOT / "adjacencies"

# Ergebnisse (eine CSV je Graph) und Plots.
RESULTS_DIR = ROOT / "data" / "results"
PLOTS_DIR = ROOT / "data" / "plots"

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
    Dateiname selbst funktioniert also weiterhin)."""
    return GRAPH_ALIASES.get(name.strip().lower(), name)


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


# Budgets relativ zur wahren Graph-Groesse |V|, z.B. 0.001 == 0.1 %.
# Fuer grosse Graphen siehe DEFAULT_BUDGETS_LARGE weiter unten.
DEFAULT_BUDGETS = (0.001, 0.005, 0.01, 0.05, 0.10, 0.20)

# Wiederholungen je (Estimator, Budget).
DEFAULT_N_RUNS = 10

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
