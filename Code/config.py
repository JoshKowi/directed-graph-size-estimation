"""Zentrale Konfiguration: Pfade und Default-Parameter des Experiments.

Schnittstelle (nur Konstanten):
    ROOT, ADJACENCIES_DIR, RESULTS_DIR, PLOTS_DIR
    DEFAULT_BUDGETS, DEFAULT_N_RUNS, DEFAULT_SEED, DEFAULT_BUDGET_METRIC, DEFAULT_VIEWS
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Originale Adjazenzlisten (.pkl) -- werden direkt verwendet.
ADJACENCIES_DIR = ROOT / "adjacencies"

# Ergebnisse (eine CSV je Graph) und Plots.
RESULTS_DIR = ROOT / "data" / "results"
PLOTS_DIR = ROOT / "data" / "plots"

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
