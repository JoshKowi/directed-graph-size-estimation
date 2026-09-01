# data/results -- Rohergebnisse

*Automatisch erzeugt von `Code/provenance.py` -- nicht von Hand aendern.*

| Daten vom | 2026-08-26 15:47 |
|---|---|
| Code-Fingerabdruck | `cb43125105ed` |
| Budget-Metrik | `queries` |
| Preise | random_node 1, neighbors 1, cache_hit 0.02 |
| Budgets (Default) | 0.001, 0.005, 0.01, 0.05, 0.1, 0.2 |
| Laeufe je Punkt | 10 |
| Seed (Default) | 42 -- je Datei unten angegeben |

Der Fingerabdruck ist ein SHA-256 ueber alle `.py` unter `Code/`. Zwei
Ergebnisse mit demselben Fingerabdruck stammen aus identischem Code.

Der Commit steht bewusst *nicht* hier: eine versionierte Datei, die den
aktuellen Commit nennt, kann nie stimmen -- beim Committen aendert sich
genau der Hash, den sie angibt. Um den passenden Stand zu finden, einen
Commit auschecken und `python Code/provenance.py` laufen lassen; stimmt
der Fingerabdruck ueberein, ist es der richtige.

Die CSVs selbst sind **nicht** im Repository (gross und aus dem Code
reproduzierbar) -- diese Datei haelt fest, woher sie stammen.

## Dateien

### `Slashdot0811__estimates.csv`

Schaetzungen fuer **Slashdot (Nov 2008)** (`Slashdot0811`), 120 Zeilen (= Estimator x View x Budget x Lauf).

- Views: directed, undirected
- Budgets: 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02 (relativ zu |V| = 77 360)
- Laeufe je Punkt: 10
- Estimators: uniform_collision
- Seed: 42
- Abbruchgrund: {'budget': 120}

Erzeugt mit:

```bash
python run_experiment.py --graphs Slashdot0811 \
    --estimators uniform_collision \
    --views directed undirected
```

### `Slashdot0811__view_comparison.csv`

Gepaarter Vergleich der Kantensichten fuer **Slashdot (Nov 2008)** (`results.compare_views`). Entsteht beim Plotten.

### `Slashdot0811__visits.csv`

Besuchshaeufigkeit je Original-Knotenname fuer **Slashdot (Nov 2008)** (Seed 42), 28 846 Zeilen. Faellt beim selben Lauf ab wie die Schaetzungen (`--no-visits` schaltet sie aus).

### `gpt4o_adj_from_dataset__estimates.csv`

Schaetzungen fuer **GPT-4o knowledge graph (with literals)** (`gpt4o_adj_from_dataset`), 120 Zeilen (= Estimator x View x Budget x Lauf).

- Views: directed, undirected
- Budgets: 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02 (relativ zu |V| = 15 723 674)
- Laeufe je Punkt: 10
- Estimators: uniform_collision
- Seed: 42
- Abbruchgrund: {'budget': 120}

Erzeugt mit:

```bash
python run_experiment.py --graphs gpt4o_adj_from_dataset \
    --estimators uniform_collision \
    --views directed undirected
```

### `gpt4o_adj_from_dataset__view_comparison.csv`

Gepaarter Vergleich der Kantensichten fuer **GPT-4o knowledge graph (with literals)** (`results.compare_views`). Entsteht beim Plotten.

### `gpt4o_adj_from_dataset__visits.csv`

Besuchshaeufigkeit je Original-Knotenname fuer **GPT-4o knowledge graph (with literals)** (Seed 42), 5 860 816 Zeilen. Faellt beim selben Lauf ab wie die Schaetzungen (`--no-visits` schaltet sie aus).

### `gpt4o_io__estimates.csv`

Schaetzungen fuer **GPT-4o knowledge graph (instances only)** (`gpt4o_io`), 120 Zeilen (= Estimator x View x Budget x Lauf).

- Views: directed, undirected
- Budgets: 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02 (relativ zu |V| = 5 693 001)
- Laeufe je Punkt: 10
- Estimators: uniform_collision
- Seed: 42
- Abbruchgrund: {'budget': 120}

Erzeugt mit:

```bash
python run_experiment.py --graphs gpt4o_io \
    --estimators uniform_collision \
    --views directed undirected
```

### `gpt4o_io__view_comparison.csv`

Gepaarter Vergleich der Kantensichten fuer **GPT-4o knowledge graph (instances only)** (`results.compare_views`). Entsteht beim Plotten.

### `gpt4o_io__visits.csv`

Besuchshaeufigkeit je Original-Knotenname fuer **GPT-4o knowledge graph (instances only)** (Seed 42), 2 121 482 Zeilen. Faellt beim selben Lauf ab wie die Schaetzungen (`--no-visits` schaltet sie aus).

## Spalten der `__estimates.csv`

| Spalte | Bedeutung |
|---|---|
| `estimate`, `rel_error` | Schaetzung und relativer Fehler gegen `true_size` |
| `budget_rel`, `budget_abs` | Budget relativ zu \|V\| und absolut |
| `queries_used` | bezahlte, gewichtete Kosten (die Budget-Waehrung) |
| `cached_queries` | Nachbar-Abfragen aus dem Cache (Preis `COST_CACHE_HIT`) |
| `n_random_node`, `n_neighbors` | Zugriffe je Art zum vollen Preis |
| `unique_nodes_used` | verschiedene beruehrte Knoten (nur Statistik) |
| `stopped_by` | warum der Lauf endete -- normal `budget` |
| `seed` | Zufallsstrom des Laufs (siehe Dateiname) |
| `nested` | Budget aus einem gemeinsamen Lauf abgelesen (s.u.) |
| `extra_*` | verfahrensspezifisch, z.B. `extra_n_samples` |

Ist `nested` wahr, stammen alle Budgets einer Laufnummer aus *einem*
Lauf (`--checkpoint-budgets`): die Stichprobe wurde dort abgeschnitten,
wo ein eigenstaendiger Lauf mit dem kleineren Budget geendet haette.
Je Budget ist die Verteilung dieselbe -- die Punkte einer Laufnummer
sind aber ueber die Budgets *genestet* und nicht unabhaengig. `seconds`
steht dann vollstaendig beim groessten Budget, die kleineren tragen 0.
Besuchszaehler entstehen in diesem Modus nur fuer das groesste Budget.

Steht in `stopped_by` etwas anderes als `budget`, hat nicht das
Kostenmodell den Lauf beendet -- die Zahlen sind dann mit Vorsicht zu
lesen. Siehe `oracles/base.py`.
