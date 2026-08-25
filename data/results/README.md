# data/results -- Rohergebnisse

*Automatisch erzeugt von `Code/provenance.py` -- nicht von Hand aendern.*

| Daten vom | 2026-08-25 18:39 |
|---|---|
| Code-Fingerabdruck | `6d0bf589aecd` |
| Budget-Metrik | `queries` |
| Preise | random_node 1, neighbors 1, cache_hit 0.02 |
| Budgets (Default) | 0.001, 0.005, 0.01, 0.05, 0.1, 0.2 |
| Laeufe je Punkt | 10 |
| Seed | 42 |

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

Schaetzungen fuer **Slashdot0811**, 60 Zeilen (= Estimator x View x Budget x Lauf).

- Views: directed
- Budgets: 0.01, 0.05 (relativ zu |V| = 77 360)
- Laeufe je Punkt: 10
- Estimators: uniform_collision, wis-katzir__indep, wis-katzir__rw-restart
- Abbruchgrund: {'budget': 60}

Erzeugt mit:

```bash
python run_experiment.py --graphs Slashdot0811 \
    --estimators uniform_collision wis-katzir__indep wis-katzir__rw-restart \
    --views directed
```

### `Slashdot0811__view_comparison.csv`

Gepaarter Vergleich der Kantensichten fuer **Slashdot0811** (`results.compare_views`). Entsteht beim Plotten.

### `Slashdot0811__visits.csv`

Besuchshaeufigkeit je Original-Knotenname fuer **Slashdot0811**, 1 614 636 Zeilen. Faellt beim selben Lauf ab wie die Schaetzungen (`--no-visits` schaltet sie aus).

### `gpt4o_io__estimates.csv`

Schaetzungen fuer **gpt4o_io**, 200 Zeilen (= Estimator x View x Budget x Lauf).

- Views: directed, undirected
- Budgets: 0.001, 0.005, 0.01, 0.05, 0.1 (relativ zu |V| = 5 693 001)
- Laeufe je Punkt: 10
- Estimators: uniform_collision, wis-katzir__rw-history
- Abbruchgrund: {'budget': 200}

Erzeugt mit:

```bash
python run_experiment.py --graphs gpt4o_io \
    --estimators uniform_collision wis-katzir__rw-history \
    --views directed undirected
```

### `gpt4o_io__visits.csv`

Besuchshaeufigkeit je Original-Knotenname fuer **gpt4o_io**, 13 731 046 Zeilen. Faellt beim selben Lauf ab wie die Schaetzungen (`--no-visits` schaltet sie aus).

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
| `extra_*` | verfahrensspezifisch, z.B. `extra_n_samples` |

Steht in `stopped_by` etwas anderes als `budget`, hat nicht das
Kostenmodell den Lauf beendet -- die Zahlen sind dann mit Vorsicht zu
lesen. Siehe `oracles/base.py`.
