# data/results -- Rohergebnisse

*Automatisch erzeugt von `Code/provenance.py` -- nicht von Hand aendern.*

| Daten vom | 2026-09-08 17:22 |
|---|---|
| Code-Fingerabdruck | `e884bdb68ddc` |
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

### `Slashdot0811__budget_breakdown.csv`

36 Zeilen.

### `Slashdot0811__estimates.csv`

Schaetzungen fuer **Slashdot (Nov 2008)** (`Slashdot0811`), 832 Zeilen (= Estimator x View x Budget x Lauf).

- Views: directed, undirected
- Budgets: 0.0001, 0.0002, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2 (relativ zu |V| = 77 360)
- Laeufe je Punkt: 10
- Estimators: capture-recapture__uniform, capture-recapture__uniform__cross-wis, nmmc-uni__exact__margin, nmmc-uni__online__margin, uniform-collision, uniform-collision__weighted, wis-durw__uniform__margin, wis-durw__uniform__w0.1__margin, wis-durw__uniform__w0.3__margin, wis-durw__uniform__w100__margin, wis-durw__uniform__w10__margin, wis-durw__uniform__w1__margin, wis-durw__uniform__w30__margin, wis-durw__uniform__w3__margin, wis-katzir__rw-backtrack__margin, wis-katzir__rw-history__margin, wis-katzir__rw-restart__margin, wis-nmmc__exact__margin, wis-nmmc__online__margin
- Seed: 42
- Einstieg: 3285
- Abbruchgrund: {'budget': 832}

Erzeugt mit:

```bash
python run_experiment.py --graphs Slashdot0811 \
    --estimators capture-recapture__uniform capture-recapture__uniform__cross-wis nmmc-uni__exact__margin nmmc-uni__online__margin uniform-collision uniform-collision__weighted wis-durw__uniform__margin wis-durw__uniform__w0.1__margin wis-durw__uniform__w0.3__margin wis-durw__uniform__w100__margin wis-durw__uniform__w10__margin wis-durw__uniform__w1__margin wis-durw__uniform__w30__margin wis-durw__uniform__w3__margin wis-katzir__rw-backtrack__margin wis-katzir__rw-history__margin wis-katzir__rw-restart__margin wis-nmmc__exact__margin wis-nmmc__online__margin \
    --views directed undirected
```

### `Slashdot0811__nmmc_trace.csv`

NMMC-Diagnose auf **Slashdot (Nov 2008)** (`Slashdot0811`), 20 Zeilen (= View x d--Quelle x Ziel x Lauf). Misst, was die Ergebnis-CSV nicht hergibt: `acc_rate` (angenommen / vorgeschlagen), `frac_dinhat_1`, `c_final`/`c_max`, `redist_share`, `dead_ends`. Dazu die Schranken, die auch ohne dieses Skript ablesbar waeren (`acc_rate_lower`, `redist_share_upper`) -- ihr Abstand zum gemessenen Wert ist der Grund fuer das Skript.

- Views: directed
- d--Quelle: exact, online
- Ziel: indeg, uniform
- alpha: 1
- Seed: 42

Erzeugt mit `python nmmc_trace.py --graph Slashdot0811` (`Code/nmmc_trace.py`).

### `Slashdot0811__view_comparison.csv`

Gepaarter Vergleich der Kantensichten fuer **Slashdot (Nov 2008)** (`results.compare_views`). Entsteht beim Plotten.

### `Slashdot0811__visits.csv`

Besuchshaeufigkeit je Original-Knotenname fuer **Slashdot (Nov 2008)** (Seed 42), 408 449 Zeilen. Faellt beim selben Lauf ab wie die Schaetzungen (`--no-visits` schaltet sie aus).

### `gpt4_io__budget_breakdown.csv`

28 Zeilen.

### `gpt4_io__estimates.csv`

Schaetzungen fuer **GPT-4 knowledge graph (instances only)** (`gpt4_io`), 2 750 Zeilen (= Estimator x View x Budget x Lauf).

- Views: directed, undirected
- Budgets: 1e-05, 2e-05, 5e-05, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1 (relativ zu |V| = 6 492 586)
- Laeufe je Punkt: 10
- Estimators: capture-recapture__uniform, rw-plain__backtrack__none, uniform-collision, wis-durw__uniform__margin, wis-durw__uniform__w0.1__margin, wis-durw__uniform__w0.3__margin, wis-durw__uniform__w100__margin, wis-durw__uniform__w10__margin, wis-durw__uniform__w1__margin, wis-durw__uniform__w30__margin, wis-durw__uniform__w3__margin, wis-katzir__rw-backtrack, wis-katzir__rw-backtrack__margin, wis-katzir__rw-backtrack__shifted, wis-katzir__rw-backtrack__simple, wis-katzir__rw-history__margin, wis-katzir__rw-restart, wis-katzir__rw-restart__margin, wis-katzir__rw-restart__margin16, wis-katzir__rw-restart__margin2, wis-katzir__rw-restart__margin32, wis-katzir__rw-restart__margin4, wis-katzir__rw-restart__margin64, wis-katzir__rw-restart__margin8, wis-katzir__rw-restart__shifted16, wis-katzir__rw-restart__shifted2, wis-katzir__rw-restart__shifted32, wis-katzir__rw-restart__shifted4, wis-katzir__rw-restart__shifted64, wis-katzir__rw-restart__shifted8
- Seed: 42
- Einstieg: Vannevar Bush
- Abbruchgrund: {'budget': 2750}

Erzeugt mit:

```bash
python run_experiment.py --graphs gpt4_io \
    --estimators capture-recapture__uniform rw-plain__backtrack__none uniform-collision wis-durw__uniform__margin wis-durw__uniform__w0.1__margin wis-durw__uniform__w0.3__margin wis-durw__uniform__w100__margin wis-durw__uniform__w10__margin wis-durw__uniform__w1__margin wis-durw__uniform__w30__margin wis-durw__uniform__w3__margin wis-katzir__rw-backtrack wis-katzir__rw-backtrack__margin wis-katzir__rw-backtrack__shifted wis-katzir__rw-backtrack__simple wis-katzir__rw-history__margin wis-katzir__rw-restart wis-katzir__rw-restart__margin wis-katzir__rw-restart__margin16 wis-katzir__rw-restart__margin2 wis-katzir__rw-restart__margin32 wis-katzir__rw-restart__margin4 wis-katzir__rw-restart__margin64 wis-katzir__rw-restart__margin8 wis-katzir__rw-restart__shifted16 wis-katzir__rw-restart__shifted2 wis-katzir__rw-restart__shifted32 wis-katzir__rw-restart__shifted4 wis-katzir__rw-restart__shifted64 wis-katzir__rw-restart__shifted8 \
    --views directed undirected
```

### `gpt4_io__nmmc_trace.csv`

NMMC-Diagnose auf **GPT-4 knowledge graph (instances only)** (`gpt4_io`), 24 Zeilen (= View x d--Quelle x Ziel x Lauf). Misst, was die Ergebnis-CSV nicht hergibt: `acc_rate` (angenommen / vorgeschlagen), `frac_dinhat_1`, `c_final`/`c_max`, `redist_share`, `dead_ends`. Dazu die Schranken, die auch ohne dieses Skript ablesbar waeren (`acc_rate_lower`, `redist_share_upper`) -- ihr Abstand zum gemessenen Wert ist der Grund fuer das Skript.

- Views: directed
- d--Quelle: cross-one, cross-online, exact, online
- Ziel: indeg, uniform
- alpha: 1
- Seed: 42

Erzeugt mit `python nmmc_trace.py --graph gpt4_io` (`Code/nmmc_trace.py`).

### `gpt4_io__view_comparison.csv`

Gepaarter Vergleich der Kantensichten fuer **GPT-4 knowledge graph (instances only)** (`results.compare_views`). Entsteht beim Plotten.

### `gpt4_io__visits.csv`

Besuchshaeufigkeit je Original-Knotenname fuer **GPT-4 knowledge graph (instances only)** (Seed 42), 36 876 367 Zeilen. Faellt beim selben Lauf ab wie die Schaetzungen (`--no-visits` schaltet sie aus).

### `gpt4o_adj_from_dataset__estimates.csv`

Schaetzungen fuer **GPT-4o knowledge graph (with literals)** (`gpt4o_adj_from_dataset`), 120 Zeilen (= Estimator x View x Budget x Lauf).

- Views: directed, undirected
- Budgets: 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02 (relativ zu |V| = 15 723 674)
- Laeufe je Punkt: 10
- Estimators: uniform_collision
- Seed: 42
- Einstieg: gleichverteilt
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

Schaetzungen fuer **GPT-4o knowledge graph (instances only)** (`gpt4o_io`), 2 720 Zeilen (= Estimator x View x Budget x Lauf).

- Views: directed, undirected
- Budgets: 1e-05, 2e-05, 5e-05, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1 (relativ zu |V| = 5 693 001)
- Laeufe je Punkt: 10
- Estimators: rw-plain__backtrack__none, uniform-collision, wis-durw__uniform__margin, wis-durw__uniform__w0.1__margin, wis-durw__uniform__w0.3__margin, wis-durw__uniform__w100__margin, wis-durw__uniform__w10__margin, wis-durw__uniform__w1__margin, wis-durw__uniform__w30__margin, wis-durw__uniform__w3__margin, wis-katzir__rw-backtrack, wis-katzir__rw-backtrack__margin, wis-katzir__rw-backtrack__shifted, wis-katzir__rw-backtrack__simple, wis-katzir__rw-history__margin, wis-katzir__rw-restart, wis-katzir__rw-restart__margin16, wis-katzir__rw-restart__margin2, wis-katzir__rw-restart__margin32, wis-katzir__rw-restart__margin4, wis-katzir__rw-restart__margin64, wis-katzir__rw-restart__margin8, wis-katzir__rw-restart__shifted16, wis-katzir__rw-restart__shifted2, wis-katzir__rw-restart__shifted32, wis-katzir__rw-restart__shifted4, wis-katzir__rw-restart__shifted64, wis-katzir__rw-restart__shifted8
- Seed: 42
- Einstieg: Vannevar Bush
- Abbruchgrund: {'budget': 2720}

Erzeugt mit:

```bash
python run_experiment.py --graphs gpt4o_io \
    --estimators rw-plain__backtrack__none uniform-collision wis-durw__uniform__margin wis-durw__uniform__w0.1__margin wis-durw__uniform__w0.3__margin wis-durw__uniform__w100__margin wis-durw__uniform__w10__margin wis-durw__uniform__w1__margin wis-durw__uniform__w30__margin wis-durw__uniform__w3__margin wis-katzir__rw-backtrack wis-katzir__rw-backtrack__margin wis-katzir__rw-backtrack__shifted wis-katzir__rw-backtrack__simple wis-katzir__rw-history__margin wis-katzir__rw-restart wis-katzir__rw-restart__margin16 wis-katzir__rw-restart__margin2 wis-katzir__rw-restart__margin32 wis-katzir__rw-restart__margin4 wis-katzir__rw-restart__margin64 wis-katzir__rw-restart__margin8 wis-katzir__rw-restart__shifted16 wis-katzir__rw-restart__shifted2 wis-katzir__rw-restart__shifted32 wis-katzir__rw-restart__shifted4 wis-katzir__rw-restart__shifted64 wis-katzir__rw-restart__shifted8 \
    --views directed undirected
```

### `gpt4o_io__view_comparison.csv`

Gepaarter Vergleich der Kantensichten fuer **GPT-4o knowledge graph (instances only)** (`results.compare_views`). Entsteht beim Plotten.

### `gpt4o_io__visits.csv`

Besuchshaeufigkeit je Original-Knotenname fuer **GPT-4o knowledge graph (instances only)** (Seed 42), 22 520 796 Zeilen. Faellt beim selben Lauf ab wie die Schaetzungen (`--no-visits` schaltet sie aus).

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
| `code` | Fingerabdruck des Codes, der die Zeile erzeugt hat |
| `seed` | Zufallsstrom des Laufs (siehe Dateiname) |
| `start_node` | Einstiegsknoten des Crawls (`config.SEED_NODES`) |
| `nested` | Budget aus einem gemeinsamen Lauf abgelesen (s.u.) |
| `walk_group` | Zeilen mit gleichem Wert stammen aus *einem* Walk |
| `extra_*` | verfahrensspezifisch, z.B. `extra_n_samples` |

Ist `nested` wahr, stammen alle Budgets einer Laufnummer aus *einem*
Lauf (`--checkpoint-budgets`): die Stichprobe wurde dort abgeschnitten,
wo ein eigenstaendiger Lauf mit dem kleineren Budget geendet haette.
Je Budget ist die Verteilung dieselbe -- die Punkte einer Laufnummer
sind aber ueber die Budgets *genestet* und nicht unabhaengig. `seconds`
steht dann vollstaendig beim groessten Budget, die kleineren tragen 0.
Besuchszaehler entstehen in diesem Modus nur fuer das groesste Budget.

Steht in `walk_group` ein Wert, haben sich mehrere Estimators einen
Walk geteilt (`--share-walks`): Thinning, Weighting und Formel sind
reine Nachbearbeitung derselben Trajektorie. Ihre Zeilen sind damit
gepaart -- fuer den Vergleich *zwischen* ihnen ein Gewinn, aber sie
sind keine unabhaengigen Beobachtungen. `seconds` steht auch hier nur
beim ersten Estimator der Gruppe.

Ergebnisse werden **angehaengt, nicht ueberschrieben**: ein zweiter
Aufruf rechnet nur, was noch fehlt. Stehen in einer Datei mehrere
Werte in `code`, stammen ihre Zeilen aus verschiedenen Codeversionen --
das ist erlaubt, solange die Aenderung den Verlauf nicht beruehrt hat.
War sie es doch, gehoeren die alten Zeilen mit `--deprecate` beiseite
(nach `data/results/deprecated/<Zeit>__<Fingerabdruck>/`).

Steht in `stopped_by` etwas anderes als `budget`, hat nicht das
Kostenmodell den Lauf beendet -- die Zahlen sind dann mit Vorsicht zu
lesen. Siehe `oracles/base.py`.
