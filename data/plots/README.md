# data/plots -- erzeugte Grafiken

*Automatisch erzeugt von `Code/provenance.py` -- nicht von Hand aendern.*

| Daten vom | 2026-09-01 15:54 |
|---|---|
| Code-Fingerabdruck | `5d874fad6e4e` |
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

Jede Grafik zeigt pro Estimator und Budget die Spanne min..max ueber die
Laeufe plus den Median, y = Schaetzung/|V| (log), gestrichelt die wahre
Groesse bei 1.0. Die x-Achse nennt das Budget relativ und absolut.

Referenzreihe in allen `wis_*`/`deadend_*`-Grafiken: `uniform_collision`.

## Dateien

### `Slashdot0811__deadend_uis__directed.png`

UIS collision counting with a random walk -- dead-end strategies (directed)

- Graph: **Slashdot (Nov 2008)** (`Slashdot0811`)
- Seed: 42
- Einstieg: Default (config.SEED_NODES)
- Views: directed
- Estimators: uniform_collision, rw_plain__restart__none, rw_plain__backtrack__none, rw_plain__history__none

Erzeugt mit `python plot_wis.py --graphs Slashdot0811` (Definition in `Code/plot_wis.py`, Eintrag `deadend_uis__directed`).

### `Slashdot0811__thinning-without-weights.png`

Keine Definition in `plot_wis.FIGURES` gefunden -- vermutlich von Hand oder mit einer aelteren Codeversion erzeugt.

### `Slashdot0811__walk_diagnosis.png`

Diagnose eines Random Walks auf **Slashdot (Nov 2008)** (Seed 42): Leiter der Groessen, Abdeckungskurve, Besuche gegen Grad, meistbesuchte Entitaeten. Erzeugt mit `python diagnose_walk.py --graph Slashdot0811 --views directed undirected` (`Code/diagnose_walk.py`).

### `Slashdot0811__wis_indep__directed.png`

WIS (Katzir) -- independent degree-weighted draws (directed)

- Graph: **Slashdot (Nov 2008)** (`Slashdot0811`)
- Seed: 42
- Einstieg: Default (config.SEED_NODES)
- Views: directed
- Estimators: uniform_collision, wis-katzir__indep

Erzeugt mit `python plot_wis.py --graphs Slashdot0811` (Definition in `Code/plot_wis.py`, Eintrag `wis_indep__directed`).

### `Slashdot0811__wis_indep__undirected.png`

WIS (Katzir) -- independent degree-weighted draws (undirected)

- Graph: **Slashdot (Nov 2008)** (`Slashdot0811`)
- Seed: 42
- Einstieg: Default (config.SEED_NODES)
- Views: undirected
- Estimators: uniform_collision, wis-katzir__indep

Erzeugt mit `python plot_wis.py --graphs Slashdot0811` (Definition in `Code/plot_wis.py`, Eintrag `wis_indep__undirected`).

### `Slashdot0811__wis_rw__undirected.png`

WIS with a true random walk -- all dead-end strategies (undirected)

- Graph: **Slashdot (Nov 2008)** (`Slashdot0811`)
- Seed: 42
- Einstieg: Default (config.SEED_NODES)
- Views: undirected
- Estimators: uniform_collision, wis-katzir__rw-restart, wis-katzir__rw-backtrack, wis-katzir__rw-history

Erzeugt mit `python plot_wis.py --graphs Slashdot0811` (Definition in `Code/plot_wis.py`, Eintrag `wis_rw__undirected`).

### `Slashdot0811__wis_rw_history__views.png`

WIS with random walk (history) -- directed vs undirected

- Graph: **Slashdot (Nov 2008)** (`Slashdot0811`)
- Seed: 42
- Einstieg: Default (config.SEED_NODES)
- Views: directed, undirected
- Estimators: uniform_collision, wis-katzir__rw-history

Erzeugt mit `python plot_wis.py --graphs Slashdot0811` (Definition in `Code/plot_wis.py`, Eintrag `wis_rw_history__views`).

### `adjacency_list_uni__seed100__walk_diagnosis.png`

Diagnose eines Random Walks auf **GPT-4 knowledge graph (with literals)** (Seed 100): Leiter der Groessen, Abdeckungskurve, Besuche gegen Grad, meistbesuchte Entitaeten. Erzeugt mit `python diagnose_walk.py --graph adjacency_list_uni --views directed undirected --seed 100` (`Code/diagnose_walk.py`).

### `adjacency_list_uni__seed123__walk_diagnosis.png`

Diagnose eines Random Walks auf **GPT-4 knowledge graph (with literals)** (Seed 123): Leiter der Groessen, Abdeckungskurve, Besuche gegen Grad, meistbesuchte Entitaeten. Erzeugt mit `python diagnose_walk.py --graph adjacency_list_uni --views directed undirected --seed 123` (`Code/diagnose_walk.py`).

### `adjacency_list_uni__seed1__walk_diagnosis.png`

Diagnose eines Random Walks auf **GPT-4 knowledge graph (with literals)** (Seed 1): Leiter der Groessen, Abdeckungskurve, Besuche gegen Grad, meistbesuchte Entitaeten. Erzeugt mit `python diagnose_walk.py --graph adjacency_list_uni --views directed undirected --seed 1` (`Code/diagnose_walk.py`).

### `adjacency_list_uni__walk_diagnosis.png`

Diagnose eines Random Walks auf **GPT-4 knowledge graph (with literals)** (Seed 42): Leiter der Groessen, Abdeckungskurve, Besuche gegen Grad, meistbesuchte Entitaeten. Erzeugt mit `python diagnose_walk.py --graph adjacency_list_uni --views directed undirected` (`Code/diagnose_walk.py`).

### `gpt4_io__thinning-without-weights.png`

Keine Definition in `plot_wis.FIGURES` gefunden -- vermutlich von Hand oder mit einer aelteren Codeversion erzeugt.

### `gpt4o_adj_from_dataset__ranges.png`

Uebersichtsraster fuer **GPT-4o knowledge graph (with literals)** (Seed 42): Spalte = Kategorie, Zeile = Kantensicht. Erzeugt mit `python plot_results.py --graphs gpt4o_adj_from_dataset` (`plotting/ranges.py`).

### `gpt4o_io__thinning-without-weights.png`

Keine Definition in `plot_wis.FIGURES` gefunden -- vermutlich von Hand oder mit einer aelteren Codeversion erzeugt.

### `gpt4o_io__walk_diagnosis.png`

Diagnose eines Random Walks auf **GPT-4o knowledge graph (instances only)** (Seed 42): Leiter der Groessen, Abdeckungskurve, Besuche gegen Grad, meistbesuchte Entitaeten. Erzeugt mit `python diagnose_walk.py --graph gpt4o_io --views directed undirected` (`Code/diagnose_walk.py`).

### `gpt4o_io__wis_rw__undirected.png`

WIS with a true random walk -- all dead-end strategies (undirected)

- Graph: **GPT-4o knowledge graph (instances only)** (`gpt4o_io`)
- Seed: 42
- Einstieg: Default (config.SEED_NODES)
- Views: undirected
- Estimators: uniform_collision, wis-katzir__rw-restart, wis-katzir__rw-backtrack, wis-katzir__rw-history

Erzeugt mit `python plot_wis.py --graphs gpt4o_io` (Definition in `Code/plot_wis.py`, Eintrag `wis_rw__undirected`).

### `gpt4o_io__wis_rw_history__views.png`

WIS with random walk (history) -- directed vs undirected

- Graph: **GPT-4o knowledge graph (instances only)** (`gpt4o_io`)
- Seed: 42
- Einstieg: Default (config.SEED_NODES)
- Views: directed, undirected
- Estimators: uniform_collision, wis-katzir__rw-history

Erzeugt mit `python plot_wis.py --graphs gpt4o_io` (Definition in `Code/plot_wis.py`, Eintrag `wis_rw_history__views`).

## `saved/` -- die versionierten Meilensteine

Die Dateien oben werden bei jedem Plot-Lauf neu erzeugt und sind
**nicht** im Repository -- sonst laege dort nach jedem Durchlauf eine
weitere vollstaendige Kopie jedes Bildes. Was einen Meilenstein
festhaelt oder in eine Praesentation geht, wird bewusst nach `saved/`
kopiert; nur dieser Ordner ist versioniert.

```bash
cp data/plots/<name>.png data/plots/saved/
git add data/plots/saved && git commit -m "Meilenstein: ..."
```

Kopien in `saved/` werden nie ueberschrieben und koennen daher aus
einer aelteren Codeversion stammen -- im Zweifel gegen die Dateien
oben pruefen und gegen den Commit, der sie hinzugefuegt hat
(`git log -- data/plots/saved/<name>.png`).

- `Slashdot0811__wis_rw_history__views.png`
- `gpt4o_io__ranges.png`
- `gpt4o_io__walk_diagnosis-sink.png`
- `gpt4o_io__walk_diagnosis-weights.png`
