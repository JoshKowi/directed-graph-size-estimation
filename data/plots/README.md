# data/plots -- erzeugte Grafiken

*Automatisch erzeugt von `Code/provenance.py` -- nicht von Hand aendern.*

| Daten vom | 2026-08-25 17:49 |
|---|---|
| Code-Fingerabdruck | `ef0b29a24f26` |
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

Jede Grafik zeigt pro Estimator und Budget die Spanne min..max ueber die
Laeufe plus den Median, y = Schaetzung/|V| (log), gestrichelt die wahre
Groesse bei 1.0. Die x-Achse nennt das Budget relativ und absolut.

Referenzreihe in allen `wis_*`/`deadend_*`-Grafiken: `uniform_collision`.

## Dateien

### `Slashdot0811__deadend_uis__directed.png`

UIS collision counting with a random walk -- dead-end strategies (directed)

- Graph: **Slashdot0811**
- Views: directed
- Estimators: uniform_collision, rw_plain__restart__none, rw_plain__backtrack__none, rw_plain__history__none

Erzeugt mit `python plot_wis.py --graphs Slashdot0811` (Definition in `Code/plot_wis.py`, Eintrag `deadend_uis__directed`).

### `Slashdot0811__wis_indep__directed.png`

WIS (Katzir) -- independent degree-weighted draws (directed)

- Graph: **Slashdot0811**
- Views: directed
- Estimators: uniform_collision, wis-katzir__indep

Erzeugt mit `python plot_wis.py --graphs Slashdot0811` (Definition in `Code/plot_wis.py`, Eintrag `wis_indep__directed`).

### `Slashdot0811__wis_indep__undirected.png`

WIS (Katzir) -- independent degree-weighted draws (undirected)

- Graph: **Slashdot0811**
- Views: undirected
- Estimators: uniform_collision, wis-katzir__indep

Erzeugt mit `python plot_wis.py --graphs Slashdot0811` (Definition in `Code/plot_wis.py`, Eintrag `wis_indep__undirected`).

### `Slashdot0811__wis_rw__undirected.png`

WIS with a true random walk -- all dead-end strategies (undirected)

- Graph: **Slashdot0811**
- Views: undirected
- Estimators: uniform_collision, wis-katzir__rw-restart, wis-katzir__rw-backtrack, wis-katzir__rw-history

Erzeugt mit `python plot_wis.py --graphs Slashdot0811` (Definition in `Code/plot_wis.py`, Eintrag `wis_rw__undirected`).

### `Slashdot0811__wis_rw_history__views.png`

WIS with random walk (history) -- directed vs undirected

- Graph: **Slashdot0811**
- Views: directed, undirected
- Estimators: uniform_collision, wis-katzir__rw-history

Erzeugt mit `python plot_wis.py --graphs Slashdot0811` (Definition in `Code/plot_wis.py`, Eintrag `wis_rw_history__views`).

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
