# Graph-Groessen-Schaetzung mit kleinem Sample

Ziel: verschiedene Verfahren vergleichen, die mit moeglichst wenig Zugriffen die
Knotenzahl |V| eines Graphen schaetzen.

## Ablauf

1. `python run_experiment.py --graphs <name>` laedt die Adjazenzliste **einmal**,
   baut daraus jede gewaehlte Kantensicht (`--views`, Default
   `directed undirected`), bestimmt fuer jedes relative Budget `b` das absolute
   Budget `round(b * |V|)` und laesst jeden Estimator `n`-mal laufen (n=10).
2. Ergebnisse landen in `data/results/<graph>__estimates.csv`, die
   Besuchshaeufigkeit je Original-Knotenname in `data/results/<graph>__visits.csv`.
3. `python plot_results.py` erzeugt `data/plots/<graph>__ranges.png`
   (Spanne min..max plus Median je Estimator und Budget; Spalte = Kategorie,
   Zeile = Kantensicht) und `data/results/<graph>__view_comparison.csv` mit den
   Vergleichszahlen zwischen den Sichten.

Alle Skripte werden aus dem Ordner `Code/` heraus gestartet.

## Struktur

```
adjacencies/            originale .pkl-Adjazenzlisten (unveraendert)
data/results/           CSVs pro Graph (Schaetzungen + Besuchsstatistik)
data/plots/             erzeugte Plots
Code/
  config.py             Pfade, Default-Budgets, n, Seed, Budget-Metrik
  run_experiment.py     CLI: Experiment ausfuehren
  plot_results.py       CLI: Plots erzeugen
  graphs/               Graph als CSR (Integer-IDs + Namensliste), Loader, Views
  oracles/              Graph-Zugriff mit Kostenzaehlung und Budget
    global_access.py      setzt Kenntnis von V voraus (gleichverteiltes Ziehen)
    local_access.py       nur Nachbarschaftsabfragen ab einem Seed
  sampling/             wie das Oracle genutzt wird
    samplers.py           UniformSampler, RandomWalkSampler
    dead_ends.py          Sackgassen: restart | backtrack | history
    thinning.py           Dependency Reduction: none | simple | shifted
  weighting/            Korrektur der Sampling-Verzerrung (w_i ~ 1/pi(u_i))
  estimators/
    base.py               Estimator-Interface, Kategorien, EstimateResult
    formulas.py           k^2/n_col (plain) und gradkorrigiert (weighted)
    pipeline.py           Oracle + Sampler + Thinning + Weighting + Formula + Aggregation
    methods/              die Verfahren selbst (ohne Kategorie-Trennung)
    __init__.py           REGISTRY -- Verfahren + Kategorie eintragen
  experiment/           Runner (Estimator x Budget x Wiederholung) + CSV-IO
  plotting/             Farbpalette und Range-Plot
```

Warum ein eigener `sampling/`-Ordner neben Oracle/Weighting/Estimator: das Oracle
regelt, *was* abgefragt werden darf (und damit, ob ein Verfahren real umsetzbar
ist), der Sampler, *wie* daraus eine Stichprobe wird. Random Walk und
unabhaengiges Ziehen nutzen dasselbe Oracle, erzeugen aber unterschiedliche
Verzerrungen -- die dann das Weighting korrigiert.

## Gerichtet vs. ungerichtet vergleichen

Alle Testgraphen sind gerichtet. `graphs/views.py` legt drei Kantensichten auf
dieselbe (vollstaendige) Knotenmenge -- |V| bleibt identisch, damit die
Schaetzungen vergleichbar sind:

| View | Nachbarn von u |
|------|----------------|
| `directed` | Ausgangskanten (Originalgraph) |
| `undirected` | Aus- und Eingangskanten, dedupliziert |
| `reverse` | nur Eingangskanten |

```bash
python run_experiment.py --graphs Slashdot0811 --views directed undirected reverse
python plot_results.py --graphs Slashdot0811
```

Die Laeufe sind **gepaart**: der Seed haengt von (Estimator, Budget, Lauf) ab,
aber nicht von der View. Lauf 3 startet in jeder Sicht mit demselben
Zufallsstrom, Unterschiede gehen also auf die Kantensicht zurueck und nicht auf
RNG-Rauschen. `results.compare_views(df)` nutzt das und gibt neben Median und
Spannweite je View die Spalte `ratio_vs_directed` aus -- den Median des
gepaarten Verhaeltnisses (> 1: in dieser Sicht wird hoeher geschaetzt).

Speicher: `undirected` und `reverse` bauen einmalig eine zweite Adjazenz auf.
Bei den grossen Graphen ggf. je View einen eigenen Lauf starten.

## Neuen Estimator hinzufuegen

Zusammengesetzt (Regelfall): Modul in `estimators/methods/` anlegen mit einer
`build()`-Funktion, die einen `PipelineEstimator` liefert -- siehe
`methods/random_walk_collision.py`.

In einem Schritt: direkt von `Estimator` erben und `estimate(graph, budget, rng)`
implementieren -- siehe `methods/capture_recapture.py`.

Danach in `estimators/__init__.py` in `REGISTRY` eintragen, zusammen mit der
Kategorie:

```python
REGISTRY = {
    "rw_collision": Entry(random_walk_collision.build, Category.REALIZABLE),
}
```

## Random-Walk-Varianten

Zwei orthogonale Achsen, als Kreuzprodukt in der REGISTRY:

| Sackgasse (`dead_end`) | Verhalten bei 0 nutzbaren ausgehenden Kanten |
|---|---|
| `restart` | Sprung zurueck zum Startknoten |
| `backtrack` | Schritte zurueck, bis ein Vorgaenger eine andere Abzweigung hat |
| `history` | Sprung auf einen zufaelligen Knoten der bisherigen Besuchsfolge |

| Thinning | Sample-Sets aus der Trajektorie |
|---|---|
| `none` | ein Set: die ganze Trajektorie |
| `simple` | ein Set: jedes n-te Sample (n=5), verwirft 4/5 des Budgets |
| `shifted` | n Sets mit Offset 0..n-1; je Set eine Schaetzung, aggregiert per Median |

Backtracking-Abfragen laufen ueber das Oracle und kosten Budget, erzeugen aber
keine Samples -- sie sind Navigation, keine Beobachtung. Deshalb liefert
`backtrack` bei gleichem Budget etwas weniger Samples.

**Die `dead_end`-Achse wirkt nur auf gerichteten Views.** In `undirected` hat
jeder Knoten mindestens die Rueckkante, ueber die der Walk ihn erreicht hat --
es gibt keine Sackgassen (Slashdot0811: 8,35 % der Knoten sind gerichtet eine
Sackgasse, symmetrisiert 0,00 %). Die drei Strategien sind dort derselbe
Algorithmus, Unterschiede in den Ergebnissen sind reines RNG-Rauschen. Fuer
einen Vergleich der Strategien also `--views directed` fahren.

**Selbstkanten:** ein Knoten, dessen einzige ausgehende Kante auf ihn selbst
zeigt, wuerde den Walk absorbieren -- in Slashdot0811 betrifft das 6418 Knoten
(8,3 %). `RandomWalkSampler(allow_self_loops=False)` (Default) wertet das als
Sackgasse, damit die dead_end-Strategie greift.

## Kategorien

`Category.COMPARISON` (nur zum Vergleich) vs. `Category.REALIZABLE` (real
umsetzbar) ist bewusst **nur ein Label in der REGISTRY**, keine Ordner- oder
Klassentrennung. Auch die Oracle-Module heissen nach dem Zugriffsmodell
(`global_access` / `local_access`), nicht nach der Kategorie. Ob ein Verfahren umsetzbar ist, entscheidet das Oracle -- und
dasselbe Verfahren kann mit einem anderen Oracle in die andere Kategorie
fallen. `estimators.build(name)` haengt das Label nach der Konstruktion an die
Instanz; ein direkt konstruierter Estimator hat `category is None`.

## Groesse und Laufzeit

Beim Laden werden die Original-Schluessel (bei gpt4o_io Strings wie
`'Vannevar Bush'`) einmal auf Integer-IDs 0..n-1 abgebildet; der Graph liegt
danach als CSR in drei numpy-Arrays. `Graph.name_of(id)` und
`Graph.id_of(name)` fuehren zurueck, und die Besuchs-CSV enthaelt wieder die
Original-Namen. Das spart bei gpt4o_io den Grossteil von ~37 GB und ist
Voraussetzung fuer die Parallelisierung: nur so uebersteht der Graph den
`fork` ohne kopiert zu werden.

**Parallelisierung.** `--jobs N` (Default in `config.DEFAULT_N_JOBS`) verteilt
die (Budget, Estimator, Lauf)-Tripel einer View auf N Prozesse. Die Ergebnisse
sind davon unabhaengig -- der Seed haengt nur an (Estimator, Budget, Lauf),
nicht an der Ausfuehrungsreihenfolge; `--jobs 1` liefert dieselbe CSV.

**Grosse Graphen.** Ab `config.LARGE_GRAPH_NODES` faellt das 20-%-Budget weg.
Auf Slashdot0811 entfallen 63 % aller Walk-Schritte auf dieses eine Budget,
und der Aufwand skaliert mit |V|. Mit `--budgets` laesst es sich erzwingen.

Warum die Walks so viele Schritte machen: bei `COST_CACHE_HIT = 0.02` sind bis
zu 50 Schritte je Budget-Einheit moeglich, und auf gerichteten Graphen
schoepfen die Random Walks das fast aus (Slashdot0811: Faktor 47). Auf
symmetrisierten Sichten liegt der Faktor bei 1,3.

## Kosten und Budget

Das Oracle kennt drei Zugriffsarten -- verschiedene Anfragen mit eigenen
Preisen in `config.py`:

| Zugriff | Preis | Default |
|---|---|---|
| `_draw()` / `_draw_by_degree()` -- Knoten ziehen | `COST_RANDOM_NODE` | 1 |
| `_fetch(u)` -- Nachbarn von u, erster Zugriff | `COST_NEIGHBORS` | 1 |
| `_fetch(u)` -- Nachbarn von u, Cache-Treffer | `COST_CACHE_HIT` | 0.02 |

**Cache:** ein Knoten kostet beim ersten Zugriff den vollen Preis, danach nur
noch `COST_CACHE_HIT` -- ein realer Crawler haelt die einmal geholte
Nachbarschaft, muss sie aber weiterhin nachschlagen. Das betrifft vor allem
`backtrack`, das gezielt in bekanntes Gebiet zurueckgeht. Eine Zufallsziehung
ist dagegen jedes Mal eine neue Anfrage und wird nie gecacht. Nutzt ein
Estimator mehrere Crawler (`capture_recapture`), teilen sie sich den Cache --
es ist derselbe Client.

Sind die Einstiegsknoten fest bekannt, ist der Zufallszugriff faktisch gratis:
dann `COST_RANDOM_NODE = 0` setzen.

**Warum der Cache-Treffer nicht gratis ist.** Mit Preis 0 laeuft ein Walk, der
sich in einer kleinen, bereits bekannten Region verfaengt, endlos gratis weiter
und sammelt beliebig viele wertlose, hochkorrelierte Samples -- die Schaetzung
haengt dann an der Abbruchkonstante statt am Verfahren. Mit einem Preis > 0
terminiert das Budget jeden Lauf von selbst, und **jeder Estimator schoepft sein
Budget aus**. Erst dadurch sind "erlaubtes" und "genutztes" Budget vergleichbar
-- vorher gab `uniform_collision` 98 % aus und ein Random Walk 5 %, bei gleicher
Sample-Zahl.

Der Preis ist damit der Regler fuer die Schrittzahl: die Decke fuer einen voll
gecachten Walk ist `budget / COST_CACHE_HIT`, bei 0.02 also 50 x Budget. Er
bestimmt die Groessenordnung der Schaetzung fuer verfangene Walks mit und
gehoert in jede Ergebnisdarstellung. `COST_CACHE_HIT = 1` entspricht "kein
Cache-Rabatt", also einem Schritt je Budget-Einheit.

**Kein globales Aufruf-Limit.** Weil jeder Zugriff einen Preis > 0 hat, endet
jeder Lauf am Budget -- ein zusaetzlicher Schritt-Deckel waere nie bindend und
wuerde nur verschleiern, wo tatsaechlich Schluss ist. Wo eine Schleife nicht
ueber den Preis endet, wird sie an ihrer eigenen Stelle begrenzt: der
`ShortWalkIndependentOracle` gibt nach `max_restarts` erfolglosen Startknoten
auf. Die Spalte `stopped_by` in der Ergebnis-CSV sagt, warum ein Lauf endete
(`budget` im Normalfall, `no_path` in dem einen Sonderfall).

Aus demselben Grund ist `"unique_nodes"` als Budget-Metrik entfallen: sie
waechst bei einem Walk in bekanntem Gebiet nicht mehr und wuerde nie
erschoepfen. Das Oracle lehnt sie mit einem `ValueError` ab.

In der Ergebnis-CSV stehen ausserdem `queries_used` (bezahlte, gewichtete
Kosten), `unique_nodes_used` (Statistik), `cached_queries` sowie
`n_random_node` / `n_neighbors`.

## Bekannte Vereinfachungen

- |V| = alle vorkommenden Knoten, also die Eintraege der Adjazenzliste **plus**
  die Knoten, die nur als Nachbar auftauchen (keine ausgehenden Kanten). Als
  CSR brauchen sie keinen Sonderfall: sie haben `indptr[u] == indptr[u+1]` und
  damit Grad 0. Die .pkl-Dateien werden nie geschrieben.
- Der Random Walk unterstellt pi(u) ~ deg(u); das gilt streng nur ungerichtet.
  Ueber `--views undirected` laesst sich der Effekt messen.
- Der Collision-Schaetzer unterstellt *unabhaengige* Samples aus pi. Ein Random
  Walk liefert stark autokorrelierte Samples, was |V| massiv unterschaetzt --
  dagegen laufen die Thinning-Varianten.
- **Faktor-2-Korrektur gegenueber Kurant.** Kurant/Butts/Markopoulou, "Graph
  Size Estimation" (arXiv 1210.0460), definiert `n_col` in Eq.(4) ueber
  *ungeordnete* Paare i<j -- damit ist `E[n_col] = C(k,2)/N`. Eq.(5)
  (`k^2/n_col`) und Eq.(6) (`(sum w)(sum 1/w)/n_col`) setzen aber `k^2` in den
  Zaehler und liegen dadurch um `2k/(k-1) ~ 2` zu hoch; Eq.(6) erbt den Faktor
  von Eq.(5), auf die sie sich unter UIS reduziert. In `estimators/formulas.py`
  sind beide auf `C(k,2)` korrigiert, jeweils mit Kommentar an der Stelle.
  Nachgerechnet (N=2000, Median ueber 200 Wdh.): vorher 4002 / 4002, nachher
  1992 / 2003. Katzirs Form (`C(k,2)*mean(w)*mean(1/w)/n_col`) war bereits
  konsistent und ist unveraendert -- nach der Korrektur liefert
  `wis-col-kurant` dieselben Werte wie `wis-col-katzir`.
- Bei `shifted` stammen die n Schaetzungen aus derselben Trajektorie und sind
  korreliert. Die Mittelung reduziert die Varianz weniger als n unabhaengige
  Laeufe (eher Batch-Means als echte Replikation).
