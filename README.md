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

## Entwurfsentscheidungen

Was ein "Knoten" und was eine "Kante" ist, ist bei diesen Daten keine
Selbstverstaendlichkeit -- und jede Festlegung verschiebt die Zahl, die
geschaetzt werden soll. Deshalb stehen sie hier, mit Begruendung und Folgen.

### 1. Mehrfachkanten werden entfernt

Die Quellen sind Tripel `(Subjekt, Praedikat, Objekt)`, die Adjazenzliste
kennt aber nur `(Subjekt, Objekt)`. Mehrere Fakten ueber dasselbe Paar fallen
dadurch auf dieselbe Kante zusammen und stehen mehrfach in der Liste.

*Entscheidung: eine Kante zaehlt einmal.* Der Grad soll die Zahl der
**verschiedenen** erreichbaren Nachbarn sein -- genau das unterstellen die
Gewichte `1/deg` und das gradproportionale Ziehen. Umgesetzt in
`graphs.graph._simplify` beim Laden, also fuer alle Sichten gleich.

Der Anteil betroffener Kanten:

| Graph | Kanten roh | ohne Mehrfachnennung | |
|---|---|---|---|
| Slashdot | 905 468 | 905 468 | 1.000x |
| gpt-4 | 100 138 669 | 96 212 315 | 1.041x |
| gpt-4o-io | 32 928 678 | 29 574 090 | 1.113x |

Das war vorher **inkonsistent**: `graphs.views._symmetric_csr` dedupliziert
beim Symmetrisieren schon immer, die gerichtete Sicht tat es nicht. Bei
gpt-4o-io verglich man also 11 % Mehrfachkanten gegen keine -- und genau
diese beiden Sichten stehen im Zentrum der Auswertung.

### 2. Schlingen werden entfernt

*Entscheidung: ein Knoten ist nicht sein eigener Nachbar.* Eine Schlinge
bringt einen Crawler nicht weiter, kostet aber eine Anfrage und zaehlt in
`deg` mit.

Das ist mehr als Kosmetik: in `Slashdot0811.pkl` enthalten **77 307 von
77 316** Adjazenzlisten den Knoten selbst. Ohne sie sinkt die Kantenzahl von
905 468 auf 828 161 (-8,5 %). Vorher filterte nur der Sampler
(`allow_self_loops=False`) die Schlinge beim *Schritt* heraus, waehrend sie im
Grad weiter mitzaehlte -- der Walk rechnete also mit einem Grad, den er nicht
nutzen konnte. Die 6 418 Slashdot-Knoten, deren einzige Kante auf sie selbst
zeigte, sind jetzt ehrliche Sackgassen mit Grad 0.

### 3. |V| = alle Knoten, auch die ohne ausgehende Kanten

*Entscheidung: die gesuchte Groesse ist die volle Knotenmenge* -- die
Schluessel der Adjazenzliste **plus** alle Knoten, die nur als Objekt
vorkommen.

Das ist die folgenreichste Festlegung, denn sie definiert die Wahrheit, gegen
die gemessen wird. Der Anteil der Knoten ohne ausgehende Kanten:

| Graph | \|V\| | davon ohne ausgehende Kanten |
|---|---|---|
| gpt-4 | 18 144 908 | 66.4 % |
| gpt-4o-io | 5 693 001 | 53.3 % |
| Slashdot | 77 360 | 0.06 % |

Wer nur die Schluessel zaehlte, suchte eine dreimal kleinere Zahl. Die Folge
ist an einer Stelle sichtbar und dort auch dokumentiert: gradproportionales
Ziehen (`DegWeightedIndependentOracle`) kann Knoten mit `deg_out = 0`
prinzipiell nie ziehen und schaetzt deshalb korrekt die Groesse von
`{v : deg_out(v) > 0}` -- bei gpt-4o-io stabil 0.4635 x |V|. Das ist kein
Fehler des Schaetzers, sondern die Antwort auf eine andere Frage.

### 4. Literale gehoeren nicht in den Graphen

Die GPT-Wissensgraphen enthalten neben Entitaeten auch Literale ("person",
"1890-03-11", "American") und unbrauchbare Fragmente ("< pre >",
"about 708,127 (2020) "). Die nodes-Tabellen unter `nodes/` typisieren jeden
Namen als `instance`, `literal` oder `undefined`.

*Entscheidung: kuenftige Adjazenzlisten enthalten nur Instanzen.* Gefiltert
wird auf beiden Kantenseiten; ein Schluessel, dessen Nachbarn alle Literale
waren, bleibt als Knoten ohne ausgehende Kanten erhalten (siehe 3.).

```bash
python build_instances_only.py --adjacency adjacency_list_uni \
    --nodes gpt4_nodes --out gpt4_io
```

**Welche nodes-Datei zu welchem Graphen gehoert**, sagen die Dateinamen nicht
zuverlaessig -- entschieden wurde es ueber die Knotenmengen. Anteil der
Knoten eines Graphen, die in der jeweiligen Tabelle vorkommen:

| | `gpt4_nodes` (18 185 374) | `gpt4o_nodes` (15 836 295) |
|---|---|---|
| `adjacency_list_uni`, \|V\| = 18 144 908 | **100.00 %** | 14.93 % |
| `gpt4o_adj_from_dataset`, \|V\| = 15 723 674 | 16.52 % | **95.26 %** |

Eindeutiger geht es nicht: `gpt4_nodes` deckt `adjacency_list_uni` bis auf
485 Namen vollstaendig ab. Dazu passen die Zeitstempel (`created_at`):
`gpt4o_nodes` beginnt am 2024-11-20, `gpt4_nodes` am 2025-06-06 -- und beide
starten bei derselben Saat-Entitaet `Vannevar Bush` mit `bfs_level = 0`. Die
aeltere Erhebung gehoert also zu gpt-4o, die neuere zu gpt-4; die Datei hiess
urspruenglich `gpt1_nodes` und ist nach diesem Abgleich umbenannt worden.

Was der Filter bewirkt (`gpt-4` -> `gpt-4-io`):

| | gpt-4 | gpt-4-io |
|---|---|---|
| \|V\| | 18 144 908 | **6 492 586** |
| Kanten | 96 212 315 | 45 638 776 |
| ohne ausgehende Kanten | 66.4 % | **9.7 %** |
| groesster Ausgangsgrad | 8 401 (`King`) | 4 237 (`King`) |
| groesster Eingangsgrad | 1 062 058 (`person`) | 920 251 (`United States`) |

Der Unterschied ist nicht kosmetisch: die Knoten ohne ausgehende Kanten gehen
von zwei Dritteln auf ein Zehntel zurueck, und der groesste Eingangsgrad
gehoert nicht mehr einem Literal (`person`), sondern einer echten Entitaet.
Fuer Random Walks auf der gerichteten Sicht ist das ein anderer Graph.

Die Altbestaende bleiben zum Vergleich liegen: `gpt-4` und `gpt-4o` enthalten
Literale, `gpt-4-io` und `gpt-4o-io` nicht. Gegen die Typtabellen
nachgeprueft sind beide zu **100.000 %** Instanzen -- Schluessel und nur als
Objekt vorkommende Knoten getrennt geprueft, kein `literal`, kein
`undefined`, kein Name, der in der Typtabelle fehlt. Das ist zugleich die
schaerfste Bestaetigung der Zuordnung oben: mit der falschen Tabelle kaeme
nie eine glatte 100 % heraus.

Wichtige Folge fuer die Auslegung der Ergebnisse: die Knoten ohne ausgehende
Kanten sind in diesen Dateien **keine Literale**, sondern Instanzen, die beim
Aufbau des Wissensgraphen nie expandiert wurden -- der offene Rand des BFS.

| | expandierte Instanzen | \|V\| | ohne ausgehende Kanten |
|---|---|---|---|
| gpt-4-io | 6 050 977 von 6 505 583 (93 %) | 6 492 586 | 9.7 % |
| gpt-4o-io | 2 657 109 von 5 693 001 (47 %) | 5 693 001 | 53.3 % |

Der Unterschied zwischen beiden Graphen ist also nicht die Filterung, sondern
wie weit der Crawl gekommen ist. `wis-katzir__indep` schaetzt auf gpt-4o-io
deshalb 0.4635 x |V|: es misst die Menge der *expandierten* Instanzen.

### 5. Feste Einstiegsknoten statt gleichverteiltem Start

Ein Crawler kann nicht gleichverteilt aus V ziehen -- dafuer muesste er V
schon kennen, also genau das, was geschaetzt werden soll. Er startet bei ein
paar bekannten Entitaeten.

*Entscheidung: je Graph eine feste, einmal festgelegte Liste* in
`config.SEED_NODES`. Welcher der Knoten einen Lauf startet, entscheidet der
Zufall des Laufs -- sonst beginnen alle Wiederholungen an derselben Stelle
und die Streuung ueber die Laeufe waere kuenstlich klein. Benutzt wird das von
`CrawlOracle.seed_nodes()`, also von allen real umsetzbaren Verfahren; die
Vergleichsverfahren mit globalem Zugriff sind nicht betroffen.

Fuer Slashdot sind es fuenf mit `Random(42)` aus den 70 898 Knoten mit
ausgehenden Kanten gezogene Knoten (3285, 14758, 30177, 33136, 37446). Ihre
kleinen Grade (1 bis 6) sind kein Versehen, sondern das, was gleichverteiltes
Ziehen in einem schwanzlastigen Graphen liefert.

Fuer die GPT-Basen fuenf Entitaeten verschiedener Art, jede in **beiden**
Basen als Schluessel vorhanden (Ausgangsgrad gpt-4-io / gpt-4o-io):

| Startknoten | gpt-4-io | gpt-4o-io | |
|---|---|---|---|
| `Vannevar Bush` | 34 | 26 | die Saat-Entitaet beider Erhebungen |
| `Isaac Newton` | 38 | 79 | Wissenschaftler |
| `United States of America` | 81 | 92 | Land |
| `Kurashiki` | 33 | 37 | mittelgrosse japanische Stadt |
| `Katsushika Hokusai` | 25 | 80 | Kuenstler |

Zwei Auswahlen sind bewusst so getroffen: Von den USA-Schreibweisen ist
`United States of America` die einzige mit aehnlichem Grad in beiden Basen
(`United States` 62/913, `USA` 81/781) -- mit den anderen startete der Crawl
in beiden Graphen unter verschiedenen Bedingungen. Und als Kuenstler
ausdruecklich nicht `Yoshitomo Nara` (13/**148 884**): der Ausreisser aus
Punkt 4 wuerde als fester Startknoten jeden Lauf auf gpt-4o-io dominieren.

### Noch offen

- **Entitaets-Identitaet.** `Yoshitomo Nara` und `Nara Yoshitomo` sind zwei
  Knoten mit je ueber 100 000 Nachbarn -- vermutlich dieselbe Person. Es wird
  nicht normalisiert (weder Gross-/Kleinschreibung noch Alias-Aufloesung),
  |V| ist dadurch tendenziell zu gross.
- **Was das Orakel in der Praxis liefert.** Modelliert sind nur ausgehende
  Kanten. Ob eine reale Schnittstelle auch eingehende liefert, entscheidet,
  ob die `undirected`-Sicht umsetzbar ist oder blosse Vergleichsgroesse
  bleibt.

## Struktur

```
adjacencies/            originale .pkl-Adjazenzlisten (unveraendert)
nodes/                  Typtabellen der Wissensgraphen (name, type, ...)
data/results/           CSVs pro Graph (Schaetzungen + Besuchsstatistik)
data/plots/             erzeugte Plots
Code/
  config.py             Pfade, Default-Budgets, n, Seed, Budget-Metrik
  run_experiment.py     CLI: Experiment ausfuehren
  plot_results.py       CLI: Plots erzeugen
  check_nested.py       CLI: genestete Budgets gegen Einzellaeufe pruefen
  build_instances_only.py  CLI: Adjazenzliste auf Instanzen filtern
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

## Mehrere Durchlaeufe je Graph: `--seed`

Der Seed bestimmt den kompletten Zufallsstrom eines Laufs. Ein zweiter Lauf mit
anderem Seed ist damit ein zweiter, gleichberechtigter Durchlauf desselben
Experiments -- die Art, hier zu pruefen, ob ein Ergebnis stabil ist oder am
Zufall haengt. Alle vier Skripte nehmen `--seed`:

```bash
python run_experiment.py --graphs Slashdot0811 --seed 7
python plot_wis.py       --graphs Slashdot0811 --seed 7   # ohne --seed: jeder vorhandene
python plot_results.py   --graphs Slashdot0811 --seed 7
python diagnose_walk.py  --graph  Slashdot0811 --seed 7
```

Der Seed ist dabei nirgends stillschweigend:

| Wo | Wie |
|---|---|
| Ergebnis-CSV | Spalte `seed` in jeder Zeile |
| Dateiname | `Slashdot0811__seed7__estimates.csv`, `..._seed7__ranges.png` |
| Grafik | klein oben rechts, z.B. `seed 7` |

Der Default-Seed (`config.DEFAULT_SEED`) bekommt **keinen** Namenszusatz --
`Slashdot0811__estimates.csv` ist der Lauf mit Seed 42. Zwei Laeufe
ueberschreiben sich dadurch nie gegenseitig, und die Plot-Skripte zeichnen ohne
`--seed` jeden vorhandenen Durchlauf einzeln: Laeufe verschiedener Seeds in
eine Spanne zu mischen waere irrefuehrend, weil die gezeigte Streuung dann
zwei Dinge auf einmal misst.

Innerhalb eines Laufs bleibt die Paarung erhalten (s.o.): der abgeleitete Strom
haengt an (Seed, Estimator, Budget, Lauf), nicht an der View.

## Alle Budgets aus einem Lauf: `--checkpoint-budgets`

Statt je Budget einen eigenen Lauf zu rechnen, laeuft *ein* Lauf mit dem
groessten Budget und haelt unterwegs fest, wo die kleineren geendet haetten.
Die Stichprobe wird dort abgeschnitten und ganz normal geschaetzt.

```bash
python run_experiment.py --graphs Slashdot0811 --checkpoint-budgets
python check_nested.py   --graph  Slashdot0811     # Aequivalenz nachrechnen
```

**Das ist exakt, nicht genaehert.** Kein Sampler kennt sein Budget -- es
steuert nur den Abbruch. Der bei Kosten b abgeschnittene Lauf ist deshalb
bitgleich mit einem eigenstaendigen Lauf desselben Zufallsstroms bei Budget b:
gleicher Cache, gleiche Historie, gleicher Abbruchpunkt. `check_nested.py`
rechnet das fuer alle Estimators und Views nach und vergleicht Schaetzwert,
Kosten, Abbruchgrund und Sample-Zahl auf Gleichheit (nicht auf Toleranz).

Was sich dadurch **nicht** aendert: die Verteilung je Budget. Ein Punkt bei 1 %
bedeutet genau dasselbe wie vorher, Bias und Streuung inklusive.

Was sich aendert: die Punkte einer Laufnummer sind ueber die Budgets
**genestet**. Ein Walk, der bei 1 % feststeckt, steckt bei 10 % immer noch
fest. Vergleiche *zwischen* Budgets werden dadurch gepaart und praeziser; die
Zahl unabhaengiger Trajektorien im Experiment sinkt aber von
`Laeufe x Budgets` auf `Laeufe`. Fuer Fragen der Art "wie oft landet ein Walk
in einer Senke?" ist das der relevante Verlust. Die Ergebnis-CSV haelt es in
der Spalte `nested` fest, die Grafiken vermerken es oben rechts.

Ersparnis: theoretisch `Sigma(Budgets) / max(Budget)`, bei der Standardleiter
also 1.66x -- aber nur fuer das *Ziehen*. Thinning und Formel laufen weiterhin
je Budget (auf dem jeweiligen Praefix). Gemessen auf Slashdot0811, gerichtet,
3 Estimators x 5 Budgets x 10 Laeufe, `--jobs 1`:

| | Rechenzeit in den Estimators |
|---|---|
| je Budget ein Lauf | 13.9 s |
| genestet | 10.7 s (**1.29x**) |

Zwei Einschraenkungen:

- `capture_recapture` teilt sein Budget vorab auf zwei Walks auf; ein Praefix
  hat dort eine andere Struktur. Der Estimator laeuft in diesem Modus weiter
  je Budget einzeln (der Runner erkennt das selbst).
- Besuchszaehler entstehen nur fuer das groesste Budget -- sie sind kumulativ,
  ein Zwischenstand muesste den ganzen Counter kopieren.

Der abgeleitete Zufallsstrom haengt in diesem Modus an (Seed, Estimator, Lauf)
statt an (Seed, Estimator, Budget, Lauf) -- es gibt ja nur noch einen Lauf.
Ergebnisse beider Modi sind deshalb nicht Zeile fuer Zeile vergleichbar, wohl
aber Verteilung gegen Verteilung.

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

- |V| = alle vorkommenden Knoten (siehe "Entwurfsentscheidungen", Punkt 3).
  Als CSR brauchen sie keinen Sonderfall: sie haben `indptr[u] == indptr[u+1]`
  und damit Grad 0. Die .pkl-Dateien unter `adjacencies/` werden nie
  ueberschrieben; neue entstehen nur ueber `build_instances_only.py`.
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
