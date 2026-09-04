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
   (Spanne min..max plus Median je Estimator und Budget; eine Spalte je
   Kantensicht, eine Farbe je Estimator) und
   `data/results/<graph>__view_comparison.csv` mit den Vergleichszahlen
   zwischen den Sichten sowie `data/results/<graph>__budget_breakdown.csv`
   mit den Kosten je Sample und ihrer Aufteilung (siehe "Kosten und Budget").

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

Welcher Einstieg benutzt wird, steuert `--start-node`:

```bash
python run_experiment.py --graphs gpt-4-io                        # Vannevar Bush
python run_experiment.py --graphs gpt-4-io --start-node Kurashiki
python run_experiment.py --graphs gpt-4-io --start-node all       # alle fuenf
```

Default ist der erste Eintrag der Liste. `all` rechnet alle fuenf
nacheinander, jeden in **eigene Dateien** -- verschiedene Einstiege sind
verschiedene Bedingungen und gehoeren nicht in dieselbe Spanne, genau wie bei
`--seed`:

| Wo | Wie |
|---|---|
| Ergebnis-CSV | Spalte `start_node` in jeder Zeile |
| Dateiname | `gpt4_io__start-kurashiki__estimates.csv` |
| Grafik | oben rechts, `seed 42  \|  start: Kurashiki` |

Der Default-Einstieg bekommt wie der Default-Seed **keinen** Namenszusatz.
Innerhalb eines Durchlaufs starten alle Wiederholungen am selben Knoten; die
Streuung im Bild ist dann reines RNG-Rauschen des Walks, nicht mehr die
Mischung aus Einstieg und Walk.

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
  check_shared.py       CLI: geteilte Walks gegen Einzellaeufe pruefen
  build_instances_only.py  CLI: Adjazenzliste auf Instanzen filtern
  graphs/               Graph als CSR (Integer-IDs + Namensliste), Loader, Views
  oracles/              Graph-Zugriff mit Kostenzaehlung und Budget
    global_access.py      setzt Kenntnis von V voraus (gleichverteiltes Ziehen)
    local_access.py       nur Nachbarschaftsabfragen ab einem Seed
  sampling/             wie das Oracle genutzt wird
    samplers.py           UniformSampler, RandomWalkSampler
    durw.py               DURW: Random Walk mit aufgebautem G_u und Sprung
    dead_ends.py          Sackgassen: restart | backtrack | history
    jumps.py              DURW-Sprungarten: uniform
    thinning.py           Dependency Reduction: none | simple | shifted
  weighting/            Korrektur der Sampling-Verzerrung (w_i ~ 1/pi(u_i))
  estimators/
    base.py               Estimator-Interface, Kategorien, EstimateResult
    formulas.py           k^2/n_col (plain) und gradkorrigiert (weighted)
    pipeline.py           Oracle + Sampler + Thinning + Weighting + Formula + Aggregation
    methods/              die Verfahren selbst (ohne Kategorie-Trennung)
    __init__.py           REGISTRY -- Verfahren + Kategorie eintragen
  experiment/           Runner (Estimator x Budget x Wiederholung) + CSV-IO
  plotting/             Farbpalette und Vergleichs-Plot (`compare.py`)
```

Warum ein eigener `sampling/`-Ordner neben Oracle/Weighting/Estimator: das Oracle
regelt, *was* abgefragt werden darf (und damit, ob ein Verfahren real umsetzbar
ist), der Sampler, *wie* daraus eine Stichprobe wird. Random Walk und
unabhaengiges Ziehen nutzen dasselbe Oracle, erzeugen aber unterschiedliche
Verzerrungen -- die dann das Weighting korrigiert.

## Bausteine der Pipeline

Ein Estimator ist im Regelfall kein eigener Algorithmus, sondern eine
Kombination austauschbarer Stufen (`estimators/pipeline.py`):

```
Oracle -> Sampler -> Thinning -> Weighting -> Formel -> Aggregation
  was       wie      Sets aus    Verzerrung   Zahl je   ueber die
darf man   wird       der Tra-   korrigieren   Set        Sets
 fragen   gezogen     jektorie
```

### 1. Oracle -- was abgefragt werden darf

Entscheidet ueber die Kategorie: globaler Zugriff setzt Kenntnis von V voraus
(also genau das, was geschaetzt werden soll) und ist deshalb nur Vergleich.

| Oracle | Modul | Zugriff | Verteilung der Samples | Kategorie |
|---|---|---|---|---|
| `UniformNodeOracle` | `global_access` | `random_node()`, `neighbors()` | gleichverteilt ueber V | Vergleich |
| `DegWeightedIndependentOracle` | `global_access` | wie oben, aber gewichtet gezogen | pi(v) ~ deg_out(v), unabhaengig | Vergleich |
| `ShortWalkIndependentOracle` | `global_access` | Endknoten eines Walks fester Laenge (`steps`) | Walk-Verzerrung, aber unabhaengig | Vergleich |
| `CrawlOracle` | `local_access` | `seed_nodes()`, `neighbors()` | was der Walk erreicht | real umsetzbar |
| `JumpCrawlOracle` | `local_access` | wie `CrawlOracle` + `random_node()` | was der Walk erreicht | haengt an der Sprungart (s.u.) |

### 2. Sampler -- wie daraus eine Stichprobe wird

| Sampler | Parameter | Ergebnis |
|---|---|---|
| `UniformSampler` | `n_walks` | unabhaengige Ziehungen; `n_walks` > 1 teilt sie in ebenso viele Faenge (Budget gleichmaessig) |
| `RandomWalkSampler` | `dead_end`, `n_seeds`, `n_walks`, `burn_in`, `restart_prob`, `allow_self_loops` | volle Trajektorie eines (oder `n_walks` nacheinander laufender) Random Walks |
| `DurwSampler` | `jump`, `jump_weight`, `n_seeds`, `n_walks`, `burn_in` | wie oben, aber DURW: baut waehrend des Laufs ein ungerichtetes G_u auf und springt gradproportional |

### 3. Sackgassen-Strategie -- nur fuer den Random Walk

Wirkt ausschliesslich auf gerichteten Views (s. "Random-Walk-Varianten").

| `dead_end` | Verhalten bei 0 nutzbaren ausgehenden Kanten |
|---|---|
| `restart` | zurueck zum Startknoten |
| `backtrack` | Schritte zurueck, bis ein Vorgaenger eine andere Abzweigung hat |
| `history` | Sprung auf einen zufaelligen bereits besuchten Knoten |

### 3b. DURW -- Random Walk, dessen Verteilung auch gerichtet bekannt ist

Der einfache Random Walk laeuft auf den gerichteten Views auf einem Graphen,
auf dem er gar nicht laufen duerfte: pi(u) ~ deg(u) gilt nur ungerichtet, und
jede Sackgassen-Strategie verschiebt die Verteilung noch einmal. DURW (Ribeiro
& Towsley, `sampling/durw.py`) loest das mit zwei Zutaten:

1. **Rueckwaerts begehbare Kanten.** Jede beobachtete Ausgangskante u -> v wird
   gemerkt; landet der Walk spaeter auf v, darf er sie rueckwaerts nehmen.
   Aber nur, solange v noch *unbesucht* ist -- Kanten auf bereits besuchte
   Knoten werden verworfen. Damit steht der Grad eines Knotens im aufgebauten
   ungerichteten Graphen G_u fest, sobald er zum ersten Mal besucht wird, und
   aendert sich nie wieder. Genau das braucht die Gewichtung: sonst hinge sie
   an Kanten, die der Walk erst spaeter sieht.
2. **Gradproportionale Spruenge.** Mit Wahrscheinlichkeit `w / (w + deg_Gu(v))`
   springt der Walk auf einen zufaellig gezogenen Knoten. Auf dem so
   entstehenden gewichteten Graphen ist

       pi(v) = (w + deg_Gu(v)) / (vol(V) + w|V|)

   -- bis auf die Normierung bekannt, sobald v besucht ist; die kuerzt der
   Kollisionsschaetzer heraus. `weighting.DurwWeighting` setzt das als
   `1/(w + deg_Gu)` ein (nicht `InverseDegreeWeighting`, die gehoert zum
   einfachen Random Walk).

Zwei Dinge fallen dadurch weg: eine **Sackgassen-Strategie** (bei `deg_Gu = 0`
ist die Sprungwahrscheinlichkeit 1 -- Sackgassen sind der Grenzfall der
Sprungregel, kein Sonderfall) und `allow_self_loops` (Schlingen sind beim
Laden schon weg). `Sample.degree` traegt bei DURW den Grad in G_u, nicht den
Ausgangsgrad.

Bei `n_walks` > 1 (Capture-Recapture) baut jeder Fang sein **eigenes** G_u auf,
sonst waeren die Faenge ueber die geteilte Historie abhaengig.

| `jump` | Sprungziel | braucht | Kategorie |
|---|---|---|---|
| `uniform` | gleichverteilt aus V | `JumpCrawlOracle.random_node()` | Vergleich |

Die Kategorie haengt an der Sprungart, nicht am Verfahren: `uniform` setzt
dieselbe Kenntnis von V voraus, die auch `uniform-collision` zum Vergleich
macht. Vergeben wird sie in `_JUMP_CATEGORY` (`estimators/__init__.py`) -- dort
kommt eine Sprungart, die ihr Ziel aus externen Daten simuliert, als real
umsetzbar hinein, ohne dass sich am Sampler etwas aendert.

Das Sprunggewicht `w` (`config.DURW_JUMP_WEIGHT`, Default 1.0) steuert den
Handel: groesseres w heisst haeufiger springen -- weniger Autokorrelation und
bessere Abdeckung, dafuer geht mehr Budget in Spruenge
(`COST_RANDOM_NODE`) statt in Schritte (`COST_CACHE_HIT` beim Wiederbesuch).

### 4. Thinning -- aus der Trajektorie werden Sample-Sets

Reine Nachbearbeitung: der Walk ist gelaufen, die Queries sind bezahlt.

| `thinning` | Sets | Zweck |
|---|---|---|
| `none` | 1 (die ganze Trajektorie) | nichts verwerfen |
| `simple` | 1 (jedes `step`-te Sample) | Abstand vergroessern, verwirft `(n-1)/n` des Budgets |
| `shifted` | `step` (Offsets 0..n-1) | wie `simple`, aber ohne Verlust: je Set eine Schaetzung |
| `by-walk` | `n_walks` (ein Set je Fang) | fuer Capture-Recapture -- Faenge, die nicht Fortsetzung voneinander sind; nicht in `THINNINGS`, wird direkt von `methods/capture_recapture.py` gesetzt |

`margin` steht in den Estimator-Namen im selben Slot, ist aber **kein**
Thinning: es verwirft keine Samples, sondern bei der Kollisionszaehlung Paare
mit Abstand <= m im Walk (`config.SAFETY_MARGIN = 10`).

### 5. Weighting -- die Sampling-Verzerrung korrigieren

Wird von den `build()`-Funktionen automatisch nach der Formel gewaehlt
(`FORMULAS[...].weighted`).

| Schema | Gewicht | Passt zu |
|---|---|---|
| `UniformWeighting` | `w_i = 1` | gleichverteilten Stichproben |
| `InverseDegreeWeighting` | `w_i = 1/deg(u_i)` | Stichproben mit pi(u) ~ deg(u) (Random Walk, DWI) |

### 6. Formel -- die Zahl

Zwei Signaturen: `EstimationFormula` rechnet je Sample-Set (die Pipeline
aggregiert danach), `SetsFormula` rechnet einmal ueber alle Sets gemeinsam.

| `formula` | Art | Rechnung | Gewichte |
|---|---|---|---|
| `uis-collision` | je Set | `C(k,2) / n_col` | nein |
| `wis-col-katzir` | je Set | gradkorrigierter Collision-Schaetzer (Katzir 2011) | ja |
| `lincoln-petersen` | ueber Sets | `\|S1\|*\|S2\| / \|S1 ∩ S2\|`, genau 2 Faenge | nein |
| `chapman` | ueber Sets | verzerrungskorrigierte Variante davon, genau 2 Faenge | nein |
| `schnabel` | ueber Sets | k Faenge (Schnabel 1938), `n_captures` frei | nein |
| `cross` | ueber Sets | Kollisionen *zwischen* den Faengen, `n_captures` frei | nein |
| `cross-wis` | ueber Sets | dasselbe mit Gradkorrektur | ja |

Alle liefern NaN ohne beobachtete Kollision; alle kennen den `margin`.

### 7. Aggregation und Budget

| Stufe | Werte | Default |
|---|---|---|
| `aggregate` | jede numpy-Funktion ueber die Einzelschaetzungen | `np.median` |
| `budget_metric` | nur `"queries"` (s. `oracles/base.py`) | `"queries"` |

### Was daraus gebaut ist

Die REGISTRY (`estimators/__init__.py`) kombiniert diese Stufen zu den
lauffaehigen Verfahren -- `python run_experiment.py --list` zeigt
die Namen:

| Namensmuster | Oracle | Sampler | Thinning | Formel |
|---|---|---|---|---|
| `uniform-collision[__weighted]` | Uniform | Uniform | none | `uis-collision` / `wis-col-katzir` |
| `wis-katzir__indep` | DegWeightedIndependent | Uniform | none | `wis-col-katzir` |
| `{uis,wis-katzir}__walk5` | ShortWalkIndependent(steps=5) | Uniform | none | beide je Set |
| `rw-plain__<dead_end>__<none\|simple\|shifted>` | Crawl | RandomWalk | alle drei | `uis-collision` |
| `rw-plain__<dead_end>__margin[N]` | Crawl | RandomWalk | none + Margin | `uis-collision` |
| `wis-katzir__rw-<dead_end>[__margin]` | Crawl | RandomWalk | none (+ Margin) | `wis-col-katzir` |
| `rw-weighted__restart__none` | Crawl | RandomWalk | none | `wis-col-katzir` |
| `capture-recapture__<dead_end>[__<formel>]` | Crawl | RandomWalk(`n_walks`) | by-walk | LP / chapman / schnabel / cross / cross-wis |
| `capture-recapture__uniform[__<formel>]` | Uniform | Uniform(`n_walks`) | by-walk | dieselben fuenf |
| `durw-plain__<jump>__<none\|simple\|shifted\|margin[N]>` | JumpCrawl | DURW | alle drei (+ Margin) | `uis-collision` |
| `wis-durw__<jump>[__<simple\|shifted\|margin[N]>]` | JumpCrawl | DURW | alle drei (+ Margin) | `wis-col-katzir` |
| `capture-recapture__durw-<jump>[__<formel>]` | JumpCrawl | DURW(`n_walks`) | by-walk | dieselben fuenf |

Das Kreuzprodukt laeuft ueber `<dead_end>` ∈ {`restart`, `backtrack`,
`history`} bzw. `<jump>` ∈ {`uniform`}. Zwei Zahlen lassen sich im Namen ueberschreiben und stehen deshalb
nicht einzeln in der REGISTRY: `...__margin20` (Safety Margin 20) und
`...__schnabel8` (8 Faenge statt `config.DEFAULT_CAPTURES = 4`).

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

## Nichts wird ueberschrieben

**Ergebnisse werden angehaengt.** Beim Start prueft `run_experiment.py`, was in
der Zieldatei schon steht, und rechnet nur das Fehlende. Ein Lauf ist durch
`(view, estimator, budget_rel, run)` bestimmt -- innerhalb einer Datei, die
ohnehin nach Graph, Seed und Einstiegsknoten getrennt ist.

```bash
python run_experiment.py --graphs slashdot --estimators uniform-collision
# -> rechnet und schreibt

python run_experiment.py --graphs slashdot --estimators uniform-collision
# [Slashdot0811] alles schon gerechnet -- data/results/Slashdot0811__estimates.csv
# [Slashdot0811] nichts zu tun (mit --replace neu rechnen)

python run_experiment.py --graphs slashdot \
    --estimators uniform-collision wis-katzir__rw-restart
# [Slashdot0811] 3 von 12 Schaetzungen liegen schon vor, gerechnet werden 9
```

Das gilt bis in den Runner hinein: Pakete, die nur bekannte Zeilen liefern
wuerden, werden gar nicht erst gestartet; bei gemischten Paketen (geteilte
Walks, genestete Budgets) faellt die bekannte Zeile nach der Rechnung weg.
`--replace` erzwingt das Neurechnen.

**Geprueft wird, bevor der Graph geladen wird.** Das Laden dauert bei den
grossen Wissensgraphen ueber eine Minute; steht schon alles in der Datei, wird
es gespart:

```
$ python run_experiment.py --graphs gpt-4-io --estimators uniform-collision
[gpt4_io] alles schon gerechnet -- data/results/gpt4_io__estimates.csv
[gpt4_io] nichts zu tun, Graph wird nicht geladen (mit --replace neu rechnen)
   4 s statt 140 s
```

Die Default-Budgets brauchen dafuer |V| (gross oder klein) -- das liest
`_known_size()` aus der ersten Zeile einer vorhandenen Ergebnisdatei, statt
den Graphen zu laden. Gibt es noch keine Ergebnisse, ist ohnehin zu rechnen.
Weicht das gespeicherte |V| spaeter vom geladenen ab, bricht der Lauf mit
einem Hinweis auf `--deprecate` ab: dann wurden die alten Zahlen auf einem
anderen Graphen gerechnet und duerfen nicht ergaenzt werden.

**Bilder auch nicht.** Existiert der Dateiname schon, entsteht `...-2.png`,
`...-3.png` (siehe `config.unique_path`). Zwei Laeufe mit verschiedenen
Parametern liegen damit nebeneinander statt uebereinander.

### Wenn sich der Verlauf aendert

Aenderungen am Graphaufbau, am Kostenmodell oder am Sampler machen alte Zeilen
unvergleichbar -- anhaengen waere dann falsch. Dafuer:

```bash
python run_experiment.py --graphs slashdot --deprecate "Schlingen entfernt"
```

Das schiebt alle vorhandenen CSVs nach
`data/results/deprecated/<Zeit>__<Code-Fingerabdruck>/` (mit `GRUND.txt`) und
faengt neu an. Verschoben statt geloescht: die Zahlen bleiben nachvollziehbar,
stehen aber nicht mehr im Weg.

Als Kontrolle traegt jede Zeile in der Spalte `code` den Fingerabdruck der
Codeversion, die sie erzeugt hat (SHA-256 ueber alle `.py` unter `Code/`).
Stehen in einer Datei mehrere Werte, stammen ihre Zeilen aus verschiedenen
Codeversionen -- unbedenklich, solange die Aenderung den Verlauf nicht
beruehrt hat, und ein Hinweis, falls doch.

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

## Ein Walk, mehrere Auswertungen: `--share-walks`

Thinning, Safety Margin und die Wahl der Formel (mit/ohne Gewichte) sind
**reine Nachbearbeitung einer Trajektorie** -- der Walk selbst haengt nur an
Oracle, Sampler, Budget und Zufallsstrom. Mit `--share-walks` teilen sich
deshalb alle Estimators mit gleichem `walk_key` einen einzigen Walk:

```bash
python run_experiment.py --graphs slashdot --views undirected \
    --share-walks --checkpoint-budgets \
    --estimators wis-katzir__rw-restart wis-katzir__rw-restart__margin10 \
                 rw-plain__restart__none rw-plain__restart__shifted
```

Wer sich mit wem zusammentut, ergibt sich von selbst aus Oracle und Sampler:

| Walk-Gruppe | Estimators |
|---|---|
| `CrawlOracle \| random_walk_restart` | `rw-plain__restart__*`, `wis-katzir__rw-restart*`, `rw-weighted__restart__none` |
| `UniformNodeOracle \| uniform` | `uniform-collision`, `uniform-collision__weighted` |
| `ShortWalkIndependentOracle(steps=5) \| uniform` | `uis__walk5`, `wis-katzir__walk5` |
| `JumpCrawlOracle \| durw_uniform` | `durw-plain__uniform__*`, `wis-durw__uniform*` |

Die Gruppierung geht ueber die Formel hinweg -- mit und ohne Gewichte laufen
auf denselben Samples. Genau das behauptet der Kommentar bei `*__walk5` in der
REGISTRY schon lange ("beide Formeln auf denselben Samples"), stimmte aber
erst mit diesem Schalter.

**Zwei Dinge gewinnt man dabei.** Rechenzeit: gemessen auf Slashdot
symmetrisiert, 12 Estimators x 6 Budgets x 5 Laeufe, `--jobs 1`, zusammen mit
`--checkpoint-budgets`:

| | Laufzeit | CPU in den Estimators |
|---|---|---|
| ohne `--share-walks` | 10.6 s | 8.3 s |
| mit | **3.5 s** | **1.5 s** |

Aus 12 Walk-Sorten werden zwei Gruppen. Und, wichtiger: der Vergleich
zwischen den Varianten wird **gepaart**. Ob `margin50` besser ist als
`margin20`, wird sonst an zwei verschiedenen Walks gemessen, der Unterschied
enthaelt also RNG-Rauschen -- obwohl die Frage rein deterministisch ist.

**Was sich aendert.** Der abgeleitete Zufallsstrom haengt in diesem Modus am
Walk-Schluessel statt am Estimator-Namen -- auch bei Gruppen der Groesse eins.
Das ist Absicht: sonst haenge das Ergebnis eines Estimators davon ab, welche
anderen zufaellig mit ausgewaehlt wurden. Ergebnisse aus beiden Modi sind
deshalb nicht Zeile fuer Zeile vergleichbar, wohl aber Verteilung gegen
Verteilung. Die Zeilen einer Gruppe sind korreliert; die CSV haelt das in der
Spalte `walk_group` fest, die Grafiken vermerken es oben rechts.

Je Wiederholung laeuft weiterhin ein eigener Walk -- die Streuung ueber die
Laeufe bleibt also aussagekraeftig, es entfaellt nur die *unnoetige* Streuung
zwischen den Varianten.

**Exakt, nicht genaehert:** `check_shared.py` rechnet fuer jede Gruppe die
geteilte gegen die einzelne Auswertung mit demselben Seed und vergleicht
Schaetzwert, Kosten, Abbruchgrund und Sample-Zahl auf Gleichheit.

```bash
python check_shared.py --graph slashdot
```

`capture_recapture` faellt aus dem Modus heraus (es teilt sein Budget vorab
auf zwei Walks auf und hat keinen `walk_key`) und laeuft unveraendert einzeln.

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

| Abhaengigkeit | Wie die Autokorrelation behandelt wird |
|---|---|
| `none` | gar nicht: ein Set, die ganze Trajektorie |
| `simple` | ein Set: jedes n-te Sample (n=5), verwirft 4/5 des Budgets |
| `shifted` | n Sets mit Offset 0..n-1; je Set eine Schaetzung, aggregiert per Median |
| `margin` | **Safety Margin**, s.u. -- verwirft kein Sample, sondern Paare |

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

### Safety Margin

Der Collision-Schaetzer setzt *unabhaengige* Ziehungen voraus. Ein Walk
liefert das Gegenteil: `u_i` und `u_{i+1}` sind Nachbarn, `u_i` und `u_{i+2}`
oft derselbe Knoten (hin und zurueck). Solche Treffer sagen nichts ueber |V|,
sondern nur, dass der Walk noch nicht gemischt hat -- gezaehlt machen sie
`n_col` zu gross und die Schaetzung zu klein.

Der Safety Margin laesst genau diese Paare aus, **ohne ein Sample zu
verwerfen**: gezaehlt werden nur Paare `(i, j)` mit `j - i > m`, und die
Normierung sinkt entsprechend mit.

    n_hat = P_m / n_col_m                          (UIS)
    n_hat = P_m * mean(w) * mean(1/w) / n_col_m    (WIS, Katzir)

    P_m     = C(k,2) - (m*k - m(m+1)/2)      betrachtete Paare
    n_col_m = #{(i,j) : i<j, j-i > m, u_i = u_j}

Die Korrektur von `C(k,2)` auf `P_m` gehoert zwingend dazu -- sonst waere der
Zaehler zu gross und die Schaetzung um genau den ausgelassenen Anteil zu hoch.

Der Preis ist winzig: bei k = 100 000 und m = 10 fallen 10^6 von 5*10^9 Paaren
weg, also 0,02 %. `simple` wirft dagegen 80 % der bezahlten Samples weg.

**Gemessen** auf Slashdot0811 symmetrisiert, Median ueber 5 Laeufe,
Schaetzung/|V|:

| Estimator | 1 % | 5 % | 10 % |
|---|---|---|---|
| `uniform-collision` (Referenz) | 0.966 | 1.110 | 1.053 |
| `wis-katzir__rw-restart` | 0.427 | 0.793 | 0.884 |
| `wis-katzir__rw-restart__margin` (m=10) | **0.935** | **1.024** | **1.003** |
| `...__margin50` | 0.910 | 1.063 | 1.012 |
| `...__margin200` | 0.971 | 0.944 | 1.006 |

Bei 1 % Budget mehr als eine Verdopplung, und der Walk liegt damit fast
gleichauf mit gleichverteiltem Ziehen. m = 10 reicht; groessere Margins
bringen nichts mehr.

**Groesse einstellen:** Default ist `config.SAFETY_MARGIN = 10`. Ein
abweichender Wert steht im Estimator-Namen und wird zur Laufzeit aufgeloest:

```bash
python run_experiment.py --graphs slashdot --views undirected \
    --estimators wis-katzir__rw-restart__margin20 rw-plain__restart__margin50
```

Solche Namen stehen nicht in der REGISTRY -- `--list` zeigt nur die
Default-Variante `__margin`.

**Schnell bleibt es** durch eine Differenz statt Paar-Aufzaehlung: bei
k = 8,5 Mio. Samples gibt es 3,6*10^13 Paare. Gerechnet wird

    n_col_m = alle Kollisionen (np.unique, wie bisher)
            - Kollisionen mit Abstand 1..m (m verschobene Array-Vergleiche)

also O(k log k) + O(k*m). Gemessen bei k = 1 Mio.: 0.070 s ohne Margin,
0.074 s mit m = 10, 0.135 s mit m = 200 -- der Zuschlag verschwindet neben dem
Sortieraufwand, der ohnehin anfaellt.

`margin` und `thinning` sind zwei Antworten auf dasselbe Problem und werden
nicht kombiniert (bei Schritt s laegen die Samples eines Sets schon s
auseinander, ein Margin m verlangte dann s*m Schritte). Die REGISTRY setzt den
Margin deshalb immer mit `thinning="none"`.

### Capture-Recapture: derselbe Bauplan

`n_hat = |S1| * |S2| / |S1 geschnitten S2|` ist nicht nur eine andere Formel --
das Verfahren schreibt auch die *Form der Ziehung* vor: zwei Faenge mit
eigenem Einstieg. Ein Halbieren derselben Trajektorie taugt nicht, die zweite
Haelfte liefe dort weiter, wo die erste aufgehoert hat.

Trotzdem passt es in dieselbe Pipeline, weil `Thinning.apply()` ohnehin eine
*Liste* von Sample-Sets liefert. Der Unterschied sitzt an drei Stellen:

| Stufe | Collision Counting | Capture-Recapture |
|---|---|---|
| Sampler | `RandomWalkSampler(n_walks=1)` | `n_walks=2`, Samples tragen `walk` |
| Thinning | `none` / `simple` / `shifted` / `margin` | `ByWalkThinning` -- ein Set je Fang |
| Formel | je Set rechnen, dann aggregieren | **einmal ueber alle Sets** (`SetsFormula`) |

Walk i endet, sobald er seinen Anteil `(i+1)/n` am Budget verbraucht hat; der
letzte laeuft bis zum Budgetende. Beide Faenge teilen sich ein Oracle und
damit den Cache -- der zweite kommt mit seiner Haelfte dadurch weiter, was die
Schaetzung nicht beruehrt (gezaehlt werden Knoten, nicht Anfragen).

**Vier Formeln ueber denselben Faengen** (`estimators/formulas.py`):

| Name | Formel | Faenge | Gewichte |
|---|---|---|---|
| `capture-recapture__<de>` | `n1*n2/m` (Lincoln-Petersen) | 2 | nein |
| `...__chapman` | `(n1+1)(n2+1)/(m+1) - 1` | 2 | nein |
| `...__schnabel` | `sum C_t*M_t / sum R_t` | k (Default 4) | nein |
| `...__cross` | Kollisionen *zwischen* den Faengen | k | nein |
| `...__cross-wis` | dasselbe, gradkorrigiert | k | **ja** |

**Chapman** ist die verzerrungskorrigierte Form von Lincoln-Petersen und
anders als dieses **immer definiert**: ohne Ueberschneidung liefert LP NaN,
Chapman eine grosse, aber endliche Zahl. **Schnabel** verallgemeinert auf k
Faenge und faellt fuer k = 2 exakt auf LP zurueck; die Zahl der Faenge steht
im Namen (`capture-recapture__restart__schnabel8`).

**Gewichte kann von diesen dreien keine nehmen** -- sie rechnen mit Mengen
*verschiedener* Knoten, und fuer eine Korrektur nach 1/pi braeuchte man die
Einschlusswahrscheinlichkeit `1-(1-pi)^k`. Zaehlt man stattdessen Paare *mit*
Vielfachheit (`cross`), steht wieder Katzirs Identitaet zur Verfuegung und die
Gewichtung ist dieselbe wie beim Collision Counting. Damit ist `cross-wis` die
einzige Capture-Recapture-Variante, die die Gradverzerrung korrigiert.

Nachgerechnet auf synthetischen Faengen (N = 2000, Median ueber 200
Wiederholungen, je Fang 1500 Ziehungen):

| | gleichverteilte Faenge | Faenge mit pi ~ deg |
|---|---|---|
| Lincoln-Petersen | 1996 | 1690 (0.84 x N) |
| Chapman | 1995 | 1689 |
| Schnabel (k=2) | 1996 | 1690 |
| `cross` | 1988 | 1537 |
| `cross-wis` | -- | **1998 (1.00 x N)** |

Auf Slashdot symmetrisiert, Budget 1 %, Median ueber 5 Laeufe, zeigt sich
dasselbe: die drei mengenbasierten Varianten liegen bei 0.09-0.11 x |V|,
`cross-wis` bei 1.21.

`--checkpoint-budgets` bleibt fuer Capture-Recapture ausgeschlossen: der
Umschaltpunkt liegt bei der Haelfte des *Gesamtbudgets*, ein Praefix bei
Budget b waere also nicht derselbe Lauf wie ein eigenstaendiger b-Lauf (der
bei b/2 umschaltete). Das Flag `PipelineEstimator.supports_nested` haelt das
fest, der Runner nimmt den Estimator dort heraus und sagt es im Log.
`--share-walks` geht dagegen: das Verfahren bildet wegen `n_walks=2` eine
eigene Walk-Gruppe.

## Kategorien

`Category.COMPARISON` (nur zum Vergleich) vs. `Category.REALIZABLE` (real
umsetzbar) ist bewusst **nur ein Label in der REGISTRY**, keine Ordner- oder
Klassentrennung. Auch die Oracle-Module heissen nach dem Zugriffsmodell
(`global_access` / `local_access`), nicht nach der Kategorie. Ob ein Verfahren umsetzbar ist, entscheidet das Oracle -- und
dasselbe Verfahren kann mit einem anderen Oracle in die andere Kategorie
fallen. `estimators.build(name)` haengt das Label nach der Konstruktion an die
Instanz; ein direkt konstruierter Estimator hat `category is None`.

Das Label steuert die *Auswahl* (`estimators.build_all(category=...)`), nicht
die Darstellung: der Plot teilt bewusst nicht danach auf, sondern zeigt alle
gewaehlten Estimators zusammen in einem Panel je Kantensicht.

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

**Was ein Sample kostet -- und warum das nicht ueberall gleich ist.** Eine
Nachbarabfrage kostet eine Einheit und liefert zweierlei zugleich: den Grad und
die Moeglichkeit, zu einem zufaelligen Nachbarn weiterzugehen. Beim Random Walk
ist der Grad damit gratis -- der Schritt musste ohnehin bezahlt werden. Beim
gleichverteilten Ziehen ist er es nicht: dort sind Ziehung und Gradabfrage zwei
verschiedene Anfragen.

Bezahlt wird deshalb nur, was auch benutzt wird. `Sample.degree` liest allein
`InverseDegreeWeighting`; jedes Verfahren mit `UniformWeighting` fragt den Grad
gar nicht erst ab (`UniformSampler(with_degree=False)`, gesetzt aus
`weighting.needs_degree`):

| Verfahren | Anfragen je Sample |
|---|---|
| Random Walk (`rw-*`, `capture-recapture__<dead_end>`) | 1 -- Grad faellt beim Schritt mit ab |
| DURW (`durw-*`, `wis-durw__*`) | 1 je Schritt (Grad faellt mit ab), plus 1 (`COST_RANDOM_NODE`) je Sprung -- im Mittel `w/(w + deg_Gu)` der Schritte |
| `*__walk5` | 1 -- die Antwort bringt die Nachbarliste mit |
| uniformes Ziehen ohne Gradgewichtung | 1 -- nur die Ziehung |
| uniformes Ziehen **mit** Gradgewichtung (`wis-katzir__indep`, `*__cross-wis`) | 2 -- Ziehung + Gradabfrage |

Vorher zahlte jedes uniforme Verfahren zwei Einheiten, auch wenn die Formel den
Grad nie anfasste -- `uniform-collision` bekam damit halb so viele Samples wie
noetig und, weil Kollisionen mit k^2 skalieren, ein Viertel der Kollisionen.
Die Aufschluesselung je Verfahren steht in
`data/results/<graph>__budget_breakdown.csv` (`results.budget_breakdown`),
`plot_results.py` gibt sie zusaetzlich auf der Konsole aus.

**Warum der Cache-Treffer nicht gratis ist.** Mit Preis 0 laeuft ein Walk, der
sich in einer kleinen, bereits bekannten Region verfaengt, endlos gratis weiter
und sammelt beliebig viele wertlose, hochkorrelierte Samples -- die Schaetzung
haengt dann an der Abbruchkonstante statt am Verfahren. Mit einem Preis > 0
terminiert das Budget jeden Lauf von selbst, und **jeder Estimator schoepft sein
Budget aus**. Erst dadurch sind "erlaubtes" und "genutztes" Budget vergleichbar
-- vorher gab `uniform-collision` 98 % aus und ein Random Walk 5 %, bei gleicher
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
`n_random_node` / `n_neighbors` -- die Spalten, aus denen
`results.budget_breakdown()` die Aufteilung rechnet.

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
