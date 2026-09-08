"""Externe Namenslisten als Ziehungsquelle.

Der Zufallssprung von DURW zieht gleichverteilt aus V und setzt damit Kenntnis
der Knotenmenge voraus -- genau das, was ein realer Crawler nicht hat. Eine
externe Namensliste ersetzt das: Namen ziehen, im Graphen nachschlagen, bei
Fehlschlag neu ziehen. Die Liste ist externes Wissen, keine Kenntnis von V, und
ein darauf gebautes Verfahren deshalb real umsetzbar.

Dieses Modul beschreibt nur, *wie aus einer Datei Namenskandidaten werden*. Den
Abgleich mit einem Graphen baut build_name_index.py, gezogen wird in
oracles.name_list.

Zwei Formen von Quelle:

    Zeilenliste  -- eine Zeile ein Name (Wikipedia-Titeldumps)
    CSV          -- mehrere Namensspalten je Eintrag, in Prioritaetsreihenfolge

Bei der CSV-Form gehoeren die Spalten *zusammen*: sie sind Schreibweisen
derselben Entitaet. Gezogen wird deshalb pro Eintrag, nicht pro Name -- sonst
waeren Entitaeten mit drei Schreibweisen dreifach gewichtet. Beim Abgleich
gewinnt der erste Treffer in der angegebenen Reihenfolge.

Die Normalisierung kommt aus title_overlap.Normalizer, damit Ziehung und
Abgleich garantiert dieselben Regeln benutzen. Fuer die Titeldumps ist das
VARIANTS["+whitespace"] -- also NFC, "_" -> Leerzeichen, casefold und
Whitespace einklappen, aber *ohne* das Streichen von Klammerzusaetzen: das
erkauft nur 0,3 bis 1,9 Prozentpunkte Abdeckung und wirft dafuer Millionen
Titel zusammen (siehe title_overlap.collisions).

Schnittstelle:
    class NameList                  -- .path, .norm, .columns, .limit, .entries()
    SOURCES: dict[str, NameList]
    resolve(name) -> NameList
    head(source, fraction=None, n=None) -> NameList
    name_lookup(graph, norm) -> dict[str, int]
    overlap(graph, source, lookup=None) -> dict
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import config
from title_overlap import VARIANTS, Normalizer, iter_titles

# Wikidata-Labels tragen bereits Leerzeichen und keine Wikipedia-Konventionen;
# dort waere "_" -> Leerzeichen wirkungslos und das Einklappen von Whitespace
# nur Rechenzeit. Uebrig bleibt, was verlangt war: Gross-/Kleinschreibung egal.
_LABEL_NORM = Normalizer(nfc=True, underscores_to_spaces=False, casefold=True,
                         collapse_whitespace=False, strip_parens=False)


@dataclass(frozen=True)
class NameList:
    """Eine Datei plus die Regeln, wie daraus Namenskandidaten werden.

    `columns` leer     -- Zeilenliste, ein Name je Zeile.
    `columns` mit str  -- Tabelle *mit* Kopfzeile; die genannten Spalten werden
                          in genau dieser Reihenfolge als Kandidaten geliefert.
    `columns` mit int  -- Tabelle *ohne* Kopfzeile; Position statt Name.

    Leere Felder fallen weg, Dubletten innerhalb eines Eintrags auch (bei
    "YouTube" sind alle drei top-q-Spalten gleich -- dreimal nachzuschlagen
    waere reine Arbeit).

    `limit` schneidet nach n Eintraegen ab. Sinnvoll nur bei *sortierten*
    Listen: top-q ist nach QRank geordnet, die In-Grad-Listen nach Eingangsgrad.
    Ein Praefix ist dort "die n prominentesten Entitaeten" -- bei einer
    unsortierten Liste waere es dagegen eine willkuerliche Teilmenge. Siehe
    head().

    `realizable` sagt, ob die Quelle ohne Kenntnis von V zu haben ist. Die
    Wikipedia-Dumps und top-q sind es (externes Wissen). Die In-Grad-Listen
    sind es *nicht*: sie sind aus dem Graphen selbst erzeugt und setzen damit
    genau das voraus, was geschaetzt werden soll. Sie taugen als Obergrenze
    und als Kreuztest (die Liste des einen Graphen gegen den anderen), nicht
    als Verfahren. estimators/__init__.py liest das Flag fuer die Kategorie.
    """

    path: Path
    norm: Normalizer
    columns: tuple = ()
    delimiter: str = ","
    skip_header: bool = True
    limit: int | None = None
    realizable: bool = True
    note: str = field(default="", compare=False)

    def entries(self) -> Iterator[tuple[str, ...]]:
        """Je Eintrag die *normalisierten* Kandidaten in Prioritaetsreihenfolge.

        Ein Eintrag kann leer sein (alle Felder leer). Er wird trotzdem
        geliefert, damit die Position in der Liste erhalten bleibt: der
        Namensindex ist positionsbasiert, und ein stillschweigend
        uebersprungener Eintrag verschoebe alles danach.
        """
        n_yielded = 0
        for cands in self._raw_entries():
            if self.limit is not None and n_yielded >= self.limit:
                return
            n_yielded += 1
            yield cands

    def _raw_entries(self) -> Iterator[tuple[str, ...]]:
        if not self.columns:
            for t in iter_titles(self.path, skip_header=self.skip_header):
                yield (self.norm.apply(t),)
            return

        positional = all(isinstance(c, int) for c in self.columns)
        with open(self.path, encoding="utf-8", newline="") as f:
            if positional:
                reader = csv.reader(f, delimiter=self.delimiter)
                if self.skip_header:
                    next(reader, None)
                pick = lambda row, c: row[c] if c < len(row) else ""  # noqa: E731
            else:
                reader = csv.DictReader(f, delimiter=self.delimiter)
                missing = [c for c in self.columns
                           if c not in (reader.fieldnames or ())]
                if missing:
                    raise ValueError(
                        f"{self.path.name}: Spalten {missing} fehlen -- "
                        f"vorhanden sind {reader.fieldnames}"
                    )
                pick = lambda row, c: row.get(c) or ""  # noqa: E731
            for row in reader:
                seen: list[str] = []
                for col in self.columns:
                    v = (pick(row, col) or "").strip()
                    if not v:
                        continue
                    nm = self.norm.apply(v)
                    if nm and nm not in seen:
                        seen.append(nm)
                yield tuple(seen)

    def count(self) -> int:
        """Zahl der Eintraege -- fuer den Fortschritt beim Indexbau."""
        return sum(1 for _ in self.entries())


SOURCES: dict[str, NameList] = {
    "dewiki": NameList(
        config.ADDITIONALS_DIR / "dewiki-20260801-all-titles-in-ns0.txt",
        VARIANTS["+whitespace"],
        note="Titel der deutschen Wikipedia (ns0, inkl. Weiterleitungen)"),
    "enwiki": NameList(
        config.ADDITIONALS_DIR / "enwiki-latest-all-titles-in-ns0.txt",
        VARIANTS["+whitespace"],
        note="Titel der englischen Wikipedia (ns0, inkl. Weiterleitungen)"),
    # Die 100 000 nach QRank hoechstplatzierten Wikidata-Entitaeten. Drei
    # Schreibweisen je Entitaet, englisch zuerst: die Testgraphen tragen
    # englische Namen, das dewiki-Feld ist die Rueckfalloption und das
    # Wikidata-Label die letzte.
    "top-q": NameList(
        config.ADDITIONALS_DIR / "top-q-entities",
        _LABEL_NORM,
        columns=("enwiki_title", "dewiki_title", "label"),
        note="Top-100k-Wikidata-Entitaeten nach QRank"),
}

# In-Grad-Listen: "<eingangsgrad>\t<name>", ohne Kopfzeile, absteigend sortiert,
# eine Zeile je Knoten. Sie stammen *aus dem Graphen selbst* und sind deshalb
# realizable=False -- gegen den eigenen Graphen ist die Trefferquote per
# Konstruktion 100 % und die Abdeckung genau der genommene Anteil. Ihr Wert
# liegt woanders:
#
#   gegen den eigenen Graphen  -- Obergrenze: so gut kann eine nach Prominenz
#                                 sortierte Liste bestenfalls sein
#   gegen den anderen Graphen  -- ein echter Test: wie weit deckt die
#                                 Prominenzliste des einen den anderen ab
for _g in ("gpt4_io", "gpt4o_io"):
    SOURCES[f"indeg-{_g}"] = NameList(
        config.ADDITIONALS_DIR / f"{_g}__in-degree.txt",
        VARIANTS["+whitespace"],
        columns=(1,), delimiter="\t", skip_header=False, realizable=False,
        note=f"Knoten von {_g}, absteigend nach Eingangsgrad")
del _g


def head(source, fraction: float | None = None, n: int | None = None) -> NameList:
    """Die ersten n bzw. den ersten Bruchteil einer *sortierten* Liste.

    Genau eines von `fraction` und `n` angeben. `fraction` bezieht sich auf die
    Zahl der Eintraege der Datei, nicht auf |V|.

    Sinnvoll nur, wo die Reihenfolge etwas bedeutet: top-q ist nach QRank
    sortiert, die In-Grad-Listen nach Eingangsgrad. Bei den Wikipedia-Dumps
    (alphabetisch) waere ein Praefix dagegen "alles was mit ! und A anfaengt"
    -- deshalb warnt die Funktion dort.
    """
    src = resolve(source) if isinstance(source, str) else source
    if (fraction is None) == (n is None):
        raise ValueError("genau eines von fraction und n angeben")
    if fraction is not None:
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction muss in (0, 1] liegen, ist {fraction}")
        n = max(1, int(round(fraction * src.count())))
    if not src.columns and src.limit is None:
        import warnings
        warnings.warn(
            f"{src.path.name} ist eine unsortierte Zeilenliste -- ein Praefix "
            "davon ist eine willkuerliche Teilmenge, keine 'Top-n'.",
            stacklevel=2)
    return replace(src, limit=int(n))


def name_lookup(graph, norm: Normalizer) -> dict[str, int]:
    """{normalisierter Knotenname: ID}. Die teure Struktur (bei 6,5 Mio Knoten
    ein bis zwei GB), deshalb einmal bauen und weiterreichen.

    Bei Kollisionen gewinnt der erste Knoten -- willkuerlich, aber
    deterministisch. Wie oft das vorkommt, sagt len(graph.names) - len(lookup).
    """
    out: dict[str, int] = {}
    for i, k in enumerate(graph.names):
        if isinstance(k, str):
            out.setdefault(norm.apply(k), i)
    return out


def overlap(graph, source, lookup: dict[str, int] | None = None) -> dict:
    """Trefferquote und Abdeckung einer Liste gegen einen Graphen.

    Die beiden Zahlen messen Verschiedenes und sind nicht ineinander
    umrechenbar:

        titles_matched -- Anteil der *Eintraege*, die im Graphen vorkommen.
                          Bestimmt die Kosten: 1/titles_matched Ziehungen je
                          Treffer. Positionsbasiert, Dubletten zaehlen mit --
                          ein doppelt vorkommender Name wird auch doppelt so
                          oft gezogen.
        nodes_covered  -- Anteil der *Knoten*, die die Liste erreicht, bezogen
                          auf das echte |V|. Bestimmt die Gueltigkeit: mehr als
                          diesen Teil des Graphen kann eine Ziehung daraus nie
                          sehen.

    `lookup` vorbauen, wenn mehrere Listen oder mehrere Bruchteile gegen
    denselben Graphen laufen -- sonst entsteht die teure Struktur jedes Mal neu.
    """
    src = resolve(source) if isinstance(source, str) else source
    if lookup is None:
        lookup = name_lookup(graph, src.norm)

    n_entries = 0
    hit_ids: set[int] = set()
    n_hits = 0
    by_rank: dict[int, int] = {}
    for cands in src.entries():
        n_entries += 1
        for rank, c in enumerate(cands):
            found = lookup.get(c)
            if found is not None:
                n_hits += 1
                hit_ids.add(found)
                by_rank[rank] = by_rank.get(rank, 0) + 1
                break
    return {
        "n_entries": n_entries,
        "n_hits": n_hits,
        "n_nodes_hit": len(hit_ids),
        "n_nodes": graph.n_nodes,
        "titles_matched": n_hits / n_entries if n_entries else 0.0,
        "nodes_covered": len(hit_ids) / graph.n_nodes if graph.n_nodes else 0.0,
        "draws_per_hit": n_entries / n_hits if n_hits else float("inf"),
        "hits_by_candidate_rank": dict(sorted(by_rank.items())),
    }


def resolve(name: str) -> NameList:
    src = SOURCES.get(name)
    if src is None:
        raise KeyError(
            f"{name!r} ist keine bekannte Namensliste. Bekannt sind: "
            f"{', '.join(sorted(SOURCES))}."
        )
    if not src.path.exists():
        raise FileNotFoundError(
            f"Quelle {name!r} verweist auf {src.path}, die Datei fehlt. "
            f"Die Rohdaten liegen in {config.ADDITIONALS_DIR} und sind nicht "
            "versioniert."
        )
    return src
