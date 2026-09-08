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
    class NameList                  -- .path, .norm, .columns, .entries()
    SOURCES: dict[str, NameList]
    resolve(name) -> NameList
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass, field
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

    `columns` leer  -- Zeilenliste, ein Name je Zeile.
    `columns` gesetzt -- CSV; je Zeile werden diese Spalten in genau dieser
                       Reihenfolge als Kandidaten geliefert. Leere Felder
                       fallen weg, Dubletten innerhalb eines Eintrags auch
                       (bei "YouTube" sind alle drei Spalten gleich -- dreimal
                       nachzuschlagen waere reine Arbeit).
    """

    path: Path
    norm: Normalizer
    columns: tuple[str, ...] = ()
    skip_header: bool = True
    note: str = field(default="", compare=False)

    def entries(self) -> Iterator[tuple[str, ...]]:
        """Je Eintrag die *normalisierten* Kandidaten in Prioritaetsreihenfolge.

        Ein Eintrag kann leer sein (alle Felder leer). Er wird trotzdem
        geliefert, damit die Position in der Liste erhalten bleibt: der
        Namensindex ist positionsbasiert, und ein stillschweigend
        uebersprungener Eintrag verschoebe alles danach.
        """
        if not self.columns:
            for t in iter_titles(self.path, skip_header=self.skip_header):
                yield (self.norm.apply(t),)
            return

        with open(self.path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in self.columns if c not in (reader.fieldnames or ())]
            if missing:
                raise ValueError(
                    f"{self.path.name}: Spalten {missing} fehlen -- vorhanden "
                    f"sind {reader.fieldnames}"
                )
            for row in reader:
                seen: list[str] = []
                for col in self.columns:
                    v = (row.get(col) or "").strip()
                    if not v:
                        continue
                    n = self.norm.apply(v)
                    if n and n not in seen:
                        seen.append(n)
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
