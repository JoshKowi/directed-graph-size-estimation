"""Ueberschneidung einer externen Titelliste mit den Knoten eines Graphen.

Hintergrund: der Zufallssprung von DURW (sampling.jumps.UniformJump) zieht
gleichverteilt aus V und setzt damit Kenntnis der Knotenmenge voraus -- genau
das, was ein realer Crawler nicht hat. Eine externe Titelliste (etwa ein
Wikipedia-Dump) koennte den Sprung emulieren: Titel ziehen, im Graphen
nachschlagen, bei Fehlschlag neu ziehen. Ob das taugt, entscheiden zwei
*verschiedene* Quoten, die hier beide berechnet werden:

    Trefferquote (titles_matched)
        Anteil der Titel, die im Graphen vorkommen. Bestimmt die *Kosten*:
        bei 10 % braucht ein Sprung im Mittel 10 Ziehungen.

    Abdeckung (nodes_covered)
        Anteil der Graph-Knoten, die ueber die Liste erreichbar sind. Bestimmt
        die *Gueltigkeit*: der emulierte Sprung ist gleichverteilt auf dieser
        Teilmenge, nicht auf V. Nur bei Abdeckung nahe 100 % ist er das, was
        DURW voraussetzt -- sonst ist pi(v) ~ (w + deg_Gu(v)) falsch, weil ein
        Teil von V nie per Sprung erreicht wird.

Die Trefferquote laesst sich durch mehr Ziehungen erkaufen, die Abdeckung
nicht. Sie ist deshalb die Zahl, an der die Idee haengt.

Normalisierung: die Seiten kommen in verschiedenen Schreibweisen. Ein
Wikipedia-Titeldump benutzt Unterstriche statt Leerzeichen, die
Wissensgraphen hier echte Leerzeichen; dazu Gross-/Kleinschreibung und
Unicode-Normalform. `Normalizer` schaltet jede Regel einzeln, damit sichtbar
bleibt, was sie bringt -- eine Regel, die Treffer erzeugt, indem sie
verschiedene Entitaeten zusammenwirft, ist keine Verbesserung.

Schnittstelle:
    class Normalizer                    -- zuschaltbare Regeln, .apply(s)
    VARIANTS: dict[str, Normalizer]     -- die ueblichen Stufen
    load_titles(path, norm) -> set[str]
    graph_keys(graph, norm) -> set[str]
    overlap(titles, keys) -> dict
    compare(graph, path, variants) -> pandas.DataFrame
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import config

# Klammerzusatz am Ende: "Merkur (Planet)", "Mercury (element)". Nur am
# Zeilenende und ohne verschachtelte Klammern -- ein "(" mitten im Namen
# ("Alpha (Beta) Gamma Ltd.") bleibt unangetastet.
_TRAILING_PAREN = re.compile(r"\s*\([^()]*\)\s*$")
_WS = re.compile(r"\s+")

# Standardpfad der Titelliste. additionals/ liegt neben Code/, siehe config.
TITLES_DIR = Path(config.__file__).resolve().parent.parent / "additionals"


@dataclass(frozen=True)
class Normalizer:
    """Zuschaltbare Schreibweisen-Regeln.

    Jede Regel einzeln schaltbar, damit ihr Beitrag messbar ist. Die
    Reihenfolge ist festgelegt und nicht beliebig: NFC zuerst (sonst
    vergleicht casefold() verschiedene Kodierungen desselben Zeichens),
    Unterstriche vor dem Whitespace-Einklappen (sonst bleibt aus "a__b" ein
    doppeltes Leerzeichen stehen), Klammerzusatz zuletzt (er soll auf dem
    bereits vereinheitlichten Namen greifen).

    `casefold` statt `lower`: es faltet auch Sonderfaelle wie das deutsche
    "ss" und ist fuer Vergleiche gedacht -- bei einem deutschsprachigen Dump
    nicht nebensaechlich.
    """

    nfc: bool = True
    underscores_to_spaces: bool = True
    casefold: bool = True
    collapse_whitespace: bool = True
    strip_parens: bool = False

    def apply(self, s: str) -> str:
        if self.nfc:
            s = unicodedata.normalize("NFC", s)
        if self.underscores_to_spaces:
            s = s.replace("_", " ")
        if self.casefold:
            s = s.casefold()
        if self.collapse_whitespace:
            s = _WS.sub(" ", s).strip()
        if self.strip_parens:
            s = _TRAILING_PAREN.sub("", s)
        return s

    def label(self) -> str:
        on = [n for n in ("nfc", "underscores_to_spaces", "casefold",
                          "collapse_whitespace", "strip_parens")
              if getattr(self, n)]
        return "+".join(on) or "roh"


# Die ueblichen Stufen, aufsteigend aggressiv. Jede Stufe kommt zur
# vorherigen dazu, damit die Differenz zwischen zwei Zeilen genau der Beitrag
# der neu zugeschalteten Regel ist.
VARIANTS: dict[str, Normalizer] = {
    "roh": Normalizer(nfc=False, underscores_to_spaces=False, casefold=False,
                      collapse_whitespace=False),
    "nfc": Normalizer(underscores_to_spaces=False, casefold=False,
                      collapse_whitespace=False),
    "+casefold": Normalizer(underscores_to_spaces=False,
                            collapse_whitespace=False),
    "+unterstriche": Normalizer(collapse_whitespace=False),
    "+whitespace": Normalizer(),
    "+ohne_klammern": Normalizer(strip_parens=True),
}


def iter_titles(path: str | Path, skip_header: bool = True) -> Iterator[str]:
    """Titel zeilenweise, ohne die Datei im Speicher zu halten.

    `skip_header`: die Dumps von Wikipedia beginnen mit der Spaltenueberschrift
    "page_title". Uebersprungen wird nur, wenn die erste Zeile tatsaechlich so
    heisst -- eine Liste ohne Kopfzeile verliert so keinen echten Titel.
    """
    with open(path, encoding="utf-8") as f:
        first = next(f, None)
        if first is None:
            return
        if not (skip_header and first.rstrip("\n") == "page_title"):
            t = first.rstrip("\n")
            if t:
                yield t
        for line in f:
            t = line.rstrip("\n")
            if t:
                yield t


def load_titles(path: str | Path, norm: Normalizer | None = None) -> set[str]:
    """Normalisierte Titelmenge. Der Rueckgabewert ist ein set, Dubletten nach
    Normalisierung fallen also zusammen -- genau das soll `collisions()`
    sichtbar machen."""
    norm = norm or Normalizer()
    return {norm.apply(t) for t in iter_titles(path)}


def graph_keys(graph, norm: Normalizer | None = None) -> set[str]:
    """Normalisierte Knotennamen. `graph` ist ein graphs.graph.Graph oder ein
    Graph-Name/Kuerzel, den graphs.loader aufloest.

    Nicht-String-Schluessel (in den Testgraphen kommen keine vor) werden
    uebersprungen statt stillschweigend zu str() gemacht -- ein 42 und ein
    "42" waeren sonst dasselbe.
    """
    norm = norm or Normalizer()
    if isinstance(graph, str):
        from graphs import loader
        graph = loader.load_graph(graph)
    return {norm.apply(k) for k in graph.names if isinstance(k, str)}


def collisions(raw_count: int, normalized: set[str]) -> int:
    """Wie viele verschiedene Rohnamen die Normalisierung zusammengeworfen hat.

    Die Kehrseite jeder Regel: sie erzeugt Treffer auch dadurch, dass sie
    Unterschiede einebnet. Ein grosser Wert heisst, dass die Quote weiter oben
    zu optimistisch ist.
    """
    return raw_count - len(normalized)


def overlap(titles: set[str], keys: set[str]) -> dict:
    """Beide Richtungen der Ueberschneidung.

    titles_matched -- Anteil der Titel mit Entsprechung im Graphen (Kosten)
    nodes_covered  -- Anteil der Graph-Knoten, die die Liste erreicht (Gueltigkeit)
    draws_per_jump -- erwartete Ziehungen je erfolgreichem Sprung, 1/Trefferquote
    """
    common = titles & keys
    matched = len(common) / len(titles) if titles else 0.0
    return {
        "n_titles": len(titles),
        "n_nodes": len(keys),
        "n_common": len(common),
        "titles_matched": matched,
        "nodes_covered": len(common) / len(keys) if keys else 0.0,
        "draws_per_jump": (1.0 / matched) if matched else float("inf"),
    }


def compare(graph, path: str | Path, variants: dict[str, Normalizer] | None = None,
            log=print):
    """Eine Zeile je Normalisierungsstufe. Braucht pandas nur hier.

    Die Titeldatei wird je Variante neu gestreamt statt einmal roh im
    Speicher gehalten: bei 5,1 Mio Titeln spart das ein paar hundert MB, und
    die Graphen belegen ohnehin schon reichlich.
    """
    import pandas as pd

    variants = variants or VARIANTS
    if isinstance(graph, str):
        from graphs import loader
        graph = loader.load_graph(graph)

    n_titles_raw = sum(1 for _ in iter_titles(path))
    n_nodes_raw = sum(1 for k in graph.names if isinstance(k, str))

    rows = []
    for name, norm in variants.items():
        if log:
            log(f"  {name} ...")
        t = load_titles(path, norm)
        k = graph_keys(graph, norm)
        row = {"variante": name, "regeln": norm.label(), **overlap(t, k)}
        row["titel_kollisionen"] = collisions(n_titles_raw, t)
        row["knoten_kollisionen"] = collisions(n_nodes_raw, k)
        rows.append(row)
        del t, k
    return pd.DataFrame(rows).set_index("variante")


def format_report(df, graph_name: str) -> str:
    """Die Tabelle als lesbarer Block -- Prozente statt Rohanteile."""
    out = [f"=== {graph_name} ==="]
    hdr = (f"{'Variante':16s} {'Treffer%':>9s} {'Abdeck%':>9s} "
           f"{'gemeinsam':>11s} {'Zieh./Sprung':>13s} {'Titel-Koll.':>12s}")
    out += [hdr, "-" * len(hdr)]
    for name, r in df.iterrows():
        out.append(f"{name:16s} {r.titles_matched:8.2%} {r.nodes_covered:8.2%} "
                   f"{int(r.n_common):11,} {r.draws_per_jump:13.1f} "
                   f"{int(r.titel_kollisionen):12,}")
    return "\n".join(out)
