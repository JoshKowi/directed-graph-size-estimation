"""Herkunft der Ergebnisse dokumentieren: erzeugt die README je Datenordner.

Von Hand gepflegte READMEs stehen nach dem naechsten Lauf falsch da. Deshalb
werden sie aus dem erzeugt, was tatsaechlich in den Ordnern liegt, plus einem
Fingerabdruck des Codes, der sie erzeugt hat.

Der Fingerabdruck ist ein SHA-256 ueber alle .py-Dateien unter Code/ (sortiert,
inkl. Pfadnamen). Er aendert sich bei jeder Codeaenderung und ist damit ein
Ersatz-Versionsstempel, solange das Projekt nicht unter Versionskontrolle
steht. Liegt ein Git-Repository vor, wird zusaetzlich der Commit ausgegeben.

Der Commit landet bewusst nicht in den erzeugten Dateien: eine versionierte
Datei, die den aktuellen Commit nennt, kann nie stimmen, weil das Committen
genau diesen Hash aendert. `git_revision()` bleibt fuer den interaktiven
Gebrauch erhalten.

Schnittstelle:
    code_fingerprint() -> str          (12 Hex-Zeichen)
    git_revision() -> str | None       (nicht im README-Kopf, s.o.)
    write_readmes() -> list[Path]
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

import config

CODE_DIR = Path(__file__).resolve().parent

# Spalten, die der aktuelle Code schreibt. Fehlen sie, stammt die CSV aus einer
# aelteren Version -- das soll in der README stehen, statt still zu verwirren.
EXPECTED_COLUMNS = ("stopped_by", "queries_used", "cached_queries", "budget_abs")


def _num(n) -> str:
    """Tausender mit schmalem Abstand statt Komma (das trennt hier Listen)."""
    return f"{int(n):,}".replace(",", "\u2009")


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for p in sorted(CODE_DIR.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(CODE_DIR).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(CODE_DIR), *args],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_revision() -> str | None:
    """Commit, mit "-dirty" bei uncommitteten Aenderungen.

    Die von hier erzeugten READMEs sind von der Dirty-Pruefung ausgenommen --
    sonst waere die Angabe selbstbezueglich: das Schreiben der README machte
    das Repo schmutzig, die naechste README meldete "dirty", das Committen
    machte sie sauber, und der Wert pendelte bei jedem Lauf hin und her.
    Gemeint ist ohnehin: gab es *sonst* etwas Uncommittetes?
    """
    rev = _git("rev-parse", "--short", "HEAD")
    if rev is None:
        return None
    status = _git("status", "--porcelain", "--", ".", ":!*/README.md") or ""
    return f"{rev}-dirty" if status.strip() else rev


def _data_timestamp() -> str:
    """Wann die zugrunde liegenden Ergebnisse gerechnet wurden.

    Bewusst weder "jetzt" noch das Alter der Bilder: beides aenderte sich bei
    jedem Lauf, die README waere staendig geaendert und das Repo dauerhaft
    dirty. Massgeblich sind die CSVs -- die Grafiken sind daraus abgeleitet,
    ein Neuzeichnen derselben Daten ist keine neue Information.
    """
    if not config.RESULTS_DIR.is_dir():
        return "-"
    files = [p for p in config.RESULTS_DIR.glob("*.csv")]
    if not files:
        return "-"
    return datetime.fromtimestamp(max(p.stat().st_mtime for p in files)).strftime("%Y-%m-%d %H:%M")


def _header(title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "*Automatisch erzeugt von `Code/provenance.py` -- nicht von Hand aendern.*",
        "",
        f"| Daten vom | {_data_timestamp()} |",
        "|---|---|",
        f"| Code-Fingerabdruck | `{code_fingerprint()}` |",
    ]
    lines += [
        f"| Budget-Metrik | `{config.DEFAULT_BUDGET_METRIC}` |",
        f"| Preise | random_node {config.COST_RANDOM_NODE}, neighbors "
        f"{config.COST_NEIGHBORS}, cache_hit {config.COST_CACHE_HIT} |",
        f"| Budgets (Default) | {', '.join(f'{b:g}' for b in config.DEFAULT_BUDGETS)} |",
        f"| Laeufe je Punkt | {config.DEFAULT_N_RUNS} |",
        f"| Seed | {config.DEFAULT_SEED} |",
        "",
        "Der Fingerabdruck ist ein SHA-256 ueber alle `.py` unter `Code/`. Zwei",
        "Ergebnisse mit demselben Fingerabdruck stammen aus identischem Code.",
        "",
        "Der Commit steht bewusst *nicht* hier: eine versionierte Datei, die den",
        "aktuellen Commit nennt, kann nie stimmen -- beim Committen aendert sich",
        "genau der Hash, den sie angibt. Um den passenden Stand zu finden, einen",
        "Commit auschecken und `python Code/provenance.py` laufen lassen; stimmt",
        "der Fingerabdruck ueberein, ist es der richtige.",
        "",
    ]
    return "\n".join(lines)


def _results_readme() -> str:
    import pandas as pd

    parts = [_header("data/results -- Rohergebnisse"),
             "Die CSVs selbst sind **nicht** im Repository (gross und aus dem Code",
             "reproduzierbar) -- diese Datei haelt fest, woher sie stammen.",
             "", "## Dateien", ""]
    for path in sorted(config.RESULTS_DIR.glob("*.csv")):
        graph, _, kind = path.stem.partition("__")
        parts.append(f"### `{path.name}`")
        parts.append("")
        try:
            df = pd.read_csv(path)
        except Exception as exc:                      # noqa: BLE001
            parts += [f"nicht lesbar: {exc}", ""]
            continue
        if kind == "estimates":
            budgets = ", ".join(f"{b:g}" for b in sorted(df["budget_rel"].unique()))
            parts += [
                f"Schaetzungen fuer **{graph}**, {_num(len(df))} Zeilen "
                "(= Estimator x View x Budget x Lauf).",
                "",
                f"- Views: {', '.join(sorted(df['view'].unique()))}",
                f"- Budgets: {budgets} (relativ zu |V| = {_num(df['true_size'].iloc[0])})",
                f"- Laeufe je Punkt: {df.groupby(['estimator', 'view', 'budget_rel']).size().max()}",
                f"- Estimators: {', '.join(sorted(df['estimator'].unique()))}",
            ]
            if "stopped_by" in df:
                counts = df["stopped_by"].value_counts().to_dict()
                warn = "" if set(counts) <= {"budget"} else "  **<- nicht nur `budget`, s.u.**"
                parts.append(f"- Abbruchgrund: {counts}{warn}")
            missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
            if missing:
                parts += [
                    "",
                    f"> **Veraltet.** Es fehlen die Spalten {', '.join(f'`{c}`' for c in missing)}, "
                    "die der aktuelle Code schreibt. Diese Datei stammt aus einer frueheren "
                    "Codeversion und ist mit den uebrigen Ergebnissen nicht vergleichbar -- "
                    "neu rechnen.",
                ]
            parts += ["", "Erzeugt mit:", "", "```bash",
                      f"python run_experiment.py --graphs {graph} \\",
                      "    --estimators " + " ".join(sorted(df["estimator"].unique())) + " \\",
                      "    --views " + " ".join(sorted(df["view"].unique())), "```", ""]
        elif kind == "visits":
            parts += [f"Besuchshaeufigkeit je Original-Knotenname fuer **{graph}**, "
                      f"{_num(len(df))} Zeilen. Faellt beim selben Lauf ab wie die "
                      "Schaetzungen (`--no-visits` schaltet sie aus).", ""]
        elif kind == "view_comparison":
            parts += [f"Gepaarter Vergleich der Kantensichten fuer **{graph}** "
                      "(`results.compare_views`). Entsteht beim Plotten.", ""]
        else:
            parts += [f"{_num(len(df))} Zeilen.", ""]
    parts += [
        "## Spalten der `__estimates.csv`",
        "",
        "| Spalte | Bedeutung |",
        "|---|---|",
        "| `estimate`, `rel_error` | Schaetzung und relativer Fehler gegen `true_size` |",
        "| `budget_rel`, `budget_abs` | Budget relativ zu \\|V\\| und absolut |",
        "| `queries_used` | bezahlte, gewichtete Kosten (die Budget-Waehrung) |",
        "| `cached_queries` | Nachbar-Abfragen aus dem Cache (Preis `COST_CACHE_HIT`) |",
        "| `n_random_node`, `n_neighbors` | Zugriffe je Art zum vollen Preis |",
        "| `unique_nodes_used` | verschiedene beruehrte Knoten (nur Statistik) |",
        "| `stopped_by` | warum der Lauf endete -- normal `budget` |",
        "| `extra_*` | verfahrensspezifisch, z.B. `extra_n_samples` |",
        "",
        "Steht in `stopped_by` etwas anderes als `budget`, hat nicht das",
        "Kostenmodell den Lauf beendet -- die Zahlen sind dann mit Vorsicht zu",
        "lesen. Siehe `oracles/base.py`.",
        "",
    ]
    return "\n".join(parts)


def _plots_readme() -> str:
    from plot_wis import FIGURES, REFERENCE

    specs = {slug: (ests, views, title) for slug, ests, views, title in FIGURES}
    parts = [
        _header("data/plots -- erzeugte Grafiken"),
        "Jede Grafik zeigt pro Estimator und Budget die Spanne min..max ueber die",
        "Laeufe plus den Median, y = Schaetzung/|V| (log), gestrichelt die wahre",
        "Groesse bei 1.0. Die x-Achse nennt das Budget relativ und absolut.",
        "",
        f"Referenzreihe in allen `wis_*`/`deadend_*`-Grafiken: `{REFERENCE}`.",
        "",
        "## Dateien",
        "",
    ]
    for path in sorted(config.PLOTS_DIR.glob("*.png")):
        graph, _, slug = path.stem.partition("__")
        parts.append(f"### `{path.name}`")
        parts.append("")
        if slug in specs:
            ests, views, title = specs[slug]
            parts += [
                f"{title}",
                "",
                f"- Graph: **{graph}**",
                f"- Views: {', '.join(views)}",
                f"- Estimators: {', '.join(ests)}",
                "",
                f"Erzeugt mit `python plot_wis.py --graphs {graph}` "
                f"(Definition in `Code/plot_wis.py`, Eintrag `{slug}`).",
                "",
            ]
        elif slug == "ranges":
            parts += [f"Uebersichtsraster fuer **{graph}**: Spalte = Kategorie, "
                      "Zeile = Kantensicht. Erzeugt mit `python plot_results.py "
                      f"--graphs {graph}` (`plotting/ranges.py`).", ""]
        else:
            parts += ["Keine Definition in `plot_wis.FIGURES` gefunden -- vermutlich "
                      "von Hand oder mit einer aelteren Codeversion erzeugt.", ""]

    saved = config.PLOTS_DIR / "saved"
    parts += [
        "## `saved/` -- die versionierten Meilensteine",
        "",
        "Die Dateien oben werden bei jedem Plot-Lauf neu erzeugt und sind",
        "**nicht** im Repository -- sonst laege dort nach jedem Durchlauf eine",
        "weitere vollstaendige Kopie jedes Bildes. Was einen Meilenstein",
        "festhaelt oder in eine Praesentation geht, wird bewusst nach `saved/`",
        "kopiert; nur dieser Ordner ist versioniert.",
        "",
        "```bash",
        "cp data/plots/<name>.png data/plots/saved/",
        "git add data/plots/saved && git commit -m \"Meilenstein: ...\"",
        "```",
        "",
        "Kopien in `saved/` werden nie ueberschrieben und koennen daher aus",
        "einer aelteren Codeversion stammen -- im Zweifel gegen die Dateien",
        "oben pruefen und gegen den Commit, der sie hinzugefuegt hat",
        "(`git log -- data/plots/saved/<name>.png`).",
        "",
    ]
    if saved.is_dir():
        parts += [f"- `{p.name}`" for p in sorted(saved.glob("*"))] + [""]
    return "\n".join(parts)


def write_readmes() -> list[Path]:
    written = []
    for directory, text in ((config.RESULTS_DIR, _results_readme()),
                            (config.PLOTS_DIR, _plots_readme())):
        if not directory.is_dir():
            continue
        path = directory / "README.md"
        path.write_text(text)
        written.append(path)
    return written


if __name__ == "__main__":
    for p in write_readmes():
        print("  ->", p)
