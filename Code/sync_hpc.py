"""CLI: Ergebnisse vom Rechenzentrum holen und nach `data/results/` einpflegen.

Ablauf in einem Aufruf: die HPC-CSVs per `rsync` nach `data/hpc/` holen, dann
jede `data/hpc/<graph>__estimates.csv` mit der lokalen
`data/results/<graph>__estimates.csv` abgleichen und nur die *fehlenden* Zeilen
anhaengen. Bestehende lokale Zeilen werden nie geloescht oder ueberschrieben.

Der rsync-Schritt laeuft standardmaessig mit (`--no-fetch` schaltet ihn aus,
`--fetch-only` bricht danach ab). Host und Remote-Pfad kommen aus den
Konstanten unten bzw. den Umgebungsvariablen `SE_HPC_HOST` / `SE_HPC_REMOTE`
oder den Flags `--host` / `--remote-dir`. Passwortloser SSH-Zugang (Key-Auth)
muss eingerichtet sein.

Zwei Zeilen beschreiben denselben Lauf, wenn sie in

    graph, view, estimator, seed, start_node, budget_rel, run

uebereinstimmen. Dann gilt:

    * Zeile lokal noch nicht vorhanden        -> wird angehaengt (NEW)
    * vorhanden, gleiches `estimate`          -> uebersprungen (DUPLICATE)
    * vorhanden, anderes `estimate`           -> FEHLER, Zeile bleibt liegen (CONFLICT)

Der Code-Fingerabdruck `code` gehoert bewusst *nicht* zum Schluessel: derselbe
Lauf aus einer anderen Codeversion mit gleichem Ergebnis zaehlt als vorhanden.
`code` wird nur in den Meldungen mit ausgegeben.

Eine HPC-Datei wird nach dem Abgleich geloescht -- aber nur, wenn sie *keinen*
Konflikt hatte. Vor dem Schreiben legt das Skript eine Kopie der lokalen Datei
unter `data/results/deprecated/hpc-sync_<zeit>/` ab (`--no-backup` schaltet das
aus).

Am Ende steht eine Zusammenfassung: uebertragene Zeilen je Estimator und View.

Beispiele:
    python sync_hpc.py                 # rsync-Fetch + Abgleich + saubere HPC-Dateien loeschen
    python sync_hpc.py --no-fetch      # nur abgleichen, was schon in data/hpc/ liegt
    python sync_hpc.py --fetch-only    # nur holen, nicht abgleichen
    python sync_hpc.py --dry-run       # rsync --dry-run + Abgleich-Vorschau
    python sync_hpc.py --keep          # abgleichen, HPC-Dateien aber behalten
    python sync_hpc.py --no-backup     # ohne Backup-Kopie
    python sync_hpc.py ../data/hpc/gpt4o_io__estimates.csv   # nur diese Datei (ohne Fetch)
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Damit `python Code/sync_hpc.py` auch aus dem Repo-Root laeuft, nicht nur aus Code/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config
from experiment import results as results_io

HPC_DIR = config.ROOT / "data" / "hpc"

# Woher `--fetch` (Default) die CSVs holt. Ueber Umgebungsvariablen oder die
# Flags --host / --remote-dir / --graphs ueberschreibbar.
HPC_HOST = os.environ.get("SE_HPC_HOST", "joko738d@dataport1.hpc.tu-dresden.de")
HPC_REMOTE_DIR = os.environ.get(
    "SE_HPC_REMOTE", "/data/horse/ws/joko738d-gptKE/sE/data/results")
HPC_FETCH_GRAPHS = ("gpt4_io", "gpt4o_io")

# Was einen Lauf eindeutig macht. `graph`, `seed` und `start_node` sind innerhalb
# einer Zieldatei konstant; damit ist dieser Schluessel deckungsgleich mit dem
# RUN_KEYS, das `results_io.append_results` intern zum Deduplizieren nutzt.
KEY = ("graph", "view", "estimator", "seed", "start_node", "budget_rel", "run")


def _key(row: pd.Series) -> tuple:
    """Schluessel-Tupel einer Zeile; NaN wird zu None vereinheitlicht."""
    return tuple(None if pd.isna(row[c]) else row[c] for c in KEY)


def _same_estimate(a, b, rel_tol: float = 1e-9) -> bool:
    a_na, b_na = pd.isna(a), pd.isna(b)
    if a_na and b_na:
        return True
    if a_na or b_na:
        return False
    try:
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=0.0)
    except (TypeError, ValueError):
        return str(a) == str(b)


def _fmt_key(key: tuple) -> str:
    return ", ".join(f"{c}={v!r}" for c, v in zip(KEY, key))


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if "nested" in df.columns:
        df = df.copy()
        df["nested"] = df["nested"].fillna(False).astype(bool)
    return df


def fetch(graphs, host: str, remote_dir: str, hpc_dir: Path,
          dry_run: bool) -> list[str]:
    """Die `<graph>__estimates.csv` per rsync nach `hpc_dir` holen.

    Rueckgabe: die Graphen, deren rsync fehlschlug (leer = alles ok). Ziel ist
    der absolute `hpc_dir`, daher unabhaengig vom Arbeitsverzeichnis.
    """
    hpc_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for g in graphs:
        src = f"{host}:{remote_dir}/{g}__estimates.csv"
        cmd = ["rsync", "-avz"] + (["--dry-run"] if dry_run else []) \
            + [src, f"{hpc_dir}/"]
        print("  $ " + " ".join(cmd), flush=True)
        try:
            rc = subprocess.run(cmd).returncode
        except FileNotFoundError:
            print("  FEHLER: 'rsync' nicht gefunden -- Fetch nicht moeglich")
            return list(graphs)
        if rc != 0:
            print(f"  FEHLER: rsync {g} endete mit Exit {rc}")
            failed.append(g)
    return failed


def process_file(hpc_path: Path, *, dry_run: bool, keep: bool,
                 backup_dir: Path | None) -> tuple[dict, int]:
    """Eine HPC-Datei abgleichen.

    Rueckgabe: (counts, n_conflicts). `counts` bildet (estimator, view) auf ein
    Dict mit den Schluesseln 'new', 'duplicate', 'conflict' ab.
    """
    graph, seed, start, kind = results_io.parse_stem(hpc_path.stem)
    counts: dict = defaultdict(lambda: {"new": 0, "duplicate": 0, "conflict": 0})

    if kind != "estimates":
        print(f"  uebersprungen (kind={kind!r}, nur 'estimates' wird abgeglichen)")
        return counts, 0

    hpc_df = _normalize(pd.read_csv(hpc_path, low_memory=False))
    missing = [c for c in KEY if c not in hpc_df.columns]
    if missing:
        print(f"  FEHLER: Spalten fehlen in {hpc_path.name}: {missing} -- uebersprungen")
        return counts, 1

    target = results_io._path(graph, "estimates", seed, start)
    loc_df = _normalize(results_io._read_csv(target))

    # lokale Zeilen nach Schluessel indizieren
    local: dict = {}
    if not loc_df.empty and all(c in loc_df.columns for c in KEY):
        for _, r in loc_df.iterrows():
            k = _key(r)
            if k in local:
                print(f"  WARNUNG: lokal mehrfach: {_fmt_key(k)} -- erste Zeile gilt")
                continue
            local[k] = (r.get("estimate"), r.get("code"))

    new_rows = []
    conflicts = 0
    for _, r in hpc_df.iterrows():
        k = _key(r)
        bucket = counts[(r["estimator"], r["view"])]
        if k not in local:
            new_rows.append(r)
            bucket["new"] += 1
            continue
        loc_est, loc_code = local[k]
        if _same_estimate(loc_est, r.get("estimate")):
            bucket["duplicate"] += 1
            continue
        conflicts += 1
        bucket["conflict"] += 1
        print(
            f"  ERROR: gleiche Parameter, anderes estimate -- nicht uebertragen\n"
            f"         {_fmt_key(k)}\n"
            f"         lokal:  estimate={loc_est!r}  code={loc_code!r}\n"
            f"         hpc:    estimate={r.get('estimate')!r}  code={r.get('code')!r}"
        )

    n_new = len(new_rows)
    if n_new and not dry_run:
        if backup_dir is not None and target.exists():
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup_dir / target.name)
        new_df = pd.DataFrame(new_rows).reindex(columns=hpc_df.columns)
        results_io.append_results(new_df, graph, "estimates", seed, start)

    action = "wuerde anhaengen" if dry_run else "angehaengt"
    print(f"  {n_new} {action} -> {target.name}"
          + (f"  ({conflicts} Konflikt(e), Datei bleibt liegen)" if conflicts else ""))

    if conflicts == 0 and not dry_run and not keep:
        hpc_path.unlink()
        print(f"  {hpc_path.name} geloescht")

    return counts, conflicts


def _print_summary(title: str, counts: dict) -> None:
    if not counts:
        return
    rows = sorted(counts.items())
    w_est = max(len(e) for (e, _v) in counts) + 2
    w_view = max(len(v) for (_e, v) in counts) + 2
    print(f"\n{title}")
    print(f"  {'estimator':<{w_est}}{'view':<{w_view}}{'new':>6}{'dup':>6}{'confl':>7}")
    tot = {"new": 0, "duplicate": 0, "conflict": 0}
    for (est, view), c in rows:
        print(f"  {est:<{w_est}}{view:<{w_view}}{c['new']:>6}{c['duplicate']:>6}"
              f"{c['conflict']:>7}")
        for k in tot:
            tot[k] += c[k]
    print(f"  {'GESAMT':<{w_est + w_view}}{tot['new']:>6}{tot['duplicate']:>6}"
          f"{tot['conflict']:>7}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", type=Path,
                   help="bestimmte HPC-CSVs (Default: alle in data/hpc/); "
                        "schaltet den Fetch aus")
    p.add_argument("--hpc-dir", type=Path, default=HPC_DIR,
                   help=f"Verzeichnis mit den HPC-CSVs (Default: {HPC_DIR})")
    p.add_argument("--no-fetch", dest="fetch", action="store_false",
                   help="rsync ueberspringen, nur abgleichen was schon da ist")
    p.add_argument("--fetch-only", action="store_true",
                   help="nur per rsync holen, danach ohne Abgleich beenden")
    p.add_argument("--graphs", nargs="+", default=list(HPC_FETCH_GRAPHS),
                   metavar="GRAPH", help="zu holende Graphen "
                   f"(Default: {' '.join(HPC_FETCH_GRAPHS)})")
    p.add_argument("--host", default=HPC_HOST,
                   help=f"SSH-Ziel fuer rsync (Default: {HPC_HOST})")
    p.add_argument("--remote-dir", default=HPC_REMOTE_DIR,
                   help=f"Remote-Ergebnisverzeichnis (Default: {HPC_REMOTE_DIR})")
    p.add_argument("--dry-run", action="store_true",
                   help="rsync --dry-run und Abgleich nur zeigen, nichts aendern")
    p.add_argument("--keep", action="store_true",
                   help="HPC-Dateien nach dem Abgleich nicht loeschen")
    p.add_argument("--no-backup", dest="backup", action="store_false",
                   help="keine Backup-Kopie der lokalen Datei anlegen")
    args = p.parse_args()

    fetch_failed: list[str] = []
    if args.fetch_only or (args.fetch and not args.files):
        print("Fetch vom Rechenzentrum:", flush=True)
        fetch_failed = fetch(args.graphs, args.host, args.remote_dir,
                             args.hpc_dir, args.dry_run)
    if args.fetch_only:
        if fetch_failed:
            print(f"\nrsync fehlgeschlagen fuer: {', '.join(fetch_failed)}")
            return 1
        print("\nFetch fertig.")
        return 0

    if args.files:
        paths = sorted(args.files)
    else:
        paths = sorted(args.hpc_dir.glob("*__estimates.csv"))

    if not paths:
        print(f"Nichts zu tun: keine __estimates.csv in {args.hpc_dir}")
        return 1 if fetch_failed else 0

    backup_dir = None
    if args.backup and not args.dry_run:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        backup_dir = config.RESULTS_DIR / "deprecated" / f"hpc-sync_{stamp}"

    total: dict = defaultdict(lambda: {"new": 0, "duplicate": 0, "conflict": 0})
    total_conflicts = 0
    for path in paths:
        print(f"\n{path}")
        if not path.exists():
            print("  FEHLER: Datei nicht gefunden -- uebersprungen")
            total_conflicts += 1
            continue
        counts, conflicts = process_file(
            path, dry_run=args.dry_run, keep=args.keep, backup_dir=backup_dir)
        _print_summary(f"{path.name} -- je Estimator x View", counts)
        for key, c in counts.items():
            for k in c:
                total[key][k] += c[k]
        total_conflicts += conflicts

    _print_summary("ZUSAMMENFASSUNG (alle Dateien) -- je Estimator x View", total)
    if backup_dir is not None and backup_dir.exists():
        print(f"\nBackups: {backup_dir}")
    if fetch_failed:
        print(f"\nrsync fehlgeschlagen fuer: {', '.join(fetch_failed)}")
    if total_conflicts:
        print(f"\n{total_conflicts} Konflikt(e) -- betroffene HPC-Dateien blieben liegen.")
    if total_conflicts or fetch_failed:
        return 1
    print("\nFertig.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
