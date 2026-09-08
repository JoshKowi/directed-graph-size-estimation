"""CLI: die Knoten mit dem hoechsten Eingangsgrad je Graph als Textliste.

Der Eingangsgrad ist auf den gerichteten Testgraphen die Groesse, die den
Walk-Verkehr auf einen Knoten zieht (siehe die `literature`-Analyse: ein
In-Hub mit deg_out = 1 leitet einen ganzen Trichter weiter). Diese Liste macht
die grossen In-Hubs sichtbar -- z. B. als Startpunkte oder als Filter.

Berechnung: `np.bincount(graph.indices, minlength=graph.n_nodes)` zaehlt je
Knoten-ID, wie oft sie als Kantenziel vorkommt -- dieselbe Formel wie in
diagnose_walk._degrees. Ausgabe: eine Datei je Graph unter additionals/,
ein Knotenname pro Zeile (absteigend nach Eingangsgrad).

Beispiele:
    python top_in_degree.py --graph gpt4_io gpt4o_io
    python top_in_degree.py --graph gpt4_io --top 5000 --with-degree
    python top_in_degree.py --graph Slashdot0811 --top 50 --out-dir /tmp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import config
from graphs import loader


def top_in_degree(graph_name: str, top: int, out_dir: Path,
                  with_degree: bool = False) -> Path:
    """Schreibt die `top` Knoten mit hoechstem Eingangsgrad nach
    <out_dir>/<kanonischer-name>__top-in-degree.txt und gibt den Pfad zurueck."""
    loader.clear_cache()                                    # nur ein Graph im RAM
    g = loader.load_graph(graph_name)                       # Kuerzel -> kanonischer Name
    in_deg = np.bincount(g.indices, minlength=g.n_nodes)    # Eingangsgrad je Knoten-ID
    order = np.argsort(in_deg, kind="stable")[::-1][:top]   # absteigend, Ties nach ID

    lines = [f"{int(in_deg[i])}\t{g.name_of(int(i))}" if with_degree
             else str(g.name_of(int(i))) for i in order]

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{g.name}__top-in-degree.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{g.name}: |V|={g.n_nodes:,}  {len(lines)} Zeilen  ->  {path}"
          .replace(",", " "))
    for i in order[:10]:
        print(f"  {int(in_deg[i]):>9,}  {g.name_of(int(i))}".replace(",", " "))
    return path


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", "--graphs", dest="graphs", nargs="+", required=True,
                   metavar="GRAPH", help="ein oder mehrere Namen/Kuerzel")
    p.add_argument("--top", type=int, default=1000,
                   help="Anzahl der Top-Knoten je Graph (Default: 1000)")
    p.add_argument("--out-dir", type=Path, default=config.ROOT / "additionals",
                   help="Zielverzeichnis (Default: additionals/)")
    p.add_argument("--with-degree", action="store_true",
                   help="jede Zeile '<eingangsgrad>\\t<name>' statt nur Name")
    args = p.parse_args()

    for name in args.graphs:
        top_in_degree(name, args.top, args.out_dir, args.with_degree)
        print()


if __name__ == "__main__":
    main()
