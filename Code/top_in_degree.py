"""CLI: alle Knoten eines Graphen, nach Eingangsgrad geordnet, als Textliste.

Der Eingangsgrad ist auf den gerichteten Testgraphen die Groesse, die den
Walk-Verkehr auf einen Knoten zieht (siehe die `literature`-Analyse: ein
In-Hub mit deg_out = 1 leitet einen ganzen Trichter weiter).

Berechnung: `np.bincount(graph.indices, minlength=graph.n_nodes)` zaehlt je
Knoten-ID, wie oft sie als Kantenziel vorkommt -- dieselbe Formel wie in
diagnose_walk._degrees. Ausgabe: eine Datei je Graph unter additionals/, eine
Zeile `<eingangsgrad>\\t<name>` je Knoten, absteigend nach Eingangsgrad; bei
Gleichstand nach Knoten-ID. Vollstaendig, sofern nicht --top gesetzt ist.

Beispiele:
    python top_in_degree.py --graph gpt4_io gpt4o_io
    python top_in_degree.py --graph gpt4_io --top 5000
    python top_in_degree.py --graph Slashdot0811 --names-only --out-dir /tmp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import config
from graphs import loader


def write_in_degree(graph_name: str, out_dir: Path, top: int | None = None,
                    names_only: bool = False) -> Path:
    """Schreibt <out_dir>/<kanonischer-name>__in-degree.txt und gibt den Pfad
    zurueck. `top=N` kappt auf die N Knoten mit hoechstem Eingangsgrad."""
    loader.clear_cache()                                    # nur ein Graph im RAM
    g = loader.load_graph(graph_name)                       # Kuerzel -> kanonischer Name
    in_deg = np.bincount(g.indices, minlength=g.n_nodes)    # Eingangsgrad je Knoten-ID
    order = np.argsort(in_deg, kind="stable")[::-1]         # absteigend, Ties nach ID
    if top is not None:
        order = order[:top]

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{g.name}__in-degree.txt"
    names = g.names                                         # ID -> Name, O(1)-Zugriff
    with path.open("w", encoding="utf-8") as f:
        if names_only:
            f.writelines(f"{names[i]}\n" for i in order)
        else:
            f.writelines(f"{int(in_deg[i])}\t{names[i]}\n" for i in order)

    print(f"{g.name}: |V|={g.n_nodes:,}  {len(order):,} Zeilen  ->  {path}"
          .replace(",", " "))
    for i in order[:10]:
        print(f"  {int(in_deg[i]):>9,}  {names[i]}".replace(",", " "))
    return path


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", "--graphs", dest="graphs", nargs="+", required=True,
                   metavar="GRAPH", help="ein oder mehrere Namen/Kuerzel")
    p.add_argument("--top", type=int, default=None,
                   help="nur die N Knoten mit hoechstem Eingangsgrad (Default: alle)")
    p.add_argument("--names-only", action="store_true",
                   help="nur Namen, ohne die Eingangsgrad-Spalte")
    p.add_argument("--out-dir", type=Path, default=config.ROOT / "additionals",
                   help="Zielverzeichnis (Default: additionals/)")
    args = p.parse_args()

    for name in args.graphs:
        write_in_degree(name, args.out_dir, args.top, args.names_only)
        print()


if __name__ == "__main__":
    main()
