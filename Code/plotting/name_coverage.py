"""Coverage- und Surplus-Kurven externer Namenslisten.

Wie viele Eintraege einer nach Relevanz sortierten Liste lohnen sich? Zwei
Groessen wachsen mit der Praefixlaenge, aber gegenlaeufig im Nutzen:

    coverage   Anteil der Graph-Knoten, den die Liste erreicht. Bestimmt die
               *Gueltigkeit* einer Ziehung daraus -- mehr als diesen Teil kann
               sie nie sehen.
    surplus    Anteil der Eintraege ohne Entsprechung im Graphen. Bestimmt die
               *Kosten* -- jeder davon ist ein Fehlschlag beim Rejection
               Sampling (oracles.name_list, config.COST_DRAW_MISS).

Mehr Eintraege heben beides. Wo der Handel kippt, soll aus dem Bild ablesbar
sein; deshalb gibt es Stuetzlinien statt einer gesetzten Optimum-Marke. Welches
Optimum gilt, haengt am Kostenverhaeltnis und an der Abdeckung, die ein
Verfahren ueberhaupt braucht -- beides steht nicht im Diagramm.

Ein Bild je (Graph, Groesse): die Kurven eines Graphen gehoeren zusammen,
coverage und surplus nicht -- sie teilen keine y-Achse.

Schnittstelle:
    METRICS: dict[str, tuple[str, str]]
    plot_curve(curves, graph_name, metric, path=None) -> Path
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from plotting.style import (GRID, INK, INK_MUTED, SURFACE,  # noqa: E402
                            apply_axes_style, color_for)

#: Schluessel -> (Achsenbeschriftung, Erklaerung, y-Achse logarithmisch?)
#
# coverage laeuft ueber sechs Zehnerpotenzen -- von einem einzelnen Knoten
# (1/|V|, rund 1.5e-5 %) bis in den zweistelligen Prozentbereich. Linear
# aufgetragen waere davon nur das letzte Zehntel der Achse sichtbar. Deshalb
# dort log.
#
# surplus lebt dagegen von Anfang an im Prozentbereich und hat sein Minimum
# irgendwo zwischen 5 und 20 % -- die Frage ist, *wo* es liegt und wie schnell
# es danach steigt. Das liest sich linear besser: gleiche Abstaende sind
# gleiche Prozentpunkte, und der Abstand zwischen zwei Kurven ist direkt der
# Unterschied im Ausschuss. Ausserdem faellt der 0-%-Anfang der In-Grad-Listen
# nicht weg.
METRICS = {
    "coverage": ("coverage: share of |V| reached",
                 "distinct graph nodes hit by the first n entries, over |V|",
                 True),
    "surplus": ("surplus: share of entries without a match",
                "entries among the first n with no counterpart in the graph",
                False),
}

#: Senkrechte Stuetzlinien -- die Dekaden, an denen abgelesen wird.
VGUIDES = (1e4, 1e5, 1e6)


def _decade_limits(vals: np.ndarray) -> tuple[float, float]:
    """Umschliessende Zehnerpotenzen zu den positiven Werten."""
    pos = vals[vals > 0]
    if pos.size == 0:
        return 1e-3, 100.0
    lo = 10.0 ** np.floor(np.log10(pos.min()))
    hi = min(100.0, 10.0 ** np.ceil(np.log10(pos.max())))
    return lo, hi


def plot_curve(curves: dict, graph_name: str, metric: str,
               path: Path | None = None, logy: bool | None = None) -> Path:
    """Ein Diagramm. `curves` ist {Beschriftung: Rueckgabe von coverage_curve}.

    `logy=None` nimmt den Default der Groesse (s. METRICS): log fuer coverage,
    linear fuer surplus. Begruendung fuer coverage: beide Groessen laufen
    laufen: coverage beginnt bei einem einzelnen Knoten (1/|V|, also rund
    1.5e-5 %) und endet im zweistelligen Prozentbereich. Linear aufgetragen
    waere davon nur das letzte Zehntel der Achse sichtbar, und der Verlauf
    saehe zwangslaeufig exponentiell aus -- coverage waechst naeherungsweise
    linear in n, und linear-in-n ist auf einer log-x-Achse eine Exponentiale.
    Doppelt logarithmisch wird daraus eine Gerade, und *Abweichungen* davon
    (also das Abflachen) werden ueberhaupt erst sichtbar.

    Nullwerte fallen auf der Log-Achse weg. Bei surplus ist das der Anfang der
    In-Grad-Listen: deren erste Eintraege treffen alle, surplus ist dort exakt
    0 und damit nicht darstellbar. Die Kurve beginnt entsprechend spaeter.
    """
    if metric not in METRICS:
        raise ValueError(f"metric {metric!r} -- moeglich: {sorted(METRICS)}")
    ylabel, explain, default_logy = METRICS[metric]
    if logy is None:
        logy = default_logy

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    fig.patch.set_facecolor(SURFACE)
    apply_axes_style(ax)
    # apply_axes_style setzt nur ein y-Gitter; hier gehoert auch eins in x dazu,
    # weil auf der Log-Achse abgelesen wird.
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)

    allv = []
    dropped = []
    for i, (label, c) in enumerate(curves.items()):
        x = np.asarray(c["n"], dtype=float)
        y = np.asarray(c[metric], dtype=float) * 100.0
        allv.append(y)
        if logy:
            keep = y > 0
            if not keep.all():
                dropped.append(f"{label} from n={int(x[keep][0]):,}".replace(",", " ")
                               if keep.any() else f"{label}: all zero")
            x, y = x[keep], y[keep]
        ax.plot(x, y, color=color_for(i), linewidth=2, zorder=3, label=label)

    for x in VGUIDES:
        ax.axvline(x, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 3)),
                   alpha=0.45, zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(1, 1e6)
    if logy:
        lo, hi = _decade_limits(np.concatenate(allv))
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        for y in 10.0 ** np.arange(np.log10(lo), np.log10(hi) + 1):
            ax.axhline(y, color=INK_MUTED, linewidth=0.6, linestyle=(0, (4, 3)),
                       alpha=0.30, zorder=1)
    else:
        vmax = float(np.concatenate(allv).max())
        for step in (1, 2, 5, 10, 20, 25, 50):
            if vmax / step <= 8:
                ticks = np.arange(0, vmax + step, step)
                break
        else:
            ticks = np.linspace(0, vmax, 6)
        ax.set_yticks(ticks)
        ax.set_ylim(0, ticks[-1])
        for y in ticks[1:]:
            ax.axhline(y, color=INK_MUTED, linewidth=0.6, linestyle=(0, (4, 3)),
                       alpha=0.30, zorder=1)

    ax.set_xlabel("entities taken from the list (log, in file order)",
                  color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel + "  [%, log]" if logy else ylabel + "  [%]",
                  color=INK_MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED,
              loc="upper left" if metric == "coverage" else "lower right")

    n_nodes = next(iter(curves.values()))["n_nodes"]
    fig.suptitle(f"{config.graph_label(graph_name)}: {metric} by list length",
                 color=INK, fontsize=12, x=0.005, ha="left")
    ax.set_title(f"{explain}   |   |V| = {n_nodes:,}".replace(",", " "),
                 color=INK_MUTED, fontsize=9, loc="left", pad=8)
    if dropped:
        # Als Fussnote unter die Achse statt in den Untertitel: dort wuerde die
        # Zeile bei zwei Kurven ueber den Rand hinauslaufen.
        fig.text(0.005, 0.012,
                 "log axis drops 0 %: curve starts at the first hit -- "
                 + "; ".join(dropped),
                 color=INK_MUTED, fontsize=7.5, ha="left")
    fig.tight_layout(rect=(0, 0.03 if dropped else 0, 1, 0.96))

    if path is None:
        config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.unique_path(
            config.PLOTS_DIR / f"{graph_name}__namelist-{metric}.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
