"""Gemeinsame Plot-Parameter: validierte Kategorienfarben und Achsen-Stil.

Die Farbreihenfolge ist fix (Slot 1..8) und wird nie zyklisch wiederverwendet;
mehr als 8 Estimators in einem Panel -> lieber auf mehrere Figuren aufteilen.

Schnittstelle:
    PALETTE, SURFACE, INK, INK_MUTED, GRID
    color_for(index) -> str
    apply_axes_style(ax)
"""

from __future__ import annotations

# Kategoriale Palette (Light-Mode), feste Slot-Reihenfolge.
PALETTE = [
    "#2a78d6",  # blau
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # gelb
    "#e87ba4",  # magenta
    "#008300",  # gruen
    "#4a3aa7",  # violett
    "#e34948",  # rot
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"


def color_for(index: int) -> str:
    if index >= len(PALETTE):
        raise ValueError(
            f"Nur {len(PALETTE)} Farbslots -- {index + 1} Estimators in einem Panel sind zu "
            "viele, um sie noch unterscheiden zu koennen. Farben werden bewusst nicht "
            "wiederholt: waehle eine Teilmenge mit "
            "plot_results.py --estimators <name> ... (oder --match <substring>)."
        )
    return PALETTE[index]


def apply_axes_style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
