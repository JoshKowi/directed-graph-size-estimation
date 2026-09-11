"""Diagnose-Plot zu nmmc_trace.py: was NMMC innen tut.

Drei Panels, je eine Gruppe pro Variante (indeg x target), Balken = Median
ueber die Laeufe, Whisker = Spanne min..max:

    Annahmequote     -- wie oft ein vorgeschlagener Zug angenommen wird. Der
                        Rest sind Umverteilungen in bereits bekanntes Gebiet.
    d- = 1           -- Anteil der Vorschlaege, bei denen der Eingangsgrad im
                        Nenner von b noch 1 war. Nahe 1 heisst: der Nenner
                        traegt keine Information, b = d+(i), die Kette ist
                        P = A/c und ihre QSD die Eigenvektorzentralitaet
                        (Korollar 3.3) -- nicht mehr die Zielverteilung.
    c_t / c          -- wie weit die gelernte Normierung an das wahre
                        c = max b_ij herangekommen ist. Solange sie darunter
                        liegt, greift die Kappung gamma = 1 und die QSD ist
                        noch gar nicht die Zielverteilung. Leer bei `online`
                        und `cross-online`: dort haengt der Nenner am Lauf,
                        ein festes c gibt es nicht (s. nmmc_trace.true_c).

Eine Farbe je View.

Schnittstelle:
    plot_nmmc_trace(df, path=None) -> Path
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from plotting.compare import VIEW_TITLES  # noqa: E402
from plotting.style import INK, INK_MUTED, SURFACE, apply_axes_style, color_for  # noqa: E402

PANELS = (
    ("acc_rate", "Annahmequote", "angenommen / vorgeschlagen"),
    ("frac_dinhat_1", "Anteil d- = 1", "Vorschlaege ohne d--Information"),
    ("c_ratio", "c_t / c", "gelernte vs. wahre Normierung"),
)


def plot_nmmc_trace(df, path: Path | None = None) -> Path:
    graph = df["graph"].iloc[0]
    seed = int(df["seed"].iloc[0])
    budget_rel = float(df["budget_rel"].iloc[0])
    alpha = float(df["alpha"].iloc[0])
    views = [v for v in VIEW_TITLES if v in set(df["view"])]
    # Die Agentenzahl gehoert in die Achse, sobald mehr als eine vorkommt --
    # sonst legte der Plot Laeufe mit 1 und mit 1000 Agenten uebereinander.
    agents = sorted(set(df["agents"])) if "agents" in df else [1]
    variants = sorted({(r.indeg, r.target, getattr(r, "agents", 1))
                       for r in df.itertuples()})
    labels = [f"{i}\n{t}" + (f"\nK={k}" if len(agents) > 1 else "")
              for i, t, k in variants]

    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.4 * len(PANELS), 4.2))
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(variants), dtype=float)
    width = 0.8 / max(len(views), 1)

    for ax, (col, title, subtitle) in zip(axes, PANELS):
        apply_axes_style(ax)
        for vi, view in enumerate(views):
            med, lo, hi = [], [], []
            for indeg, target, k in variants:
                rows = df[(df["view"] == view) & (df["indeg"] == indeg)
                          & (df["target"] == target)]
                if "agents" in df:
                    rows = rows[rows["agents"] == k]
                sel = rows[col].to_numpy(dtype=float)
                med.append(np.median(sel) if sel.size else np.nan)
                lo.append(sel.min() if sel.size else np.nan)
                hi.append(sel.max() if sel.size else np.nan)
            med, lo, hi = np.array(med), np.array(lo), np.array(hi)
            pos = x + (vi - (len(views) - 1) / 2) * width
            ax.bar(pos, med, width * 0.9, color=color_for(vi), zorder=3,
                   label=VIEW_TITLES.get(view, view))
            # Spanne ueber die Laeufe: ohne sie liest sich ein Median als
            # Konstante, obwohl die Laeufe weit auseinanderliegen koennen.
            ax.errorbar(pos, med, yerr=[med - lo, hi - med], fmt="none",
                        ecolor=INK_MUTED, elinewidth=1, capsize=3, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, color=INK_MUTED)
        ax.set_ylim(0, max(1.0, float(np.nanmax(df[col].to_numpy(dtype=float))) * 1.1))
        ax.set_title(title, color=INK, fontsize=11, pad=18)
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8, color=INK_MUTED)

    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_MUTED)
    fig.suptitle(f"NMMC-Diagnose -- {config.graph_label(graph)}",
                 color=INK, fontsize=13, y=0.99)
    shared = bool(df["shared_c"].iloc[0]) if "shared_c" in df else False
    fig.text(0.99, 0.985,
             f"Budget {budget_rel:g}, alpha {alpha:g}, Seed {seed}"
             + (", c_t geteilt" if shared else ""),
             ha="right", va="top", fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if path is None:
        tag = f"__seed{seed}" if seed != config.DEFAULT_SEED else ""
        path = config.unique_path(
            config.PLOTS_DIR / f"{graph}{tag}__nmmc-trace.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
