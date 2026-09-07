"""Survival-Kurve: wie viele Random Walks einem Sink entgehen.

x = Anteil des Budgets, der verbraucht wurde (0..100 %).
y = Zahl der Laeufe, die zu diesem Zeitpunkt noch neue Knoten finden.

Jeder Lauf steuert genau einen Wert bei: `sunk_frac` (Budget-Anteil im Moment
des Versinkens) bzw. NaN, wenn er das volle Budget ueberlebt hat. Die Kurve ist
`n_live(x) = n_runs - #{sunk_frac <= x}` -- eine monoton fallende
Treppenfunktion. Ueberlebende zaehlen bis x = 1 als live (rechts-zensiert), die
gestrichelte Linie markiert ihre Zahl.

Eine Linie je View.

Schnittstelle:
    plot_survival(df, path=None) -> Path
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

import config  # noqa: E402
from plotting.compare import VIEW_TITLES  # noqa: E402
from plotting.style import INK, INK_MUTED, SURFACE, apply_axes_style, color_for  # noqa: E402


def plot_survival(df, path: Path | None = None) -> Path:
    graph = df["graph"].iloc[0]
    seed = int(df["seed"].iloc[0])
    dead_end = "+".join(sorted(df["dead_end"].unique()))
    z = int(df["z"].iloc[0])
    budget_rel = float(df["budget_rel"].iloc[0])

    views = [v for v in VIEW_TITLES if (df["view"] == v).any()]
    views += [v for v in df["view"].unique() if v not in views]

    # x = verbrauchtes Budget als Anteil von |V| (0..budget_rel), nicht als
    # Anteil des Budgets selbst. sunk_frac steht in der CSV bezogen auf das
    # Budget -- hier auf |V| umgerechnet.
    x = np.linspace(0.0, budget_rel, 501)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(SURFACE)

    for i, view in enumerate(views):
        sub = df[df["view"] == view]
        n_runs = len(sub)
        fracs = np.sort(sub["sunk_frac"].dropna().to_numpy()) * budget_rel
        n_live = n_runs - np.searchsorted(fracs, x, side="right")
        color = color_for(i)
        ax.step(x, n_live, where="post", lw=2, color=color,
                label=f"{VIEW_TITLES.get(view, view)}  (n={n_runs})")
        survivors = n_runs - len(fracs)
        ax.axhline(survivors, color=color, lw=1, linestyle=(0, (4, 3)), alpha=0.5)

    apply_axes_style(ax)
    ax.set_xlim(0.0, budget_rel)
    ax.set_ylim(0, max(len(df[df["view"] == v]) for v in views))
    ax.set_xticks(np.linspace(0.0, budget_rel, 5))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Budget consumed (fraction of |V|)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("Random walks still live", color=INK_MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="lower left")

    fig.suptitle(f"{config.graph_label(graph)}: how many random walks avoid a sink",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(0.99, 0.985, f"seed {seed}  |  dead_end={dead_end}  |  z={z}",
             color=INK_MUTED, fontsize=9, ha="right", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    from experiment.results import seed_tag
    if path is None:
        path = config.unique_path(
            config.PLOTS_DIR / f"{graph}__{seed_tag(seed)}walk_survival.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
