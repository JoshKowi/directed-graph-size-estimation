"""Range-Plot: Streuung der Schaetzungen je Estimator, Budget und Kantensicht.

Raster aus Small Multiples: eine Spalte je Kategorie (Vergleich / real
umsetzbar), eine Zeile je View (directed / undirected / ...). Farbe steht
durchgehend fuer den Estimator -- der Unterschied zwischen den Views ist damit
ein senkrechter Vergleich an derselben x-Position.

Gezeigt wird pro Estimator und Budget die Spanne min..max ueber die n Laeufe
plus der Median, jeweils exakt auf der Budget-Position; die gestrichelte Linie bei 1.0 ist die wahre Groesse. Die
y-Achse ist das Verhaeltnis Schaetzung/|V| (log), damit Ueber- und
Unterschaetzung symmetrisch lesbar sind.

Schnittstelle:
    budget_ticks(panel, budgets) -> list[str]
    plot_ranges(summary, graph_name=None, path=None, note=None)
        -> matplotlib.figure.Figure
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from estimators.base import Category  # noqa: E402
from plotting.style import INK, INK_MUTED, SURFACE, apply_axes_style, color_for  # noqa: E402


def _format_budget(b: float) -> str:
    return f"{b * 100:.3f}".rstrip("0").rstrip(".") + " %"


def budget_ticks(panel, budgets: list[float]) -> list[str]:
    """Zweizeilige x-Beschriftung: relatives und absolutes erlaubtes Budget.

    Eine dritte Zeile mit dem tatsaechlich ausgegebenen Budget erscheint nur,
    wenn es merklich abweicht. Mit config.COST_CACHE_HIT > 0 schoepft jeder
    Lauf sein Budget aus, die Zeile sollte also nie auftauchen -- sie ist die
    Kontrolle, dass diese Annahme haelt.

    panel: die Summary-Zeilen des Panels (ein View, die gezeigten Estimators).
    """
    labels = []
    for b in budgets:
        rows = panel[panel["budget_rel"] == b]
        if rows.empty:
            labels.append(_format_budget(b))
            continue
        allowed = float(rows["budget_abs"].iloc[0])
        label = f"{_format_budget(b)}\n{allowed:,.0f}".replace(",", " ")
        used_lo, used_hi = rows["used_median"].min(), rows["used_median"].max()
        if used_lo < 0.98 * allowed:          # sonst redundant
            fmt = lambda v: f"{v:,.0f}".replace(",", " ")  # noqa: E731
            span = fmt(used_lo) if used_hi <= 1.02 * used_lo else f"{fmt(used_lo)}-{fmt(used_hi)}"
            label += f"\nused {span}"
        labels.append(label)
    return labels


# Beschriftungen im Bild sind durchgehend englisch (die Grafiken gehen in
# Praesentationen); Kommentare und Docstrings bleiben deutsch.
CATEGORY_TITLES = {
    Category.COMPARISON: "Reference only (access not realizable)",
    Category.REALIZABLE: "Realizable",
}

VIEW_TITLES = {
    "directed": "directed (original)",
    "undirected": "undirected (symmetrized)",
    "reverse": "reverse (in-edges only)",
}


def plot_ranges(summary, graph_name: str | None = None, path: Path | None = None,
                note: str | None = None):
    """summary: DataFrame aus experiment.results.summarize()."""
    if graph_name is not None:
        summary = summary[summary["graph"] == graph_name]
    graph_name = graph_name or str(summary["graph"].iloc[0])

    budgets = sorted(summary["budget_rel"].unique())
    x = np.arange(len(budgets))
    categories = [c for c in CATEGORY_TITLES if (summary["category"] == str(c)).any()]
    views = [v for v in VIEW_TITLES if (summary["view"] == v).any()]

    # Farbzuordnung einmal global: derselbe Estimator hat in jedem Panel
    # dieselbe Farbe, auch wenn er nicht ueberall vorkommt.
    colors = {est: color_for(i) for i, est in enumerate(sorted(summary["estimator"].unique()))}

    fig, axes = plt.subplots(
        len(views),
        len(categories),
        figsize=(6.2 * len(categories), 3.9 * len(views) + 0.6),
        sharey=True,
        squeeze=False,
    )
    fig.patch.set_facecolor(SURFACE)

    for r, view in enumerate(views):
        for c, category in enumerate(categories):
            ax = axes[r][c]
            panel = summary[(summary["view"] == view) & (summary["category"] == str(category))]
            estimators = sorted(panel["estimator"].unique())

            for est in estimators:
                rows = panel[panel["estimator"] == est].set_index("budget_rel").reindex(budgets)
                rel = lambda col: rows[col] / rows["true_size"]  # noqa: E731
                color = colors[est]
                # Alle Estimators sitzen exakt auf der Budget-Position; die
                # Spannen sind leicht transparent, damit Ueberlappungen sichtbar
                # bleiben statt sich gegenseitig zu verdecken.
                ax.vlines(x, rel("est_min"), rel("est_max"),
                          color=color, linewidth=2, alpha=0.75, zorder=3)
                ax.plot(x, rel("est_median"), "o", markersize=8,
                        color=color, markeredgecolor=SURFACE, markeredgewidth=2,
                        label=est, zorder=4)

            ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
            ax.set_yscale("log")
            ax.set_xticks(x, budget_ticks(panel, budgets))
            apply_axes_style(ax)

            title = CATEGORY_TITLES[category] if r == 0 else None
            if title:
                ax.set_title(title, color=INK, fontsize=10, loc="left", pad=10)
            if r == len(views) - 1:
                ax.set_xlabel("Budget (fraction of |V| / allowed queries)",
                              color=INK_MUTED, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{VIEW_TITLES[view]}\nEstimate / true size",
                              color=INK_MUTED, fontsize=9)
            # Legende einmal je Spalte (oben): Farbe bedeutet in allen Zeilen
            # denselben Estimator, eine Wiederholung waere nur Rauschen.
            if r == 0:
                ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED,
                          ncol=len(estimators), loc="lower right", bbox_to_anchor=(1.0, 1.0),
                          handletextpad=0.4, columnspacing=1.4)

    fig.suptitle(f"{graph_name}: spread of size estimates by edge view",
                 color=INK, fontsize=12, x=0.01, ha="left")
    if note:      # Herkunft des Bildes, v.a. der Seed -- siehe plotting.compare
        fig.text(0.99, 0.985, note, color=INK_MUTED, fontsize=9, ha="right", va="top")
    fig.tight_layout()

    if path is None:
        config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.PLOTS_DIR / f"{graph_name}__ranges.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    return fig
