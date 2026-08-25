"""Vergleichs-Plot: frei gewaehlte Estimators, frei gewaehlte Kantensichten.

Anders als plotting.ranges (festes Raster Kategorie x View) stellt diese
Funktion genau die Reihen nebeneinander, die man vergleichen will -- eine
Spalte je View, eine Farbe je Estimator.

Gezeigt wird pro Estimator und Budget die Spanne min..max ueber die n Laeufe
plus der Median, je Estimator leicht gegeneinander versetzt (sonst verdecken
sich Verfahren mit gleichem Wert). Die y-Achse ist Schaetzung/|V| (log), die gestrichelte Linie
bei 1.0 ist die wahre Groesse.

Schnittstelle:
    plot_comparison(summary, graph_name, estimators, views, title, path,
                    colors=None) -> matplotlib.figure.Figure
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from plotting.ranges import VIEW_TITLES, budget_ticks  # noqa: E402
from plotting.style import INK, INK_MUTED, SURFACE, apply_axes_style, color_for  # noqa: E402


def plot_comparison(
    summary,
    graph_name: str,
    estimators: list[str],
    views: list[str],
    title: str,
    path: Path | None = None,
    colors: dict[str, str] | None = None,
):
    """summary: DataFrame aus experiment.results.summarize()."""
    summary = summary[(summary["graph"] == graph_name)
                      & (summary["estimator"].isin(estimators))
                      & (summary["view"].isin(views))]
    if summary.empty:
        raise ValueError(f"Keine Zeilen fuer {graph_name} / {estimators} / {views}")

    budgets = sorted(summary["budget_rel"].unique())
    x = np.arange(len(budgets))
    if colors is None:
        colors = {e: color_for(i) for i, e in enumerate(estimators)}
    # Zwei Estimators in derselben Farbe waeren im Bild nicht auseinander zu
    # halten -- lieber hier scheitern als eine unlesbare Grafik ausliefern.
    used = [colors[e] for e in estimators]
    if len(set(used)) != len(used):
        dupes = sorted({c for c in used if used.count(c) > 1})
        raise ValueError(
            f"Farbkollision in {title!r}: {dupes} doppelt vergeben. "
            f"Estimators: {estimators}"
        )

    fig, axes = plt.subplots(1, len(views), figsize=(6.6 * len(views), 4.6),
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for c, view in enumerate(views):
        ax = axes[0][c]
        panel = summary[summary["view"] == view]

        # Kleiner horizontaler Versatz je Estimator: liegen zwei Verfahren auf
        # demselben Wert -- was durchaus vorkommt -- wuerden sie sich sonst
        # gegenseitig vollstaendig verdecken.
        present = [e for e in estimators if not panel[panel["estimator"] == e].empty]
        span = 0.10 * (len(present) - 1)
        for i, est in enumerate(present):            # feste Reihenfolge -> feste Farbe
            rows = (panel[panel["estimator"] == est]
                    .set_index("budget_rel").reindex(budgets))
            rel = lambda col: rows[col] / rows["true_size"]  # noqa: E731
            xi = x + (0.10 * i - span / 2)
            ax.vlines(xi, rel("est_min"), rel("est_max"),
                      color=colors[est], linewidth=2, alpha=0.75, zorder=3)
            ax.plot(xi, rel("est_median"), "o", markersize=8, color=colors[est],
                    markeredgecolor=SURFACE, markeredgewidth=2, label=est, zorder=4)

        ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.set_yscale("log")
        ax.set_xticks(x, budget_ticks(panel, budgets))
        apply_axes_style(ax)
        ax.set_xlabel("Budget (fraction of |V| / allowed queries)",
                      color=INK_MUTED, fontsize=9)
        if len(views) > 1:
            ax.set_title(VIEW_TITLES[view], color=INK, fontsize=10, loc="left", pad=8)
        if c == 0:
            ax.set_ylabel("Estimate / true size", color=INK_MUTED, fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK_MUTED,
               ncol=min(len(labels), 3), loc="upper left",
               bbox_to_anchor=(0.01, 0.93), handletextpad=0.4, columnspacing=1.4)

    wrapped = textwrap.fill(title, width=52 * len(views))
    n_legend_rows = -(-len(labels) // min(len(labels), 3))
    top = 0.90 - 0.035 * n_legend_rows - 0.03 * wrapped.count("\n")
    fig.suptitle(wrapped, color=INK, fontsize=11, x=0.01, ha="left", va="top", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, top))

    if path is None:
        config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.PLOTS_DIR / f"{graph_name}__comparison.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return fig
