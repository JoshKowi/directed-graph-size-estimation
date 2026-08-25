"""Vier-Panel-Diagnose eines Random Walks: Sackgassen oder falsche Gewichte?

Je View eine Zeile, vier Spalten:

    1  Leiter    |V| -> erreichbar -> besucht -> effektiver Traeger ->
                 Schaetzung ohne/mit Gewichten. Man sieht auf einen Blick, an
                 welcher Sprosse die Schaetzung abreisst.
    2  Abdeckung verschiedene Knoten ueber die Schritte (log-log). Ein Plateau
                 heisst: der Walk findet nichts Neues mehr -- Sackgasse.
    3  Grad      Besuche gegen Ausgangs- und Eingangsgrad (gebinnte Mediane).
                 Folgen die Besuche dem Eingangsgrad, passt 1/deg_out nicht.
    4  Entitaeten die meistbesuchten Knoten mit Namen und beiden Graden.

Schnittstelle:
    plot_diagnosis(results, path=None) -> Path
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from plotting.ranges import VIEW_TITLES  # noqa: E402
from plotting.style import GRID, INK, INK_MUTED, PALETTE, SURFACE, apply_axes_style  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = PALETTE[:6]


def _ladder(ax, d):
    n = d["n_nodes"]
    rows = [
        ("|V| (truth)", n, INK_MUTED),
        ("reachable from seed", d["reachable"], BLUE),
        ("reachable from end node", d["reachable_end"], MAGENTA),
        ("distinct visited", d["distinct"], AQUA),
        ("effective support", d["n_eff"], YELLOW),
        ("estimate, no weights", d["uis"], ORANGE),
        ("estimate, 1/deg_out", d["wis"], GREEN),
    ]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [max(v, 1e-9) for _, v, _ in rows],
            color=[c for _, _, c in rows], height=0.62)
    ax.set_xscale("log")
    ax.set_yticks(y, [lbl for lbl, _, _ in rows], fontsize=8)
    ax.set_xlim(max(min(v for _, v, _ in rows) * 0.3, 1e-1), n * 3)
    for yi, (_, v, _) in zip(y, rows):
        ax.text(v * 1.35, yi, f"{v:,.0f}".replace(",", " "),
                va="center", fontsize=8, color=INK_MUTED)
    ax.axvline(n, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=0)
    apply_axes_style(ax)
    ax.grid(False, axis="y")
    ax.set_xlabel("nodes (log)", color=INK_MUTED, fontsize=8)


def _coverage(ax, d):
    if d["coverage"]:
        steps, seen = np.array(d["coverage"]).T
        ax.plot(steps, seen, color=BLUE, linewidth=2)
    ax.axhline(d["n_nodes"], color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(1.2, d["n_nodes"] * 0.6, "|V|", color=INK_MUTED, fontsize=8)
    ax.axhline(d["reachable"], color=ORANGE, linewidth=1, linestyle=(0, (2, 2)))
    ax.text(1.2, d["reachable"] * 0.6, "reachable from seed", color=ORANGE, fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    apply_axes_style(ax)
    ax.set_xlabel("steps (log)", color=INK_MUTED, fontsize=8)
    ax.set_ylabel("distinct nodes seen", color=INK_MUTED, fontsize=8)


def _binned(x, y, bins=18):
    """Median von y je logarithmischem x-Bin -- robuster als ein Punktwolken-Plot."""
    keep = x > 0
    x, y = x[keep], y[keep]
    if len(x) < 10:
        return np.array([]), np.array([])
    edges = np.geomspace(x.min(), x.max() + 1, bins)
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    xs, ys = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 5:
            xs.append(np.median(x[m])); ys.append(np.median(y[m]))
    return np.array(xs), np.array(ys)


def _degree_panel(ax, d):
    counts, nodes = d["counts"], d["nodes"]
    for deg, color, label, rho in (
        (d["out_deg"][nodes], ORANGE, "out-degree", d["rho_out"]),
        (d["in_deg"][nodes], BLUE, "in-degree", d["rho_in"]),
    ):
        xs, ys = _binned(deg.astype(float), counts)
        if len(xs):
            ax.plot(xs, ys, "o-", color=color, markersize=4, linewidth=1.6,
                    label=f"{label}  (rho={rho:+.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    apply_axes_style(ax)
    ax.set_xlabel("degree (log)", color=INK_MUTED, fontsize=8)
    ax.set_ylabel("median visits", color=INK_MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper left")


def _entities(ax, d, top=10):
    rows = d["top"][:top]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r["share"] for r in rows], color=MAGENTA, height=0.6)
    labels = []
    for r in rows:
        name = str(r["name"])
        name = name if len(name) <= 26 else name[:23] + "..."
        labels.append(f"{name}\nout {r['deg_out']} / in {r['deg_in']}")
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("share of all visits", color=INK_MUTED, fontsize=8)
    apply_axes_style(ax)
    ax.grid(False, axis="y")
    for yi, r in zip(y, rows):
        ax.text(r["share"], yi, f"  {r['share']:.1%}", va="center",
                fontsize=7, color=INK_MUTED)


def plot_diagnosis(results: list[dict], path: Path | None = None) -> Path:
    rows = len(results)
    fig, axes = plt.subplots(rows, 4, figsize=(21, 4.9 * rows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for r, d in enumerate(results):
        _ladder(axes[r][0], d)
        _coverage(axes[r][1], d)
        _degree_panel(axes[r][2], d)
        _entities(axes[r][3], d)
        axes[r][0].set_title(
            f"{VIEW_TITLES.get(d['view'], d['view'])} -- where the estimate breaks",
            color=INK, fontsize=10, loc="left", pad=8)
        for c, title in enumerate(("", "coverage over time",
                                   "visits vs degree", "most visited entities")):
            if title:
                axes[r][c].set_title(title, color=INK, fontsize=10, loc="left", pad=8)

    d0 = results[0]
    fig.suptitle(
        f"{d0['graph']}: random walk diagnosis (dead_end={d0['dead_end']}, "
        f"budget={d0['budget_rel']:g})",
        color=INK, fontsize=12, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    if path is None:
        config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.PLOTS_DIR / f"{d0['graph']}__walk_diagnosis.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
