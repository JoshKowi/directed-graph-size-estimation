"""Vergleichs-Plot: frei gewaehlte Estimators, frei gewaehlte Kantensichten.

Eine Spalte je View, eine Farbe je Estimator -- nebeneinander stehen genau die
Reihen, die man vergleichen will. Die Kategorie (Vergleich / real umsetzbar)
teilt das Bild bewusst *nicht* auf: sie ist ein Label der REGISTRY und sagt
nichts darueber, welche Verfahren man nebeneinander sehen moechte.

Gezeigt wird pro Estimator und Budget die Spanne min..max ueber die n Laeufe
plus der Median, je Estimator leicht gegeneinander versetzt (sonst verdecken
sich Verfahren mit gleichem Wert). Die y-Achse ist Schaetzung/|V| (log), die gestrichelte Linie
bei 1.0 ist die wahre Groesse.

Die Breite je Spalte ist fest, bis mehr als BUDGETS_PER_PANEL Budgets auf der
x-Achse stehen; danach waechst sie je weiterem Budget um PANEL_GROWTH Zoll,
damit die zweizeiligen Budget-Labels nicht zusammenruecken.

`note` steht klein oben rechts und nennt die Herkunft des Bildes -- vor allem
den Seed. Ohne ihn ist nicht zu erkennen, ob zwei Bilder desselben Graphen
denselben Zufallsstrom zeigen oder zwei unabhaengige Durchlaeufe.

Schnittstelle:
    VIEW_TITLES: dict[str, str]
    budget_ticks(panel, budgets) -> list[str]
    plot_comparison(summary, graph_name, estimators, views, title, path,
                    colors=None, note=None) -> matplotlib.figure.Figure
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from plotting.style import INK, INK_MUTED, SURFACE, apply_axes_style, color_for  # noqa: E402


# Ab wie vielen Budgets die Spalte breiter wird, und um wie viel Zoll je
# weiterem Budget. 6.6 Zoll tragen sechs zweizeilige x-Labels bequem.
BUDGETS_PER_PANEL = 6
PANEL_GROWTH = 0.9

# Titelzeichen je Zoll Figurbreite bei fontsize 11 -- nur fuer den Umbruch.
CHARS_PER_INCH = 7.9

#: Hoehe je zusaetzlicher Legendenzeile (ab der dritten), in Zoll.
LEGEND_ROW_INCHES = 0.26


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
# Praesentationen); Kommentare und Docstrings bleiben deutsch. Die Reihenfolge
# ist zugleich die Spaltenreihenfolge im Bild -- nicht alphabetisch sortieren,
# sonst stuende `reverse` vor `undirected`.
#
# `undirected` steht links: dort gilt pi ~ deg, die Schaetzer arbeiten unter
# ihrer eigenen Voraussetzung. Das ist der Bezugspunkt, gegen den sich rechts
# ablesen laesst, was die Richtung kostet. Umgekehrt gelesen -- erst der
# schwierige Fall, dann der einfache -- steht die Referenz hinter dem, was sie
# einordnen soll.
#: Ab dieser Spanne in Prozentpunkten zeigt die Legende sie statt eines Werts.
JUMP_SPAN_SHOWN_FROM = 2.0


def jump_rate_label(rates) -> str:
    """Sprunganteil einer Datenreihe als Legenden-Zusatz.

    Der Anteil (n_random_node / extra_n_samples) ist ueber die Budgets fast
    konstant: bei w = 100 schwankt er zwischen 93,0 und 94,1 %, bei w = 0,1
    zwischen 1,8 und 2,6 %. Er gehoert deshalb an die *Reihe*, nicht an jeden
    Punkt -- eine eigene Farbdimension dafuer waere viel Apparat fuer wenig
    Information und naehme der Reihenkennung die Farbe weg.

    Schwankt er merklich -- bei w = 1 auf gpt4_io von 12 auf 20 %, weil die
    Einfrier-Regel spaet besuchten Knoten kleinere G_u-Grade gibt und der Walk
    dadurch mit der Zeit oefter springt -- steht die Spanne da statt eines
    Werts, der das verschweigt.
    """
    vals = rates.dropna() * 100.0
    if vals.empty:
        return ""
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < JUMP_SPAN_SHOWN_FROM:
        return f"  ({(lo + hi) / 2:.0f} % jumps)"
    return f"  ({lo:.0f}-{hi:.0f} % jumps)"


VIEW_TITLES = {
    "undirected": "undirected (symmetrized)",
    "directed": "directed (original)",
    "reverse": "reverse (in-edges only)",
}


def plot_comparison(
    summary,
    graph_name: str,
    estimators: list[str],
    views: list[str],
    title: str,
    path: Path | None = None,
    colors: dict[str, str] | None = None,
    note: str | None = None,
    jump_rates: bool = False,
):
    """summary: DataFrame aus experiment.results.summarize().

    `jump_rates` haengt jeder Legendenzeile den Sprunganteil an, also
    n_random_node/extra_n_samples: bei DURW die Spruenge, bei
    oracles.name_list die erfolgreichen Ziehungen.

    Gedacht fuer den w-Sweep: `w` steuert ueber w/(w + deg_Gu(v)), wie oft DURW
    springt, und die Sprungrate ist der Grund, warum derselbe Schaetzer auf
    gpt4o_io schon bei 0,5 % Budget trifft und auf gpt4_io bei 10 % noch nicht
    (28 % gegen 15 % Spruenge). So steht die Ursache im selben Bild wie die
    Wirkung, ohne eine Achse dafuer zu verbrauchen.
    """
    if jump_rates and "jump_rate_median" not in summary.columns:
        raise ValueError(
            "Spalte jump_rate_median fehlt -- die Ergebnisse stammen aus einer "
            "Version vor n_random_node/extra_n_samples. Ohne sie gibt es keinen "
            "Sprunganteil zu zeigen."
        )
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

    # Ab BUDGETS_PER_PANEL Budgets waechst die Spalte mit: bis dahin passen die
    # zweizeiligen x-Labels (relativ + absolut) nebeneinander, danach ruecken
    # sie zusammen und werden unleserlich. Statt sie zu drehen oder zu kuerzen
    # bekommt die Grafik je weiterem Budget ein Stueck Breite dazu.
    panel_width = 6.6 + PANEL_GROWTH * max(0, len(budgets) - BUDGETS_PER_PANEL)
    fig_width = panel_width * len(views)

    # Legendentext, Spaltenzahl und Zeilenzahl stehen *vor* der Figur fest --
    # die Figurhoehe haengt davon ab. Der Sprunganteil wird ueber alle
    # gezeigten Views gebildet, nicht nur ueber das erste Panel: sonst stuende
    # in der Legende die Rate einer Sicht und daneben die Punkte einer anderen.
    label_for = {
        e: e + (jump_rate_label(summary.loc[summary["estimator"] == e,
                                            "jump_rate_median"])
                if jump_rates else "")
        for e in estimators
    }
    shown = [label_for[e] for e in estimators]
    widest = max(len(x) for x in shown) + 5           # Symbol + Abstand
    ncol = max(1, min(len(shown), 3, int(CHARS_PER_INCH * fig_width) // widest))
    n_legend_rows = -(-len(shown) // ncol)
    # Ab drei Legendenzeilen waechst die Figur mit, statt die Panels zu
    # stauchen -- mit --jump-rates sind die Zeilen laenger, passen also
    # weniger nebeneinander, und bei sieben Reihen bliebe sonst kaum
    # Zeichenflaeche uebrig.
    fig_height = 4.6 + LEGEND_ROW_INCHES * max(0, n_legend_rows - 2)
    fig, axes = plt.subplots(1, len(views), figsize=(fig_width, fig_height),
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
                    markeredgecolor=SURFACE, markeredgewidth=2, label=label_for[est],
                    zorder=4)

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

    # Kopfzone von oben nach unten: Titel (links) neben Hinweis (rechts),
    # darunter die Legende, darunter die Achsen. Titel und Legende sind beide
    # mehrzeilig, ihre Hoehe muss deshalb durchgereicht werden -- sonst schiebt
    # sich bei einer schmalen Figur der umbrochene Titel in die Legende.
    #
    # Der Hinweis steht in der Titelzeile und braucht dort Platz: ohne den
    # Abzug laeuft der Titel bei nur einer Spalte in ihn hinein.
    wrapped = textwrap.fill(
        title, width=max(30, int(CHARS_PER_INCH * fig_width)
                             - (len(note) // 2 if note else 0)))
    fig.suptitle(wrapped, color=INK, fontsize=11, x=0.01, ha="left", va="top", y=0.985)
    if note:
        fig.text(0.99, 0.985, note, color=INK_MUTED, fontsize=9, ha="right", va="top")

    handles, labels = axes[0][0].get_legend_handles_labels()
    legend_top = 0.985 - 0.042 * (wrapped.count("\n") + 1) - 0.01
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK_MUTED,
               ncol=ncol, loc="upper left",
               bbox_to_anchor=(0.01, legend_top), handletextpad=0.4,
               columnspacing=1.4)

    # Reservierung in Zoll, nicht als fester Anteil: die Figurhoehe waechst
    # oben mit der Zeilenzahl, ein Anteil liesse die Luecke mitwachsen.
    fig.tight_layout(rect=(0, 0, 1, legend_top
                           - (0.19 * n_legend_rows + 0.10) / fig_height))

    if path is None:
        config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.unique_path(config.PLOTS_DIR / f"{graph_name}__comparison.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return fig
