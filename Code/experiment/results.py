"""Speichern und Laden der Ergebnisse (CSV je Graph unter data/results).

Schnittstelle:
    save_results(df, graph_name, kind="estimates") -> Path
    load_results(graph_name=None, kind="estimates") -> pd.DataFrame
    summarize(df) -> pd.DataFrame     (min/median/max je View x Estimator x Budget,
                                       plus erlaubtes/genutztes Budget)
    compare_views(df, reference="directed") -> pd.DataFrame
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config


def _path(graph_name: str, kind: str) -> Path:
    return config.RESULTS_DIR / f"{graph_name}__{kind}.csv"


def save_results(df: pd.DataFrame, graph_name: str, kind: str = "estimates") -> Path:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(graph_name, kind)
    df.to_csv(path, index=False)
    return path


def load_results(graph_name: str | None = None, kind: str = "estimates") -> pd.DataFrame:
    if graph_name is not None:
        return pd.read_csv(_path(graph_name, kind))
    frames = [pd.read_csv(p) for p in sorted(config.RESULTS_DIR.glob(f"*__{kind}.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Range der Schaetzungen je View, Estimator und Budget."""
    return (
        df.groupby(["graph", "view", "category", "estimator", "budget_rel"])
        .agg(
            n_runs=("estimate", "size"),
            est_min=("estimate", "min"),
            est_median=("estimate", "median"),
            est_max=("estimate", "max"),
            true_size=("true_size", "first"),
            # fuer die Achsenbeschriftung: erlaubtes und tatsaechlich
            # ausgegebenes Budget (siehe plotting.ranges.budget_ticks)
            budget_abs=("budget_abs", "first"),
            used_median=("queries_used", "median"),
            used_min=("queries_used", "min"),
            used_max=("queries_used", "max"),
        )
        .reset_index()
    )


def visit_summary(visits: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Haeufigste und seltenste besuchte Knoten je Estimator und Budget
    (Original-Knotennamen)."""
    out = []
    for _, grp in visits.groupby(["graph", "estimator", "budget_rel"]):
        grp = grp.sort_values("visits", ascending=False)
        for label, part in (("top", grp.head(top)), ("bottom", grp.tail(top))):
            part = part.copy()
            part["rank_group"] = label
            out.append(part)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def compare_views(df: pd.DataFrame, reference: str = "directed") -> pd.DataFrame:
    """Vergleichszahlen zwischen den Kantensichten.

    Je View: Median und Spannweite der Schaetzung, jeweils relativ zur wahren
    Groesse. Zusaetzlich `ratio_vs_<reference>`: der Median des *gepaarten*
    Verhaeltnisses (gleicher Lauf, gleicher Seed) gegenueber der Referenz-View.
    Werte > 1 heissen: in dieser View schaetzt das Verfahren hoeher.
    """
    keys = ["graph", "category", "estimator", "budget_rel"]

    agg = (
        df.groupby(keys + ["view"])
        .agg(
            median_rel=("estimate", lambda x: x.median()),
            min_rel=("estimate", "min"),
            max_rel=("estimate", "max"),
            true_size=("true_size", "first"),
        )
        .reset_index()
    )
    for col in ("median_rel", "min_rel", "max_rel"):
        agg[col] = agg[col] / agg["true_size"]
    agg["spread_rel"] = agg["max_rel"] - agg["min_rel"]

    # gepaarter Vergleich: gleicher run == gleicher Zufallsstrom
    ref = df[df["view"] == reference][keys + ["run", "estimate"]]
    paired = df.merge(ref, on=keys + ["run"], suffixes=("", "_ref"))
    paired["ratio"] = paired["estimate"] / paired["estimate_ref"]
    ratio = (
        paired.groupby(keys + ["view"])["ratio"]
        .median()
        .reset_index()
        .rename(columns={"ratio": f"ratio_vs_{reference}"})
    )

    out = agg.merge(ratio, on=keys + ["view"], how="left")
    return out[keys + ["view", "median_rel", "min_rel", "max_rel", "spread_rel",
                       f"ratio_vs_{reference}", "true_size"]].sort_values(keys + ["view"])
