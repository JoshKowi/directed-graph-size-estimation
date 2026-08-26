"""Speichern und Laden der Ergebnisse (CSV je Graph unter data/results).

Ein Lauf ist durch (Graph, Seed) bestimmt. Der Seed steht als Spalte in der
CSV *und* im Dateinamen, damit zwei Laeufe desselben Graphen sich nicht
gegenseitig ueberschreiben: `<graph>__estimates.csv` fuer den Default-Seed,
`<graph>__seed7__estimates.csv` fuer jeden anderen. Der Default bleibt ohne
Zusatz, damit aeltere Ergebnisse weiter gefunden werden.

Schnittstelle:
    seed_tag(seed) -> str
    parse_stem(stem) -> (graph, seed, kind)
    save_results(df, graph_name, kind="estimates", seed=None) -> Path
    load_results(graph_name=None, kind="estimates", seed=None) -> pd.DataFrame
    summarize(df) -> pd.DataFrame     (min/median/max je View x Estimator x Budget,
                                       plus erlaubtes/genutztes Budget)
    compare_views(df, reference="directed") -> pd.DataFrame
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

import config


def seed_tag(seed: int | None) -> str:
    """Namensbestandteil fuer einen Seed -- leer beim Default.

    Ein Lauf mit dem Default-Seed heisst weiterhin `<graph>__estimates.csv`;
    nur abweichende Seeds bekommen einen Zusatz. Sonst haetten aeltere
    Ergebnisse ploetzlich den falschen Namen.
    """
    return "" if seed is None or int(seed) == config.DEFAULT_SEED else f"seed{int(seed)}__"


def parse_stem(stem: str) -> tuple[str, int, str]:
    """Dateinamen zerlegen -- gilt fuer CSVs *und* Bilder.

        gpt4o_io__estimates            -> ("gpt4o_io", DEFAULT_SEED, "estimates")
        gpt4o_io__seed7__estimates     -> ("gpt4o_io", 7, "estimates")
        gpt4o_io__seed7__wis_rw__views -> ("gpt4o_io", 7, "wis_rw__views")

    Der Rest bleibt am Stueck: Plot-Slugs duerfen selbst `__` enthalten.
    """
    graph, _, rest = stem.partition("__")
    m = re.match(r"seed(-?\d+)__(.*)", rest)
    if m:
        return graph, int(m.group(1)), m.group(2)
    return graph, config.DEFAULT_SEED, rest


def _path(graph_name: str, kind: str, seed: int | None = None) -> Path:
    return config.RESULTS_DIR / f"{graph_name}__{seed_tag(seed)}{kind}.csv"


def save_results(df: pd.DataFrame, graph_name: str, kind: str = "estimates",
                 seed: int | None = None) -> Path:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(graph_name, kind, seed)
    df.to_csv(path, index=False)
    return path


def load_results(graph_name: str | None = None, kind: str = "estimates",
                 seed: int | None = None) -> pd.DataFrame:
    """Ergebnisse laden; `seed=None` laedt alle vorhandenen Seeds.

    Mehrere Seeds landen in *einem* Frame -- unterschieden werden sie ueber die
    Spalte `seed`. Wer sie getrennt auswerten will (die Plot-Skripte tun das),
    gruppiert danach; wer sie zusammenwirft, mittelt ueber Laeufe verschiedener
    Zufallsstroeme, was fuer die Streuungsangabe falsch waere.
    """
    frames = []
    for path in sorted(config.RESULTS_DIR.glob("*.csv")):
        g, s, k = parse_stem(path.stem)
        if k != kind or (graph_name is not None and g != graph_name):
            continue
        if seed is not None and s != int(seed):
            continue
        df = pd.read_csv(path)
        if "seed" not in df.columns:      # CSV aus einer Version vor --seed
            df["seed"] = s
        frames.append(df)
    if not frames:
        if graph_name is not None:
            raise FileNotFoundError(
                f"Keine Ergebnisse fuer {graph_name!r} (kind={kind}"
                + (f", seed={seed}" if seed is not None else "") + ")")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def seeds_available(df: pd.DataFrame) -> list[int]:
    """Die Seeds, die in einem geladenen Frame stecken -- aufsteigend."""
    if df.empty or "seed" not in df.columns:
        return []
    return sorted(int(s) for s in df["seed"].dropna().unique())


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Range der Schaetzungen je View, Estimator und Budget."""
    if "seed" not in df.columns:          # Frame aus einer Version vor --seed
        df = df.assign(seed=config.DEFAULT_SEED)
    return (
        # Der Seed ist Teil des Schluessels: zwei Laeufe mit verschiedenen
        # Zufallsstroemen sind verschiedene Laeufe, ihre Spannen duerfen nicht
        # in einen Punkt zusammenfallen.
        df.groupby(["graph", "view", "category", "estimator", "budget_rel", "seed"],
                   dropna=False)
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
    if "seed" in df.columns:      # sonst paart der Merge unten Laeufe
        keys.append("seed")       # aus verschiedenen Zufallsstroemen

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
