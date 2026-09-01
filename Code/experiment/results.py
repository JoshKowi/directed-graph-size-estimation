"""Speichern und Laden der Ergebnisse (CSV je Graph unter data/results).

Ein Lauf ist durch (Graph, Seed, Einstiegsknoten) bestimmt. Alle drei stehen
als Spalte in der CSV *und* im Dateinamen, damit sich zwei Laeufe desselben
Graphen nicht gegenseitig ueberschreiben:

    <graph>__estimates.csv                       Default-Seed, Default-Einstieg
    <graph>__seed7__estimates.csv                anderer Seed
    <graph>__start-isaac-newton__estimates.csv   anderer Einstiegsknoten
    <graph>__seed7__start-isaac-newton__estimates.csv

Die Defaults bleiben ohne Zusatz, damit aeltere Ergebnisse weiter gefunden
werden.

Schnittstelle:
    seed_tag(seed) -> str
    start_tag(graph, start_node) -> str
    parse_stem(stem) -> (graph, seed, start, kind)
    save_results(df, graph_name, kind="estimates", seed=None, start=None) -> Path
    load_results(graph_name=None, kind="estimates", seed=None, start=None)
        -> pd.DataFrame
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


def start_tag(graph: str, start_node) -> str:
    """Namensbestandteil fuer den Einstiegsknoten -- leer beim Default.

    Default ist der erste Eintrag in config.SEED_NODES; Graphen ohne
    Einstiegsknoten (dort wird gleichverteilt gezogen) bekommen nie einen Zusatz.
    """
    if start_node is None:
        return ""
    known = config.seed_nodes(graph)
    if not known or str(start_node) == str(known[0]):
        return ""
    return f"start-{config.start_slug(start_node)}__"


def parse_stem(stem: str) -> tuple[str, int, str | None, str]:
    """Dateinamen zerlegen -- gilt fuer CSVs *und* Bilder.

        gpt4o_io__estimates          -> ("gpt4o_io", DEFAULT_SEED, None, "estimates")
        gpt4o_io__seed7__estimates   -> ("gpt4o_io", 7, None, "estimates")
        gpt4o_io__start-kurashiki__estimates
                                     -> ("gpt4o_io", DEFAULT_SEED, "kurashiki",
                                         "estimates")
        gpt4o_io__seed7__wis_rw__views -> ("gpt4o_io", 7, None, "wis_rw__views")

    Der Rest bleibt am Stueck: Plot-Slugs duerfen selbst `__` enthalten.
    Zurueck kommt der *Slug* des Einstiegsknotens (oder None fuer den Default),
    nicht der Knotenname -- aus dem Dateinamen ist er nicht wiederherstellbar.
    """
    graph, _, rest = stem.partition("__")
    seed, start = config.DEFAULT_SEED, None
    m = re.match(r"seed(-?\d+)__(.*)", rest)
    if m:
        seed, rest = int(m.group(1)), m.group(2)
    m = re.match(r"start-([a-z0-9-]+)__(.*)", rest)
    if m:
        start, rest = m.group(1), m.group(2)
    return graph, seed, start, rest


def _path(graph_name: str, kind: str, seed: int | None = None, start=None) -> Path:
    return (config.RESULTS_DIR /
            f"{graph_name}__{seed_tag(seed)}{start_tag(graph_name, start)}{kind}.csv")


def save_results(df: pd.DataFrame, graph_name: str, kind: str = "estimates",
                 seed: int | None = None, start=None) -> Path:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(graph_name, kind, seed, start)
    df.to_csv(path, index=False)
    return path


def load_results(graph_name: str | None = None, kind: str = "estimates",
                 seed: int | None = None, start=None) -> pd.DataFrame:
    """Ergebnisse laden; `seed=None` laedt alle vorhandenen Seeds.

    Mehrere Seeds landen in *einem* Frame -- unterschieden werden sie ueber die
    Spalte `seed`. Wer sie getrennt auswerten will (die Plot-Skripte tun das),
    gruppiert danach; wer sie zusammenwirft, mittelt ueber Laeufe verschiedener
    Zufallsstroeme, was fuer die Streuungsangabe falsch waere.
    """
    frames = []
    want_start = None if start is None else config.start_slug(start)
    for path in sorted(config.RESULTS_DIR.glob("*.csv")):
        g, s, st, k = parse_stem(path.stem)
        if k != kind or (graph_name is not None and g != graph_name):
            continue
        if seed is not None and s != int(seed):
            continue
        if want_start is not None:
            known = config.seed_nodes(g)
            default = config.start_slug(known[0]) if known else None
            if (st or default) != want_start:
                continue
        df = pd.read_csv(path)
        if "seed" not in df.columns:      # CSV aus einer Version vor --seed
            df["seed"] = s
        if "nested" not in df.columns:    # ... bzw. vor --checkpoint-budgets
            df["nested"] = False
        if "walk_group" not in df.columns:   # ... bzw. vor --share-walks
            df["walk_group"] = None
        if "start_node" not in df.columns:   # ... bzw. vor --start-node
            known = config.seed_nodes(g)
            df["start_node"] = str(known[0]) if known else "<zufaellig>"
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


def conditions_available(df: pd.DataFrame) -> list[tuple]:
    """Die (Seed, Einstiegsknoten)-Paare eines Frames.

    Beides trennt Laeufe voneinander und darf beim Plotten nicht vermischt
    werden -- die Plot-Skripte zeichnen je Paar ein eigenes Bild.
    """
    if df.empty:
        return []
    if "start_node" not in df.columns:
        return [(s, None) for s in seeds_available(df)]
    pairs = df[["seed", "start_node"]].drop_duplicates()
    return sorted((int(s), n) for s, n in pairs.itertuples(index=False))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Range der Schaetzungen je View, Estimator und Budget."""
    if "seed" not in df.columns:          # Frame aus einer Version vor --seed
        df = df.assign(seed=config.DEFAULT_SEED)
    if "nested" not in df.columns:
        df = df.assign(nested=False)
    if "start_node" not in df.columns:
        df = df.assign(start_node="<zufaellig>")
    if "walk_group" not in df.columns:
        df = df.assign(walk_group=None)
    return (
        # Der Seed ist Teil des Schluessels: zwei Laeufe mit verschiedenen
        # Zufallsstroemen sind verschiedene Laeufe, ihre Spannen duerfen nicht
        # in einen Punkt zusammenfallen.
        df.groupby(["graph", "view", "category", "estimator", "budget_rel", "seed",
                    "start_node"], dropna=False)
        .agg(
            n_runs=("estimate", "size"),
            est_min=("estimate", "min"),
            est_median=("estimate", "median"),
            est_max=("estimate", "max"),
            true_size=("true_size", "first"),
            # Genestet: die Punkte einer Laufnummer stammen aus einem Lauf und
            # sind ueber die Budgets korreliert (siehe experiment/runner.py).
            nested=("nested", "any"),
            # Geteilter Walk: dieselbe Trajektorie wie die anderen Estimators
            # derselben Gruppe -- der Vergleich zwischen ihnen ist gepaart.
            shared=("walk_group", "first"),
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
