"""Laden der Testgraphen aus adjacencies/ -- einmal pro Experiment-Lauf.

Es wird bewusst hoechstens ein Graph gleichzeitig im RAM gehalten (die
Dateien sind bis ~2 GB gross).

Schnittstelle:
    available_graphs() -> list[str]
    load_graph(name) -> Graph      (cached, verdraengt den vorherigen Graphen)
    clear_cache()
"""

from __future__ import annotations

import gc

import config
from graphs.graph import Graph, load_pickle

_CACHE: dict[str, Graph] = {}


def available_graphs() -> list[str]:
    return sorted(p.stem for p in config.ADJACENCIES_DIR.glob("*.pkl"))


def load_graph(name: str) -> Graph:
    name = config.resolve_graph(name)      # Kuerzel erlauben, s. config
    if name not in _CACHE:
        clear_cache()
        _CACHE[name] = load_pickle(config.ADJACENCIES_DIR / f"{name}.pkl")
    return _CACHE[name]


def clear_cache() -> None:
    _CACHE.clear()
    gc.collect()
