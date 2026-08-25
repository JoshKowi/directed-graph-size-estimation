"""Estimator-Basis: gemeinsame Schnittstelle aller Schaetzverfahren.

Ein Estimator bekommt Graph, absolutes Budget und RNG und liefert eine
Schaetzung von |V| plus die tatsaechlich verbrauchten Kosten.

Die Kategorie (nur zum Vergleich / real umsetzbar) haengt nicht am Estimator
selbst, sondern wird beim Registrieren vergeben -- dasselbe Verfahren kann je
nach Oracle in beide Kategorien fallen. `category` ist deshalb erst nach
`estimators.build(name)` gesetzt, siehe estimators/__init__.py.

Schnittstelle:
    class Category(StrEnum)
    class EstimateResult   -- value, cost, visits, extra
    class Estimator        -- .name, .category (spaet gesetzt), .estimate(...)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from graphs.graph import Graph


class Category(StrEnum):
    COMPARISON = "comparison"
    REALIZABLE = "realizable"


@dataclass
class EstimateResult:
    value: float
    cost: dict[str, int] = field(default_factory=dict)
    # Wie oft welcher Original-Knoten beruehrt wurde (aus dem Oracle).
    visits: Counter = field(default_factory=Counter)
    extra: dict = field(default_factory=dict)


class Estimator(ABC):
    name: str = "estimator"
    # Wird von estimators.build() aus der REGISTRY gesetzt, nicht hier.
    category: Category | None = None

    @abstractmethod
    def estimate(self, graph: Graph, budget: int, rng: random.Random) -> EstimateResult:
        """Schaetzt |V| unter Einhaltung des Budgets."""
