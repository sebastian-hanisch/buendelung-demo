"""Szenario: Instanzen der Gerüst-Aufgabe (aus n Kandidaten die k billigsten wählen). Durch die echte Aufgabe ersetzen."""

from dataclasses import dataclass

import numpy as np

import bue_constants as C


@dataclass(frozen=True)
class Instance:
    costs: tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.costs)


def make_instance(n: int, seed: int) -> Instance:
    """Deterministisch aus (n, seed): gleiche Eingabe, gleiche Instanz."""
    rng = np.random.default_rng(seed)
    return Instance(tuple(round(float(c), 2) for c in rng.uniform(C.COST_LOW, C.COST_HIGH, size=n)))
