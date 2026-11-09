"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """Diskrete Nachfrage gleichverteilt auf 0, 1, 2, 3 (Mittelwert 1,5). Handrechnung für n = 2 (Summe 0 bis 6 mit Gewichten 1, 2, 3, 4, 3, 2, 1 durch 16):
    Servicegrad 0,9: Einzelquantil 3, Summenquantil 5 (kumuliert 15/16 = 0,9375 >= 0,9, davor 13/16 = 0,8125); SB1 = 3 - 1,5 = 1,5; SBn = 5 - 3 = 2; V = 2 * 1,5 / 2 = 1,5."""
    return {"values": [0.0, 1.0, 2.0, 3.0], "probs": [0.25] * 4, "sum_weights": [1, 2, 3, 4, 3, 2, 1]}
