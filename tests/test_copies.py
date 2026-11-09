"""Wörtliche Kopie des Zufallsmoduls: bue_rng.py ist bis auf die Kopfzeile gleich ab-test-demo/abt_rng.py (Portfolio-Konvention SplitMix64)."""

from pathlib import Path

import pytest

import bue_rng

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT.parent / "ab-test-demo" / "abt_rng.py"


def test_bue_rng_is_a_verbatim_copy_of_abt_rng_apart_from_the_header():
    if not SOURCE.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    mine = (ROOT / "bue_rng.py").read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith("# Wörtliche Kopie von ab-test-demo/abt_rng.py")
    assert mine[1:] == SOURCE.read_text(encoding="utf-8").splitlines(), "die Kopie weicht vom Original ab: Änderungen gehören ins Original, danach neu kopieren"


def test_the_copy_runs_on_its_own_with_reference_values():
    """SplitMix64 mit Startwert 0 (Referenzimplementierung von Vigna): die ersten Ausgaben."""
    g = bue_rng.SplitMix64(0)
    assert g.next() == 16294208416658607535
    assert g.next() == 7960286522194355700


def test_uniform_block_equals_scalar_stream():
    g = bue_rng.SplitMix64(5)
    assert list(bue_rng.uniform_block(5, 20)) == [g.uniform() for _ in range(20)]
