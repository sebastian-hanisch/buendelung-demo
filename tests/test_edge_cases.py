"""Randfälle der Kernroutinen (DEMO-PLAYBOOK §12): n = 1, n = 2, Reglerränder, ungültige Parameter, nicht positiver gemeinsamer Bestand."""

import math

import pytest

import bue_algorithm as A
import bue_constants as C
import bue_evaluation as ev


def test_n_equals_one_has_advantage_one_for_every_distribution():
    for d in (A.Dist("t", 3.0), A.Dist("pareto", 1.5), A.Dist("lognormal", 1.0), A.Dist("normal")):
        res = A.sum_quantiles_fft(d, [1], [0.95])
        assert A.advantage(d, 1, 0.95, res.quantiles[(1, 0.95)]) == pytest.approx(1.0, rel=0.02), d


def test_exactly_two_locations_for_every_distribution():
    """n = 2: der kleinste Fall mit echter Bündelung; V liegt für jede der acht Verteilungen über 1 und unter 1,5 (bei Lognormal σ = 0,5 sogar über sqrt(2): V > √n ist möglich)."""
    for d in ev.default_dists():
        res = A.sum_quantiles_fft(d, [2], [0.95])
        v = A.advantage(d, 2, 0.95, res.quantiles[(2, 0.95)])
        assert 1.0 < v < 1.5, (d, v)


def test_non_positive_pooled_safety_stock_gives_nan_not_a_number():
    d = A.Dist("pareto", 1.5)
    assert math.isnan(A.advantage(d, 4, 0.9, 4 * A.mean(d)))            # Quantil genau gleich n mal Mittelwert
    assert math.isnan(A.advantage(d, 4, 0.9, 3.0))                      # darunter


def test_invalid_distribution_parameters_are_rejected():
    for bad in (("t", 1.0), ("t", 0.5), ("pareto", 1.0), ("lognormal", 0.0), ("lognormal", -1.0), ("cauchy", 2.0)):
        with pytest.raises(ValueError):
            A.Dist(*bad)


def test_pmf_power_rejects_n_zero():
    import numpy as np

    with pytest.raises(ValueError):
        A.pmf_power_trunc(np.array([1.0]), 0, 1)


def test_correlation_one_removes_the_advantage_and_zero_gives_sqrt():
    for n in (2, 17, 256):
        assert A.advantage_normal(n, 1.0) == pytest.approx(1.0)
        assert A.advantage_normal(n, 0.0) == pytest.approx(math.sqrt(n))


@pytest.mark.parametrize("kind", C.KIND_ORDER)
@pytest.mark.parametrize("q", C.Q_OPTIONS)
def test_regler_maxima_and_minima_give_finite_or_flagged_results(kind, q):
    """Jede Verteilung am schwersten und am leichtesten Rand der Parameter, größtes n, jeder Servicegrad: kein Fehler; Ergebnis endlich oder als nicht verlässlich/nicht definiert gekennzeichnet."""
    params = {"normal": [0.0], "t": [C.NU_MIN, C.NU_MAX], "pareto": [C.ALPHA_MIN, C.ALPHA_MAX], "lognormal": [C.SIGMA_MIN, C.SIGMA_MAX]}[kind]
    for p in params:
        for n in (C.N_MIN, C.N_MAX):
            s = ev.Settings(kind=kind, nu=p if kind == "t" else 3.0, alpha=p if kind == "pareto" else 1.5, sigma=p if kind == "lognormal" else 1.0, q=q, n=n)
            a = ev.analyse(s)
            assert (math.isfinite(a.v) and a.v > 0) or math.isnan(a.v) or not a.reliable, (kind, p, q, n, a.v)
            text = ev.verdict_kernfrage(a)[1]
            assert text and "nan" not in text


def test_pareto_alpha_min_low_service_can_give_advantage_below_one():
    d = A.Dist("pareto", C.ALPHA_MIN)
    res = A.sum_quantiles_fft(d, [16], [0.9])
    assert A.advantage(d, 16, 0.9, res.quantiles[(16, 0.9)]) < 1.0
