"""Einheiten des Kerns gegen Handrechnung und einfache Gegenrechnungen (DEMO-PLAYBOOK §1b)."""

import math

import numpy as np
import pytest

import bue_algorithm as A
import bue_constants as C


def test_service_quantile_by_hand(mini):
    v, p = mini["values"], mini["probs"]
    assert A.service_quantile(v, p, 0.9) == 3.0
    assert A.service_quantile(v, p, 0.75) == 2.0          # genau 0,75 kumuliert: der Wert 2 genügt (inklusive Grenze)
    assert A.service_quantile(v, p, 0.7500001) == 3.0
    assert A.service_quantile(v, p, 0.0) == 0.0


def test_hand_example_n2_advantage_is_1_5(mini):
    pmf = np.array(mini["probs"])
    two = A.pmf_power_trunc(pmf, 2, 8)
    assert two[:7] == pytest.approx(np.array(mini["sum_weights"]) / 16.0)
    values = np.arange(8, dtype=float)
    q_sum = A.service_quantile(values, two, 0.9)
    assert q_sum == 5.0
    sb1, sbn = 3.0 - 1.5, q_sum - 2 * 1.5
    assert 2 * sb1 / sbn == pytest.approx(1.5)


def test_distribution_means_and_cdfs():
    assert A.mean(A.Dist("normal")) == 0.0 and A.mean(A.Dist("t", 3.0)) == 0.0
    assert A.mean(A.Dist("pareto", 1.5)) == pytest.approx(3.0)
    assert A.mean(A.Dist("lognormal", 1.0)) == pytest.approx(math.exp(0.5))
    for d in (A.Dist("normal"), A.Dist("t", 3.0), A.Dist("pareto", 2.0), A.Dist("lognormal", 0.7)):
        u = np.array([0.1, 0.5, 0.9, 0.99])
        assert A.cdf(d, A.ppf(d, u)) == pytest.approx(u, rel=1e-9)


def test_pareto_by_hand():
    d = A.Dist("pareto", 2.0)
    assert float(A.cdf(d, 2.0)) == pytest.approx(0.75)           # 1 - 2^-2
    assert float(A.ppf(d, 0.75)) == pytest.approx(2.0)
    assert A.mean(d) == pytest.approx(2.0)


def test_partial_mean_is_the_integral_of_t_f():
    """Teilmoment gegen numerische Integration (Trapezregel), für jede Verteilung."""
    for d, lo, hi in ((A.Dist("normal"), -8, 1.3), (A.Dist("t", 3.0), -300, 2.0), (A.Dist("pareto", 3.0), 1.0, 4.0), (A.Dist("lognormal", 0.6), 1e-9, 3.0)):
        xs = np.linspace(lo, hi, 400001)
        f = np.gradient(A.cdf(d, xs), xs)
        integral = np.trapezoid(xs * f, xs)
        assert float(A.partial_mean(d, hi) - A.partial_mean(d, lo)) == pytest.approx(integral, rel=2e-3, abs=2e-3), d


@pytest.mark.parametrize("d", [A.Dist("normal"), A.Dist("t", 2.5), A.Dist("pareto", 1.5), A.Dist("lognormal", 1.0)])
def test_lattice_preserves_mass_and_mean(d):
    h = 0.05
    if d.kind in ("pareto", "lognormal"):
        pmf = A.lattice_masses(d, h, 0, 40000)
        nodes = np.arange(len(pmf)) * h
        upper = len(pmf) * h
        mass = float(A.cdf(d, upper - h) - A.cdf(d, 0.0))
        assert pmf.sum() == pytest.approx(float(A.cdf(d, (len(pmf) - 1) * h)), abs=1e-12)
        assert float(np.sum(nodes * pmf)) == pytest.approx(float(A.partial_mean(d, (len(pmf) - 1) * h)), rel=1e-9)
        assert mass > 0.9
    else:
        pmf = A.lattice_masses(d, h, -4000, 8000)
        nodes = np.arange(-4000, 4001) * h
        assert pmf.sum() == pytest.approx(float(A.cdf(d, 4000 * h) - A.cdf(d, -4000 * h)), abs=1e-12)
        assert float(np.sum(nodes * pmf)) == pytest.approx(float(A.partial_mean(d, 4000 * h) - A.partial_mean(d, -4000 * h)), abs=1e-9)


def test_convolve_trunc_matches_direct_convolution():
    rng = np.random.default_rng(3)
    a, b = rng.random(37), rng.random(37)
    assert A.convolve_trunc(a, b, 37) == pytest.approx(np.convolve(a, b)[:37], rel=1e-9, abs=1e-12)


def test_pmf_power_matches_repeated_convolution():
    p = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    expected = p.copy()
    for _ in range(4):
        expected = np.convolve(expected, p)
    assert A.pmf_power_trunc(p, 5, 12) == pytest.approx(expected[:12], abs=1e-12)
    assert A.pmf_power_trunc(p, 1, 12) == pytest.approx(p)


def test_circular_power_for_a_shifted_lattice():
    """Zwei Würfel-ähnliche Werte um 0: Masse 1/3 auf -1, 0, 1; die Summe von 2 hat Gewichte 1, 2, 3, 2, 1 durch 9."""
    m = 16
    p = np.zeros(m)
    p[m // 2 - 1: m // 2 + 2] = 1.0 / 3.0
    s = A.pmf_power_circular(p, 2)
    assert s[m // 2 - 2: m // 2 + 3] == pytest.approx(np.array([1, 2, 3, 2, 1]) / 9.0)


def test_lattice_quantile_interpolates_inside_the_cell():
    pmf = np.array([0.25, 0.25, 0.25, 0.25])
    assert A.lattice_quantile(pmf, 1.0, 0.5) == pytest.approx(1.5)      # Zelle 1 reicht von 0,5 bis 1,5, kumuliert 0,5 am rechten Rand
    assert A.lattice_quantile(pmf, 1.0, 0.375) == pytest.approx(1.0)
    assert math.isnan(A.lattice_quantile(pmf, 1.0, 1.1))


def test_safety_stock_single_for_the_standard_normal():
    assert A.safety_single(A.Dist("normal"), 0.95) == pytest.approx(1.6448536269514722)
    assert A.safety_single(A.Dist("pareto", 2.0), 0.75) == pytest.approx(2.0 - 2.0)


def test_advantage_normal_closed_form():
    assert A.advantage_normal(8) == pytest.approx(math.sqrt(8))
    assert A.advantage_normal(8, 0.5) == pytest.approx(math.sqrt(8) / math.sqrt(4.5))
    assert A.advantage_normal(5, 1.0) == pytest.approx(1.0)


def test_growth_exponent_of_a_power_law():
    assert A.growth_exponent(3.0 * 4 ** 0.3, 4, 3.0 * 64 ** 0.3, 64) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        A.growth_exponent(1.0, 4, 1.0, 4)


def test_asymptotic_exponent_by_class():
    assert A.asymptotic_exponent(A.Dist("pareto", 1.5)) == pytest.approx(1 / 3)
    assert A.asymptotic_exponent(A.Dist("t", 1.2)) == pytest.approx(0.2 / 1.2)
    assert A.asymptotic_exponent(A.Dist("pareto", 2.0)) == 0.5 and A.asymptotic_exponent(A.Dist("lognormal", 1.0)) == 0.5


def test_variance_limit_factor_known_cases():
    assert A.variance_limit_factor(A.Dist("normal"), 0.99) == 1.0
    assert math.isnan(A.variance_limit_factor(A.Dist("pareto", 1.5), 0.95))
    d = A.Dist("t", 3.0)
    assert A.variance_limit_factor(d, 0.95) == pytest.approx((2.3533634348 - 0.0) / (1.6448536269514722 * math.sqrt(3.0)), rel=1e-6)


def test_mc_columns_are_independent_streams_and_chain_equals_single():
    d = A.Dist("normal")
    chain = A.mc_sum_chain(d, [2, 4], 1000, 7)
    assert chain[4] == pytest.approx(A.mc_sum_samples(d, 4, 1000, 7))
    assert chain[2] == pytest.approx(A.mc_sum_samples(d, 2, 1000, 7))
    assert not np.allclose(chain[2], chain[4])
    a, b = A.draw(d, 11, 500), A.draw(d, 12, 500)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.15
    assert A.draw(d, 11, 500) == pytest.approx(a)             # gleicher Seed, gleiche Ziehungen


def test_mc_quantile_equals_service_quantile(mini):
    sample = np.array([0.0, 1.0, 2.0, 3.0] * 25)
    for q in (0.1, 0.5, 0.75, 0.9, 0.99):
        assert A.mc_quantile(sample, q) == A.service_quantile(mini["values"], mini["probs"], q)


def test_limits_documented_in_constants_are_consistent():
    assert C.MC_N_MAX <= max(C.N_MC_TABLE) and set(C.N_MC_TABLE) <= set(C.N_SWEEP) and C.N_SWEEP[-1] == C.N_MAX
