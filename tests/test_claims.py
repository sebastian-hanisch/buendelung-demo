"""Zahlen der README und der App-Texte, nachgerechnet (DEMO-PLAYBOOK §4 und §11a). Jede Zahl steht mit ihrer Voraussetzung hier; Bänder statt exakter Gleitkommawerte.

Faltung = FFT-Faltung auf feinem Gitter (ein Lauf je Zelle, deterministisch); Monte Carlo = Tabelle bue_mc_tabelle.json (5 Zufalls-Seeds, 400.000 Ziehungen je Zelle)."""

import math

import pytest

import bue_algorithm as A
import bue_constants as C
import bue_evaluation as ev


def _v(d, n, q):
    return A.advantage(d, n, q, A.sum_quantiles_fft(d, [n], [q]).quantiles[(n, q)])


@pytest.fixture(scope="module")
def sweep95():
    return ev.sweep_distributions(0.95)


@pytest.fixture(scope="module")
def exponents():
    return ev.exponent_table(ev.exponent_dists())


def test_b01_normal_is_sqrt_n_exactly_and_the_fft_agrees_within_0_3_percent():
    for n in C.N_SWEEP:
        assert ev.advantage_curve(A.Dist("normal"), 0.95)["v"][n] == pytest.approx(math.sqrt(n), rel=3e-3)
        assert A.advantage_normal(n) == pytest.approx(math.sqrt(n), rel=1e-12)


def test_b02_pareto_1_5_n32_95_percent_is_2_19_which_is_39_percent_of_sqrt_n():
    v = _v(A.Dist("pareto", 1.5), 32, 0.95)
    assert 2.17 <= v <= 2.20 and 0.38 <= v / math.sqrt(32) <= 0.40
    lo, med, hi = ev.mc_band(ev.load_mc_table(), A.Dist("pareto", 1.5), 0.95, 32)
    assert lo <= v <= hi and 2.16 <= lo and hi <= 2.21 and 2.16 <= med <= 2.19          # Monte Carlo: 5 Seeds, 400.000 Ziehungen


def test_b03_student_t3_n32_95_percent_is_4_87_and_inside_the_monte_carlo_band():
    v = _v(A.Dist("t", 3.0), 32, 0.95)
    lo, med, hi = ev.mc_band(ev.load_mc_table(), A.Dist("t", 3.0), 0.95, 32)
    assert 4.85 <= v <= 4.89 and lo <= v <= hi and 4.85 <= lo and hi <= 4.91


def test_b04_five_of_eight_distributions_are_more_than_5_percent_below_sqrt_n_at_n_8_32_and_256(sweep95):
    for n in (8, 32, 256):
        below = [r["label"] for r in sweep95 if r["v"][n] < 0.95 * math.sqrt(n)]
        assert len(below) == 5, (n, below)
        assert "Normal" not in below and "Student-t ν = 10,0" in [r["label"] for r in sweep95]
    worst = min(sweep95, key=lambda r: r["v"][256])
    assert worst["label"] == "Pareto α = 1,5" and 4.0 <= worst["v"][256] <= 4.25 and 0.25 <= worst["v"][256] / 16 <= 0.27


def test_b05_ranking_at_n32_95_percent_from_low_to_high(sweep95):
    order = [r["label"] for r in sorted(sweep95, key=lambda r: r["v"][32])]
    assert order == ["Pareto α = 1,5", "Pareto α = 2,0", "Pareto α = 3,0", "Student-t ν = 3,0", "Lognormal σ = 1,0", "Student-t ν = 10,0", "Normal", "Lognormal σ = 0,5"]


def test_b06_lognormal_sigma_0_5_exceeds_sqrt_n_at_n32_95_percent(sweep95):
    row = next(r for r in sweep95 if r["label"] == "Lognormal σ = 0,5")
    assert 6.1 <= row["v"][32] <= 6.3 and row["v"][32] > math.sqrt(32)


def test_b07_monte_carlo_table_median_is_within_1_5_percent_of_the_convolution_in_all_80_cells_and_the_convolution_is_inside_min_to_max():
    table = ev.load_mc_table()
    assert table["draws"] == 400_000 and len(table["seeds"]) == 5
    cells = 0
    for d in ev.default_dists():
        for q in (0.95, 0.99):
            res = A.sum_quantiles_fft(d, list(C.N_MC_TABLE), [q])
            for n in C.N_MC_TABLE:
                v = A.advantage(d, n, q, res.quantiles[(n, q)])
                lo, med, hi = ev.mc_band(table, d, q, n)
                assert abs(med / v - 1.0) <= 0.015 and lo <= v <= hi, (d, q, n)
                cells += 1
    assert cells == 80


def test_b08_one_monte_carlo_cell_of_the_table_is_reproduced_from_the_seed():
    d, q, n, seed = A.Dist("pareto", 2.0), 0.95, 8, C.SWEEP_SEEDS[0]
    sample = A.mc_sum_chain(d, [n], C.MC_DRAWS, seed)[n]
    v = A.advantage(d, n, q, A.mc_quantile(sample, q))
    assert v == pytest.approx(ev.load_mc_table()["cells"][ev.mc_key(d, q, n)][0], rel=1e-5)


def test_b09_service_level_raises_v_in_7_of_8_distributions_and_not_in_normal_at_n32():
    rows = ev.service_matrix(32)
    rising = [r["label"] for r in rows if r["v"][99.9] > r["v"][90.0] * (1 + 1e-6)]
    assert len(rising) == 7 and "Normal" not in rising
    p15 = next(r for r in rows if r["label"] == "Pareto α = 1,5")["v"]
    assert 1.55 <= p15[90.0] <= 1.70 and 3.0 <= p15[99.9] <= 3.2
    t3 = next(r for r in rows if r["label"] == "Student-t ν = 3,0")["v"]
    assert 4.4 <= t3[90.0] <= 4.5 and 8.3 <= t3[99.9] <= 8.6


def test_b10_correlation_n8_rho_half_is_1_33_limit_1_41_and_rho_one_removes_the_advantage():
    assert A.advantage_normal(8, 0.5) == pytest.approx(1.3333, abs=1e-3)
    assert 1.0 / math.sqrt(0.5) == pytest.approx(1.4142, abs=1e-3)
    assert A.advantage_normal(10 ** 6, 0.5) < 1.0 / math.sqrt(0.5) and A.advantage_normal(10 ** 6, 0.5) == pytest.approx(1.4142, abs=1e-3)
    assert all(A.advantage_normal(n, 1.0) == pytest.approx(1.0) for n in C.N_SWEEP)


def test_b11_slope_n64_to_256_is_within_0_05_of_the_asymptotic_exponent_in_12_of_12_heavy_tail_cells(exponents):
    heavy = [r for r in exponents if r["dist"].kind in ("pareto", "t") and r["dist"].param < 2.0]
    assert len(heavy) == 12 and all(abs(r["e_high"] - r["asymptote"]) <= 0.05 for r in heavy)
    cell = {(r["label"], r["q"]): r for r in exponents}
    assert 0.325 <= cell[("Pareto α = 1,5", 0.999)]["e_high"] <= 0.335 and 0.30 <= cell[("Pareto α = 1,5", 0.95)]["e_high"] <= 0.32
    assert 0.425 <= cell[("Pareto α = 1,8", 0.99)]["e_high"] <= 0.435 and 0.39 <= cell[("Pareto α = 1,8", 0.95)]["e_high"] <= 0.40


def test_b12_level_is_below_the_power_law_through_v1_equal_1(exponents):
    cell = {(r["label"], r["q"]): r for r in exponents}
    assert 0.63 <= cell[("Pareto α = 1,5", 0.95)]["level"] <= 0.66 and 0.96 <= cell[("Pareto α = 1,5", 0.999)]["level"] <= 0.98
    heavy = [r for r in exponents if r["dist"].kind in ("pareto", "t") and r["dist"].param < 2.0]
    assert all(r["level"] < 1.0 for r in heavy) and len([r for r in heavy if r["level"] < 0.9]) == 6


def test_b13_finite_variance_at_high_service_level_slope_above_one_half_and_v_above_sqrt_n(exponents):
    cell = {(r["label"], r["q"]): r for r in exponents}
    p3 = cell[("Pareto α = 3,0", 0.999)]
    assert 0.61 <= p3["e_high"] <= 0.64 and 1.9 <= p3["factor"] <= 2.0 and 3.1 <= p3["limit_factor"] <= 3.25 and p3["factor"] < p3["limit_factor"]
    assert 4.25 <= _v(A.Dist("t", 3.0), 16, 0.99) <= 4.35 and _v(A.Dist("t", 3.0), 16, 0.99) > 4.0
    ln = cell[("Lognormal σ = 1,0", 0.999)]
    assert 0.61 <= ln["e_high"] <= 0.64 and ln["factor"] > 1.0


def test_b14_advantage_below_one_for_pareto_1_2_at_90_percent_n16():
    assert 0.5 <= _v(A.Dist("pareto", 1.2), 16, 0.90) <= 0.6


def test_b15_default_preset_stock_numbers_in_pieces():
    a = ev.analyse(ev.Settings())
    assert a.total_single == pytest.approx(8 * 20 * 1.6448536, rel=1e-6) and 263.0 <= a.total_single <= 263.4
    assert a.sbn == pytest.approx(20 * 1.6448536 * math.sqrt(8), rel=1e-6) and 92.9 <= a.sbn <= 93.1


def test_b16_fft_accuracy_is_stable_against_the_grid_size_to_three_digits():
    for d, n, q in ((A.Dist("pareto", 1.5), 64, 0.99), (A.Dist("t", 1.5), 64, 0.99), (A.Dist("lognormal", 1.0), 64, 0.99)):
        coarse = A.advantage(d, n, q, A.sum_quantiles_fft(d, [n], [q], points_max=1 << 16).quantiles[(n, q)])
        fine = A.advantage(d, n, q, A.sum_quantiles_fft(d, [n], [q], points_max=1 << 19).quantiles[(n, q)])
        assert coarse == pytest.approx(fine, rel=3e-3), d


def test_b17_every_grid_in_the_experiment_tables_is_reliable(sweep95, exponents):
    assert all(r["reliable"] for r in sweep95) and all(r["reliable"] for r in exponents)


def test_b18_finite_variance_cells_at_99_and_99_9_percent_have_slope_above_one_half_in_6_of_6(exponents):
    fin = [r for r in exponents if r["q"] >= 0.99 and r["dist"].kind in ("pareto", "t", "lognormal") and (r["dist"].kind == "lognormal" or r["dist"].param > 2.0)]
    assert len(fin) == 6 and all(r["e_high"] > 0.5 for r in fin)
    # Einzelbestand gegen Normal-Näherung z * sigma: bei Servicegrad 99,9 % und Pareto 3 mehr als das Doppelte (Grenzfaktor = SB1 / (z sigma))
    assert A.variance_limit_factor(A.Dist("pareto", 3.0), 0.999) > 2.0
