"""Orakel-Tests (DEMO-PLAYBOOK §12): jede Kernroutine gegen einen anderen Rechenweg.

(1) Normal: gegen die geschlossene Formel. (2) Summe schwerer Verteilungen: Faltung der Gitterverteilung gegen die Faltung der scipy-Dichten
(`scipy.stats`, `scipy.signal.fftconvolve`, Quantil aus der aufsummierten Dichte) und gegen Monte Carlo. (3) Gleichkorrelation gegen simulierte Normalvektoren
(`scipy.stats.multivariate_normal`). (4) Quantil-Funktion gegen `numpy.quantile` auf der Aufzählung kleiner diskreter Verteilungen (n = 2).
Das Orakel (2) wurde vorher an der Handrechnung geprüft (Normal: Summe von 2 hat Standardabweichung sqrt(2))."""

import math

import numpy as np
import pytest

import bue_algorithm as A

stats = pytest.importorskip("scipy.stats")
signal = pytest.importorskip("scipy.signal")


# --- (1) Normal gegen die geschlossene Formel ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 8, 32, 256])
@pytest.mark.parametrize("q", [0.9, 0.99])
def test_normal_fft_matches_sqrt_n(n, q):
    d = A.Dist("normal")
    res = A.sum_quantiles_fft(d, [n], [q])
    assert A.advantage(d, n, q, res.quantiles[(n, q)]) == pytest.approx(math.sqrt(n), rel=3e-3)


# --- (2) Summe schwerer Verteilungen: Gitterfaltung gegen scipy-Dichten -------------------------------------------------------------------

def _scipy_pdf(d, x):
    if d.kind == "t":
        return stats.t.pdf(x, d.param)
    if d.kind == "pareto":
        return stats.pareto.pdf(x, d.param)
    if d.kind == "lognormal":
        return stats.lognorm.pdf(x, d.param)
    return stats.norm.pdf(x)


def _sum_quantile_by_density_convolution(d, n, q, h, lo, hi):
    """Quantil der Summe von n Variablen: Dichte auf dem Gitter lo, lo + h, ... (mit Zellmitten-Regel), n-fache Faltung per scipy.signal.fftconvolve, aufsummieren."""
    xs = lo + h * (np.arange(int(round((hi - lo) / h))) + 0.5)
    f = _scipy_pdf(d, xs) * h
    total, offset = f.copy(), n * lo
    for _ in range(n - 1):
        total = signal.fftconvolve(total, f)
    cum = np.cumsum(total)
    idx = int(np.searchsorted(cum, q))
    # Knoten k der Summe liegt bei n*lo + (k + n/2) h (n Zellmitten); lineare Interpolation innerhalb der Zelle
    prev = cum[idx - 1] if idx else 0.0
    return offset + (idx + n / 2.0 - 0.5) * h + h * (q - prev) / total[idx]


@pytest.mark.parametrize(
    "d, n, q, lo, hi, h",
    [
        (A.Dist("t", 3.0), 2, 0.95, -400.0, 400.0, 0.01),
        (A.Dist("t", 3.0), 4, 0.99, -400.0, 400.0, 0.01),
        (A.Dist("pareto", 1.5), 2, 0.95, 1.0, 600.0, 0.01),
        (A.Dist("pareto", 3.0), 4, 0.95, 1.0, 200.0, 0.01),
        (A.Dist("lognormal", 1.0), 4, 0.95, 1e-6, 300.0, 0.01),
        (A.Dist("t", 10.0), 4, 0.9, -200.0, 200.0, 0.01),
    ],
)
def test_sum_quantile_matches_scipy_density_convolution(d, n, q, lo, hi, h):
    mine = A.sum_quantiles_fft(d, [n], [q]).quantiles[(n, q)]
    oracle = _sum_quantile_by_density_convolution(d, n, q, h, lo, hi)
    assert mine == pytest.approx(oracle, rel=0.01), (d, n, q)


@pytest.mark.parametrize("d, n, q", [(A.Dist("t", 3.0), 4, 0.95), (A.Dist("pareto", 1.5), 4, 0.95), (A.Dist("lognormal", 1.0), 4, 0.99), (A.Dist("pareto", 3.0), 8, 0.9)])
def test_fft_matches_monte_carlo_within_error(d, n, q):
    v_fft = A.advantage(d, n, q, A.sum_quantiles_fft(d, [n], [q]).quantiles[(n, q)])
    v_mc, se = A.mc_advantage_with_error(d, n, q, 200_000, 5)
    assert abs(v_mc - v_fft) <= 4.0 * se + 0.01 * v_fft, (d, n, q, v_fft, v_mc, se)


def test_scipy_distributions_agree_with_the_demos_cdf_and_ppf():
    for d, ref in ((A.Dist("t", 3.0), stats.t(3.0)), (A.Dist("pareto", 1.5), stats.pareto(1.5)), (A.Dist("lognormal", 0.8), stats.lognorm(0.8)), (A.Dist("normal"), stats.norm())):
        x = np.array([0.5, 1.2, 2.0, 5.0])
        assert A.cdf(d, x) == pytest.approx(ref.cdf(x), rel=1e-9, abs=1e-12)
        assert A.ppf(d, np.array([0.1, 0.9, 0.99])) == pytest.approx(ref.ppf(np.array([0.1, 0.9, 0.99])), rel=1e-9)
        assert A.mean(d) == pytest.approx(float(ref.mean()), rel=1e-9)


# --- (3) Gleichkorrelation gegen simulierte Normalvektoren ------------------------------------------------------------------------------

@pytest.mark.parametrize("n, rho", [(4, 0.5), (8, 0.2), (3, 0.9)])
def test_equicorrelation_matches_simulated_normal_vectors(n, rho):
    cov = np.full((n, n), rho) + (1.0 - rho) * np.eye(n)
    sample = stats.multivariate_normal(mean=np.zeros(n), cov=cov).rvs(size=400_000, random_state=np.random.default_rng(2024))
    total = sample.sum(axis=1)
    q = 0.95
    v_sim = n * float(stats.norm.ppf(q)) / float(np.quantile(total, q))
    assert A.advantage_normal(n, rho) == pytest.approx(v_sim, rel=0.02)


# --- (4) Quantil gegen numpy.quantile auf der Aufzählung kleiner diskreter Verteilungen (n = 2) -----------------------------------------------------

def test_service_quantile_and_convolution_against_enumeration():
    rng = np.random.default_rng(11)
    for _ in range(60):
        k = int(rng.integers(2, 7))
        counts = rng.integers(1, 5, size=k)                  # Gewichte als Häufigkeiten: die Aufzählung wiederholt jeden Wert so oft
        values = np.sort(rng.choice(np.arange(0, 12), size=k, replace=False)).astype(float)
        probs = counts / counts.sum()
        population = np.repeat(values, counts)
        pairs = (population[:, None] + population[None, :]).ravel()          # alle Paare: exakte Faltung für n = 2
        lattice = np.zeros(int(values.max()) + 1)
        lattice[values.astype(int)] = probs
        two = A.pmf_power_trunc(lattice, 2, 2 * len(lattice))
        nodes = np.arange(len(two), dtype=float)
        for q in (0.05, 0.5, 0.8, 0.9, 0.95, 0.99):
            assert A.service_quantile(values, probs, q) == float(np.quantile(population, q, method="inverted_cdf")), (values, counts, q)
            assert A.service_quantile(nodes, two, q) == float(np.quantile(pairs, q, method="inverted_cdf")), (values, counts, q)
