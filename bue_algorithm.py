"""Kern der Demo: Sicherheitsbestand einzeln und gebündelt, Bündelungsvorteil V(n) und Wachstumsexponent.

Alle Größen sind in einer **standardisierten** Nachfrage Z gerechnet (Skala 1); die Nachfrage der App ist Mittel + Skala · (Z - E[Z]) mit Skala 20.
Der Vorteil V(n) = n · SB1 / SBn ist von Mittelwert und Skala unabhängig, nur der Sicherheitsbestand in Stück braucht die Skala.

Einheiten (DEMO-PLAYBOOK §1b), jede mit Ein- und Ausgaben und ohne versteckten Zustand:
  Verteilung      `mean`, `cdf`, `ppf`, `partial_mean` (Teilmoment, für die Mittelwert erhaltende Gitterdarstellung)
  Gitter          `lattice_masses` (Massen auf einem Gitter, Mittelwert und Masse erhalten), `convolve_trunc`, `pmf_power_trunc`
                  (n-fache Faltung nicht negativer Verteilungen, exakt unterhalb der Abschneidegrenze), `pmf_power_circular` (vorzeichenbehaftet, FFT)
  Quantile        `lattice_quantile`, `service_quantile` (kleinster Wert mit P(D <= Wert) >= q, die Definition des Servicegrads)
  Zwei Wege       `sum_quantiles_fft` (Faltung auf feinem Gitter) und `mc_sum_samples` (Monte Carlo mit SplitMix64)
  Kennzahlen      `safety_single`, `advantage`, `advantage_normal`, `growth_exponent`, `asymptotic_exponent`

Normal: V = sqrt(n), mit Gleichkorrelation V = sqrt(n) / sqrt(1 + (n - 1) rho) (geschlossen, `advantage_normal`)."""

import math
from dataclasses import dataclass

import numpy as np
from scipy import special as sp

import bue_constants as C
from bue_rng import SplitMix64, uniform_block

KINDS = ("normal", "t", "pareto", "lognormal")


@dataclass(frozen=True)
class Dist:
    """Standardisierte Nachfrage: normal N(0,1); t = Student-t mit param = nu > 1 Freiheitsgraden; pareto = P(Z > x) = x^(-param), x >= 1, param = alpha > 1;
    lognormal = exp(param * N(0,1)), param = sigma > 0."""

    kind: str
    param: float = 0.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unbekannte Verteilung {self.kind}")
        if self.kind in ("t", "pareto") and not self.param > 1:
            raise ValueError("Student-t und Pareto brauchen param > 1 (sonst gibt es keinen Mittelwert)")
        if self.kind == "lognormal" and not self.param > 0:
            raise ValueError("Lognormal braucht sigma > 0")


# --- Verteilung ----------------------------------------------------------------------------------------------------------------------------

def mean(d: Dist) -> float:
    """Erwartungswert der standardisierten Nachfrage."""
    if d.kind == "normal" or d.kind == "t":
        return 0.0
    if d.kind == "pareto":
        return d.param / (d.param - 1.0)
    return math.exp(d.param ** 2 / 2.0)


def cdf(d: Dist, x):
    """Verteilungsfunktion F(x)."""
    x = np.asarray(x, dtype=float)
    if d.kind == "normal":
        return sp.ndtr(x)
    if d.kind == "t":
        return sp.stdtr(d.param, x)
    if d.kind == "pareto":
        return np.where(x >= 1.0, 1.0 - np.power(np.maximum(x, 1.0), -d.param), 0.0)
    pos = x > 0
    return np.where(pos, sp.ndtr(np.log(np.where(pos, x, 1.0)) / d.param), 0.0)


def ppf(d: Dist, u):
    """Quantilfunktion F^-1(u) für u in (0, 1)."""
    u = np.asarray(u, dtype=float)
    if d.kind == "normal":
        return sp.ndtri(u)
    if d.kind == "t":
        return sp.stdtrit(d.param, u)
    if d.kind == "pareto":
        return np.power(1.0 - u, -1.0 / d.param)
    return np.exp(d.param * sp.ndtri(u))


def partial_mean(d: Dist, x):
    """Teilmoment M(x) = Integral von -inf bis x über t f(t) dt (für die Mittelwert erhaltende Gitterdarstellung)."""
    x = np.asarray(x, dtype=float)
    if d.kind == "normal":
        return -np.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)
    if d.kind == "t":
        nu = d.param
        c = math.exp(math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(nu * math.pi)
        return -(nu / (nu - 1.0)) * c * np.power(1.0 + x * x / nu, -(nu - 1.0) / 2.0)
    if d.kind == "pareto":
        a = d.param
        return np.where(x >= 1.0, a / (a - 1.0) * (1.0 - np.power(np.maximum(x, 1.0), 1.0 - a)), 0.0)
    pos = x > 0
    s = d.param
    return np.where(pos, math.exp(s * s / 2.0) * sp.ndtr(np.log(np.where(pos, x, 1.0)) / s - s), 0.0)


def draw(d: Dist, seed: int, count: int) -> np.ndarray:
    """`count` Ziehungen der standardisierten Nachfrage per Inversion aus dem SplitMix64-Strom `seed` (kein numpy.random)."""
    u = np.maximum(uniform_block(seed, count), 2.0 ** -53)
    return ppf(d, u)


# --- Gitter und Faltung --------------------------------------------------------------------------------------------------------------------

def lattice_masses(d: Dist, h: float, j_first: int, j_count: int) -> np.ndarray:
    """Massen auf den Knoten j*h, j = j_first .. j_first + j_count (j_count + 1 Werte).

    Die Masse jeder Zelle [j h, (j+1) h) wird so auf ihre beiden Randknoten verteilt, dass Masse UND Mittelwert der Zelle erhalten bleiben
    (lokale Momentanpassung). Masse außerhalb der Randkanten fehlt (der Aufrufer weiß, wie viel das ist)."""
    edges = np.arange(j_first, j_first + j_count + 1, dtype=float) * h
    mass = np.diff(cdf(d, edges))
    moment = np.diff(partial_mean(d, edges))
    right = (moment - edges[:-1] * mass) / h          # Anteil der Zellmasse, der auf den rechten Knoten geht
    out = np.zeros(j_count + 1)
    out[:-1] += mass - right
    out[1:] += right
    return out


def _fft_size(m: int) -> int:
    return 1 << (m - 1).bit_length()


def convolve_trunc(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Faltung zweier Massenvektoren auf dem Gitter 0, h, 2h ..., abgeschnitten auf die ersten k Knoten (Summen unterhalb k*h sind exakt)."""
    size = _fft_size(len(a) + len(b) - 1)
    out = np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b, size), size)[:k]
    return np.maximum(out, 0.0)


def pmf_power_trunc(p: np.ndarray, n: int, k: int) -> np.ndarray:
    """n-fache Faltung eines Massenvektors auf 0, h, ..., (k-1) h (nicht negative Werte) durch wiederholtes Quadrieren; Masse oberhalb k*h fällt weg."""
    if n < 1:
        raise ValueError("n muss mindestens 1 sein")
    result = None
    base = p[:k]
    while n:
        if n & 1:
            result = base if result is None else convolve_trunc(result, base, k)
        n >>= 1
        if n:
            base = convolve_trunc(base, base, k)
    return result


def pmf_power_circular(p_ordered: np.ndarray, n: int) -> np.ndarray:
    """n-fache zyklische Faltung für vorzeichenbehaftete Verteilungen. `p_ordered` hat M Werte für die Knoten j = -M/2 .. M/2 - 1 in dieser Reihenfolge;
    das Ergebnis liegt in derselben Reihenfolge. Summen außerhalb von ±(M/2) h laufen um (Randfehler, siehe `sum_quantiles_fft`)."""
    m = len(p_ordered)
    p = np.roll(p_ordered, m // 2)
    s = np.fft.irfft(np.fft.rfft(p) ** n, m)
    return np.roll(s, -(m // 2))


def lattice_quantile(pmf: np.ndarray, h: float, q: float, first_node: int = 0) -> float:
    """q-Quantil einer Gitterverteilung: Knoten j steht für die Zelle [x_j - h/2, x_j + h/2], darin wird linear interpoliert. nan, wenn die Masse nicht bis q reicht."""
    cum = np.cumsum(pmf)
    idx = int(np.searchsorted(cum, q, side="left"))
    if idx >= len(cum):
        return float("nan")
    prev = cum[idx - 1] if idx > 0 else 0.0
    frac = (q - prev) / pmf[idx] if pmf[idx] > 0 else 0.5
    return (first_node + idx) * h - h / 2.0 + min(max(frac, 0.0), 1.0) * h


def service_quantile(values, probs, q: float) -> float:
    """Servicegrad-Quantil einer diskreten Verteilung: der kleinste Wert x mit P(D <= x) >= q (gleich numpy.quantile(..., method='inverted_cdf'))."""
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(values, kind="stable")
    cum = np.cumsum(probs[order])
    idx = int(np.searchsorted(cum, q * cum[-1] - 1e-12, side="left"))
    return float(values[order][min(idx, len(values) - 1)])


# --- Weg 1: Faltung -----------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FFTResult:
    quantiles: dict          # {(n, q): Quantil der Summe von n Nachfragen, standardisiert}; nan, wenn das Gitter nicht reicht
    h: float                 # Gitterweite
    limit: float             # Abschneidegrenze L (nicht negativ: Summen unterhalb L sind exakt; vorzeichenbehaftet: Gitter reicht von -L bis L)
    lost: float              # größte verlorene Masse über alle n (nicht negativ: P(Summe >= L); vorzeichenbehaftet: Schranke n P(|Z| > L) für den Randfehler)
    reliable: bool           # Verlust höchstens Toleranz mal (1 - q) für das größte q: sonst reicht das Gitter nicht für die Quantile


def _limit_guess(d: Dist, n: int, q: float) -> float:
    """Grobe obere Schranke für das q-Quantil der Summe: Mittel plus Einzelquantil bei Tail-Wahrscheinlichkeit (1-q)/n plus Streuungsteil (Großsprung-Näherung)."""
    jump = float(ppf(d, 1.0 - (1.0 - q) / n))
    return n * abs(mean(d)) + abs(jump) + 6.0 * math.sqrt(n) * (1.0 + abs(mean(d)))


def sum_quantiles_fft(d: Dist, ns, qs, h_target: float = C.FFT_H_TARGET, points_max: int = C.FFT_POINTS_MAX) -> FFTResult:
    """Quantile der Summe von n unabhängigen Nachfragen durch n-fache Faltung der Gitterverteilung (FFT). Ein Gitter für alle n und q.

    Nicht negative Verteilungen (Pareto, Lognormal): Abschneiden bei L, Summen unterhalb L sind exakt (alle Summanden >= 0); L wird verdoppelt, bis
    die verlorene Masse höchstens die Hälfte von 1-q beträgt. Student-t und Normal: zyklisch auf [-L, L); die Mittelwert erhaltende Gitterdarstellung
    gilt für jede Gitterweite, der Rest ist der Randfehler (siehe `lost`)."""
    ns = sorted(set(int(n) for n in ns))
    qs = tuple(qs)
    n_max, q_max = ns[-1], max(qs)
    x_est = _limit_guess(d, n_max, q_max)
    out = {}
    if d.kind in ("pareto", "lognormal"):
        limit = 2.0 * x_est
        for _ in range(6):
            k = min(points_max, _fft_size(int(limit / h_target) + 1))
            h = limit / k
            p = lattice_masses(d, h, 0, k)[:k]
            powers = {n: pmf_power_trunc(p, n, k) for n in ns}
            lost = max(1.0 - float(np.sum(pw)) for pw in powers.values())
            if lost <= C.FFT_TAIL_TOLERANCE_EXACT * (1.0 - q_max):
                break
            limit *= 2.0
        for n, pw in powers.items():
            for q in qs:
                out[(n, q)] = lattice_quantile(pw, h, q)
        return FFTResult(out, h, limit, lost, lost <= C.FFT_TAIL_TOLERANCE_EXACT * (1.0 - q_max))
    limit = 4.0 * x_est
    points_max = max(points_max, C.FFT_POINTS_MAX_SIGNED)
    for _ in range(5):
        m = min(points_max, _fft_size(int(2.0 * limit / h_target) + 1))
        h = 2.0 * limit / m
        edge = n_max * float(cdf(d, -limit) + 1.0 - cdf(d, limit))
        if edge <= C.FFT_TAIL_TOLERANCE * (1.0 - q_max):
            break
        limit *= 2.0
    half = m // 2
    p = lattice_masses(d, h, -half, m)
    p[0] += p[-1]
    p = p[:m]
    base = np.fft.rfft(np.roll(p, half))
    for n in ns:
        s = np.roll(np.fft.irfft(base ** n, m), -half)
        for q in qs:
            out[(n, q)] = lattice_quantile(s, h, q, first_node=-half)
    return FFTResult(out, h, limit, edge, edge <= C.FFT_TAIL_TOLERANCE * (1.0 - q_max))


# --- Weg 2: Monte Carlo -------------------------------------------------------------------------------------------------------------------

def mc_sum_samples(d: Dist, n: int, n_draws: int, seed: int) -> np.ndarray:
    """`n_draws` Summen von je n unabhängigen Nachfragen. Jede der n Spalten ist ein eigener SplitMix64-Strom (Startwerte aus einem Strom des Seeds)."""
    return mc_sum_chain(d, [n], n_draws, seed)[n]


def mc_sum_chain(d: Dist, ns, n_draws: int, seed: int) -> dict:
    """Summen für mehrere n aus derselben Kette von Spalten: die Summe für n besteht aus den ersten n Spalten ({n: Stichprobe}); spart Rechenzeit in Tabellen."""
    ns = sorted(set(int(n) for n in ns))
    starts = SplitMix64(seed)
    total = np.zeros(n_draws)
    out = {}
    for column in range(1, ns[-1] + 1):
        total += draw(d, starts.next(), n_draws)
        if column in ns:
            out[column] = total.copy()
    return out


def mc_quantile(sample: np.ndarray, q: float) -> float:
    """Quantil einer Stichprobe mit derselben Definition wie `service_quantile` (inverted_cdf)."""
    return float(np.quantile(sample, q, method="inverted_cdf"))


def mc_advantage_with_error(d: Dist, n: int, q: float, n_draws: int, seed: int, batches: int = C.MC_BATCHES) -> tuple[float, float]:
    """(V, Standardfehler) per Monte Carlo: V aus allen Ziehungen, der Standardfehler aus der Streuung von V über `batches` gleich große Teilstichproben."""
    sample = mc_sum_samples(d, n, n_draws, seed)
    v_all = advantage(d, n, q, mc_quantile(sample, q))
    parts = np.array_split(sample, batches)
    vs = np.array([advantage(d, n, q, mc_quantile(part, q)) for part in parts])
    vs = vs[np.isfinite(vs)]
    se = float(np.std(vs, ddof=1) / math.sqrt(len(vs))) if len(vs) > 1 else float("nan")
    return v_all, se


# --- Kennzahlen ---------------------------------------------------------------------------------------------------------------------------

def safety_single(d: Dist, q: float) -> float:
    """Sicherheitsbestand eines einzelnen Standorts: q-Quantil minus Mittelwert (standardisiert)."""
    return float(ppf(d, q)) - mean(d)


def safety_pooled(d: Dist, n: int, q: float, quantile_of_sum: float) -> float:
    """Sicherheitsbestand des gemeinsamen Lagers: Quantil der Summe minus n mal Mittelwert (standardisiert)."""
    return quantile_of_sum - n * mean(d)


def advantage(d: Dist, n: int, q: float, quantile_of_sum: float) -> float:
    """Bündelungsvorteil V = n SB1 / SBn. nan, wenn der gebündelte Sicherheitsbestand nicht positiv ist (Quantil der Summe nicht über dem Mittelwert)."""
    sb_n = safety_pooled(d, n, q, quantile_of_sum)
    if not (sb_n > 0) or not math.isfinite(sb_n):
        return float("nan")
    return n * safety_single(d, q) / sb_n


def advantage_normal(n: int, rho: float = 0.0) -> float:
    """Normalverteilte Nachfrage mit Gleichkorrelation rho: V = sqrt(n) / sqrt(1 + (n - 1) rho) (geschlossen, unabhängig vom Quantil)."""
    return math.sqrt(n) / math.sqrt(1.0 + (n - 1) * rho)


def growth_exponent(v1: float, n1: int, v2: float, n2: int) -> float:
    """Steigung von log V gegen log n zwischen n1 < n2."""
    if n2 <= n1:
        raise ValueError("n2 muss größer als n1 sein")
    return math.log(v2 / v1) / math.log(n2 / n1)


def asymptotic_exponent(d: Dist) -> float:
    """Exponent von V(n) für n gegen unendlich: bei Tail-Index a in (1, 2) (Pareto-Exponent alpha, Freiheitsgrade nu) gilt (a-1)/a für die Summe
    stabiler Nachfrage; bei endlicher Varianz 1/2 (Normal, Lognormal, Pareto/Student-t mit Index >= 2, hier ohne Beweis und nur als Grenzwert)."""
    if d.kind in ("pareto", "t") and d.param < 2.0:
        return (d.param - 1.0) / d.param
    return 0.5


def variance_limit_factor(d: Dist, q: float) -> float:
    """Bei endlicher Varianz ist V(n) für große n etwa sqrt(n) mal SB1 / (z_q sigma) mit z_q dem Normal-Quantil (Zentraler Grenzwertsatz). nan bei unendlicher Varianz."""
    if d.kind == "normal":
        return 1.0
    if d.kind == "t":
        if d.param <= 2.0:
            return float("nan")
        sigma = math.sqrt(d.param / (d.param - 2.0))
    elif d.kind == "pareto":
        if d.param <= 2.0:
            return float("nan")
        a = d.param
        sigma = math.sqrt(a / ((a - 1.0) ** 2 * (a - 2.0)))
    else:
        s2 = d.param ** 2
        sigma = math.sqrt((math.exp(s2) - 1.0) * math.exp(s2))
    return safety_single(d, q) / (float(sp.ndtri(q)) * sigma)
