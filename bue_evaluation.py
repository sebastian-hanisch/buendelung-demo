"""Kennzahlen, Experimente und Verdikte (reine Funktionen). Jedes Verdikt ist `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import bue_algorithm as A
import bue_constants as C

MC_TABLE_PATH = Path(__file__).resolve().parent / "bue_mc_tabelle.json"


def fmt(x: float, digits: int = 2) -> str:
    """Zahl mit Dezimalkomma (Text der App und der Verdikte)."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    kind: str = C.DEFAULT_KIND
    nu: float = C.DEFAULT_NU
    alpha: float = C.DEFAULT_ALPHA
    sigma: float = C.DEFAULT_SIGMA
    q: float = C.DEFAULT_Q          # Servicegrad in Prozent
    n: int = C.DEFAULT_N
    rho: float = C.DEFAULT_RHO
    seed: int = C.DEFAULT_SEED


def dist_of(s: Settings) -> A.Dist:
    """Die Verteilung der Einstellung; der Parameter ist je nach Auswahl nu, alpha oder sigma."""
    if s.kind == "normal":
        return A.Dist("normal")
    if s.kind == "t":
        return A.Dist("t", float(s.nu))
    if s.kind == "pareto":
        return A.Dist("pareto", float(s.alpha))
    return A.Dist("lognormal", float(s.sigma))


def dist_label(d: A.Dist) -> str:
    """Kurzname mit Parameter, z. B. „Pareto α = 1,5“."""
    if d.kind == "normal":
        return "Normal"
    sym = {"t": "ν", "pareto": "α", "lognormal": "σ"}[d.kind]
    return f"{C.KIND_LABELS[d.kind]} {sym} = {fmt(d.param, 1)}"


@dataclass(frozen=True)
class Analysis:
    dist: A.Dist
    n: int
    q: float                 # Servicegrad als Anteil
    rho: float               # wirksam nur bei Normal
    v: float                 # Bündelungsvorteil n SB1 / SBn (nan, wenn SBn nicht positiv)
    sb1: float               # Sicherheitsbestand eines Standorts in Stück (Skala 20)
    sbn: float               # Sicherheitsbestand des gemeinsamen Lagers in Stück
    sqrt_n: float            # Wurzel-Regel (bei Normal mit Korrelation: die geschlossene Formel)
    reliable: bool           # Faltungsgitter reicht (immer wahr bei Normal)
    asymptote: float         # Wachstumsexponent für n gegen unendlich

    @property
    def total_single(self) -> float:
        return self.n * self.sb1

    @property
    def saving(self) -> float:
        """Eingesparter Sicherheitsbestand in Stück: Summe der Einzelbestände minus gemeinsamer Bestand."""
        return self.total_single - self.sbn

    @property
    def ratio(self) -> float:
        """V gegen die Wurzel-Regel sqrt(n)."""
        return self.v / math.sqrt(self.n)


def analyse(s: Settings) -> Analysis:
    d, q, n = dist_of(s), s.q / 100.0, int(s.n)
    sb1_std = A.safety_single(d, q)
    if d.kind == "normal":
        v = A.advantage_normal(n, s.rho)
        rho, reliable = s.rho, True
    else:
        res = A.sum_quantiles_fft(d, [n], [q])
        v = A.advantage(d, n, q, res.quantiles[(n, q)])
        rho, reliable = 0.0, res.reliable
    sbn_std = n * sb1_std / v if math.isfinite(v) and v > 0 else float("nan")
    if not math.isfinite(v) and d.kind != "normal":
        sbn_std = A.safety_pooled(d, n, q, res.quantiles[(n, q)])
    return Analysis(d, n, q, rho, v, C.SCALE * sb1_std, C.SCALE * sbn_std, A.advantage_normal(n, rho) if d.kind == "normal" else math.sqrt(n),
                    reliable, A.asymptotic_exponent(d))


# --- Experimente --------------------------------------------------------------------------------------------------------------------------

def advantage_curve(d: A.Dist, q: float, ns=C.N_SWEEP) -> dict:
    """V(n) für alle n der Liste aus einer Faltung (ein Gitter für alle n); {n: V} und das Gitter-Zeugnis."""
    res = A.sum_quantiles_fft(d, ns, [q])
    return {"v": {n: A.advantage(d, n, q, res.quantiles[(n, q)]) for n in ns}, "reliable": res.reliable}


def sweep_distributions(q: float, ns=C.N_SWEEP, dists=None) -> list[dict]:
    """Experiment 1 und 2: V(n) je Verteilung (Faltung)."""
    rows = []
    for d in dists or default_dists():
        c = advantage_curve(d, q, ns)
        rows.append({"dist": d, "label": dist_label(d), "v": c["v"], "reliable": c["reliable"]})
    return rows


def default_dists() -> list[A.Dist]:
    """Die Verteilungen der Vergleichs-Experimente (feste Liste, damit die Zählungen in den Tests und im Text übereinstimmen)."""
    return [A.Dist("normal"), A.Dist("t", 10.0), A.Dist("t", 3.0), A.Dist("lognormal", 0.5), A.Dist("lognormal", 1.0),
            A.Dist("pareto", 3.0), A.Dist("pareto", 2.0), A.Dist("pareto", 1.5)]


def service_matrix(n: int, dists=None, qs=C.Q_OPTIONS, ns_curve=None) -> list[dict]:
    """Experiment 3: V je Verteilung und Servicegrad bei festem n."""
    rows = []
    for d in dists or default_dists():
        res = A.sum_quantiles_fft(d, [n], [x / 100.0 for x in qs])
        rows.append({"dist": d, "label": dist_label(d), "v": {x: A.advantage(d, n, x / 100.0, res.quantiles[(n, x / 100.0)]) for x in qs}, "reliable": res.reliable})
    return rows


def correlation_table(ns=(2, 8, 32, 256), rhos=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0)) -> list[dict]:
    """Experiment 4: V gegen Gleichkorrelation (Normal, geschlossen); Grenzwert für n gegen unendlich: 1/sqrt(rho)."""
    return [{"n": n, "v": {r: A.advantage_normal(n, r) for r in rhos}} for n in ns]


EXPONENT_NS = (16, 64, 256)


def exponent_table(dists, qs=(0.95, 0.99, 0.999)) -> list[dict]:
    """Experiment 5: gemessene Steigung zwischen n = 64 und n = 256 (und zwischen 16 und 64) gegen den asymptotischen Exponenten, je Verteilung und Servicegrad."""
    rows = []
    for d in dists:
        res = A.sum_quantiles_fft(d, EXPONENT_NS, list(qs))
        for q in qs:
            vs = {n: A.advantage(d, n, q, res.quantiles[(n, q)]) for n in EXPONENT_NS}
            ok = all(math.isfinite(v) and v > 0 for v in vs.values())
            rows.append({
                "dist": d, "label": dist_label(d), "q": q, "v": vs, "asymptote": A.asymptotic_exponent(d), "reliable": res.reliable,
                "e_low": A.growth_exponent(vs[16], 16, vs[64], 64) if ok else float("nan"),
                "e_high": A.growth_exponent(vs[64], 64, vs[256], 256) if ok else float("nan"),
                "level": vs[256] / (256 ** A.asymptotic_exponent(d)) if ok else float("nan"),   # V(256) gegen die Potenzfunktion n^e durch V(1) = 1
                "factor": vs[256] / 16.0 if ok else float("nan"),                                # V(256) / sqrt(256)
                "limit_factor": A.variance_limit_factor(d, q),
            })
    return rows


def exponent_dists() -> list[A.Dist]:
    return [A.Dist("pareto", 1.2), A.Dist("pareto", 1.5), A.Dist("pareto", 1.8), A.Dist("pareto", 3.0), A.Dist("t", 1.5), A.Dist("t", 3.0), A.Dist("lognormal", 1.0)]


def load_mc_table() -> dict:
    """Vorgerechnete Monte-Carlo-Tabelle (tools/make_mc_table.py): {"draws", "seeds", "cells": {"kind|param|q|n": [V je Seed]}}."""
    with open(MC_TABLE_PATH, encoding="utf-8") as f:
        return json.load(f)


def mc_key(d: A.Dist, q: float, n: int) -> str:
    return f"{d.kind}|{d.param:g}|{q:g}|{n}"


def mc_band(table: dict, d: A.Dist, q: float, n: int):
    """(Minimum, Median, Maximum) von V über die Seeds der Tabelle; None, wenn die Zelle fehlt."""
    vals = table["cells"].get(mc_key(d, q, n))
    if not vals:
        return None
    arr = np.array(vals, dtype=float)
    return float(arr.min()), float(np.median(arr)), float(arr.max())


# --- Verdikte -----------------------------------------------------------------------------------------------------------------------------

RATIO_BAND = 0.05   # |V / sqrt(n) - 1| höchstens so viel: „wie die Wurzel-Regel“


def verdict_kernfrage(a: Analysis) -> tuple[str, str]:
    """Kernfrage: Wie viel Sicherheitsbestand spart die Allianz? Vergleich von V mit der Wurzel-Regel."""
    name = dist_label(a.dist)
    head = f"{name}, {a.n} Standorte, Servicegrad {fmt(a.q * 100, 1)} %: "
    if not a.reliable:
        return "warning", head + "das Faltungsgitter reicht für dieses Quantil nicht aus (Tail zu schwer für den Gitterbereich); die Zahl wird nicht als Ergebnis gezeigt."
    if not math.isfinite(a.v):
        return "warning", head + "das Quantil der Summe liegt nicht über dem Mittelwert, der gemeinsame Sicherheitsbestand ist nicht positiv; V ist hier nicht definiert."
    sentence = f"Einzelbestände zusammen {fmt(a.total_single, 1)} Stück, gemeinsamer Bestand {fmt(a.sbn, 1)} Stück, also **V = {fmt(a.v)}**"
    if a.dist.kind == "normal" and a.rho > 0:
        return "info", head + f"{sentence} bei Gleichkorrelation {fmt(a.rho)}; ohne Korrelation wäre V = √n = {fmt(math.sqrt(a.n))}."
    if a.v < 1.0:
        return "warning", head + f"{sentence}: kleiner als 1, das gemeinsame Lager braucht nach dieser Definition (Quantil minus Mittelwert) mehr Sicherheitsbestand als alle Einzellager zusammen."
    if abs(a.ratio - 1.0) <= RATIO_BAND + 1e-9:
        return "success", head + f"{sentence}, das ist die Wurzel-Regel √n = {fmt(a.sqrt_n)} (Abweichung höchstens {int(RATIO_BAND * 100)} %)."
    if a.ratio > 1.0:
        return "info", head + f"{sentence}, mehr als die Wurzel-Regel √n = {fmt(a.sqrt_n)} ({fmt(a.ratio * 100, 0)} % davon)."
    return "warning", head + f"{sentence}, weniger als die Wurzel-Regel √n = {fmt(a.sqrt_n)} ({fmt(a.ratio * 100, 0)} % davon)."


def verdict_exp1(rows: list[dict], n: int) -> tuple[str, str]:
    """Experiment 1: bei welchen Verteilungen liegt V(n) unter der Wurzel-Regel (mehr als RATIO_BAND darunter)?"""
    below = [r for r in rows if r["v"][n] < (1.0 - RATIO_BAND) * math.sqrt(n)]
    worst = min(rows, key=lambda r: r["v"][n])
    text = (f"Bei n = {n} liegt V bei {len(below)} von {len(rows)} Verteilungen mehr als {int(RATIO_BAND * 100)} % unter √n = {fmt(math.sqrt(n))}; "
            f"am niedrigsten bei {worst['label']} mit V = {fmt(worst['v'][n])} ({fmt(worst['v'][n] / math.sqrt(n) * 100, 0)} % von √n).")
    return ("warning" if below else "success"), text


def verdict_exp2(rows: list[dict], n: int, mc_runs: int) -> tuple[str, str]:
    """Experiment 2: Rangfolge der Verteilungen bei festem n und Servicegrad (aufsteigend nach V)."""
    ordered = sorted(rows, key=lambda r: r["v"][n])
    ranking = " < ".join(f"{r['label']} ({fmt(r['v'][n])})" for r in ordered)
    return "info", f"Rangfolge von V bei n = {n} (Faltung, ein Lauf je Verteilung; Monte-Carlo-Gegenprobe {mc_runs} Läufe je Zelle in der Tabelle): {ranking}."


def verdict_exp3(rows: list[dict], qs=C.Q_OPTIONS) -> tuple[str, str]:
    """Experiment 3: Zählung, bei wie vielen Verteilungen V mit dem Servicegrad wächst (Servicegrad höchster gegen niedrigster)."""
    lo, hi = qs[0], qs[-1]
    rising = [r for r in rows if r["v"][hi] > r["v"][lo] * (1.0 + 1e-6)]
    flat = [r for r in rows if abs(r["v"][hi] / r["v"][lo] - 1.0) <= 1e-6]
    text = (f"V steigt vom Servicegrad {fmt(lo, 1)} % zu {fmt(hi, 1)} % bei {len(rising)} von {len(rows)} Verteilungen, bleibt gleich bei {len(flat)} (Normal), "
            f"sinkt bei {len(rows) - len(rising) - len(flat)}.")
    return "info", text


def verdict_exp4(n: int, rho: float) -> tuple[str, str]:
    """Experiment 4: Normal, Gleichkorrelation: V, Wurzel-Regel und Grenzwert 1/sqrt(rho)."""
    v = A.advantage_normal(n, rho)
    if rho <= 0:
        return "info", f"Ohne Korrelation ist V = √n = {fmt(v)} bei n = {n}."
    return "info", (f"Bei n = {n} und ρ = {fmt(rho)} ist V = {fmt(v)} statt √n = {fmt(math.sqrt(n))}; "
                    f"für beliebig viele Standorte wächst V höchstens bis 1/√ρ = {fmt(1.0 / math.sqrt(rho))}.")


def verdict_exp5(rows: list[dict], tol: float = 0.05) -> tuple[str, str]:
    """Experiment 5: in wie vielen Zellen mit Tail-Index unter 2 liegt die gemessene Steigung (n = 64 bis 256) höchstens `tol` unter dem asymptotischen Exponenten (a - 1) / a?"""
    heavy = [r for r in rows if r["dist"].kind in ("pareto", "t") and r["dist"].param < 2.0 and math.isfinite(r["e_high"])]
    if not heavy:
        return "info", "Keine Verteilung mit Tail-Index unter 2 in der Auswahl."
    near = [r for r in heavy if abs(r["e_high"] - r["asymptote"]) <= tol]
    far = [r for r in heavy if abs(r["e_high"] - r["asymptote"]) > tol]
    text = (f"Bei Tail-Index unter 2 liegt die gemessene Steigung zwischen n = 64 und n = 256 in {len(near)} von {len(heavy)} Zellen höchstens {fmt(tol)} neben dem Exponenten (a − 1)/a")
    if far:
        worst = max(far, key=lambda r: abs(r["e_high"] - r["asymptote"]))
        text += f"; am weitesten entfernt: {worst['label']} bei Servicegrad {fmt(worst['q'] * 100, 1)} % ({fmt(worst['e_high'], 3)} statt {fmt(worst['asymptote'], 3)})."
    else:
        text += "."
    return ("success" if not far else "info"), text
