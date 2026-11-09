"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text; je Preset die erwartete Verdikt-Zeile."""

import math

import pytest

import bue_algorithm as A
import bue_constants as C
import bue_evaluation as ev


def _a(v, n=16, kind="t", reliable=True, rho=0.0, sbn=None):
    d = A.Dist("normal") if kind == "normal" else A.Dist(kind, 3.0)
    sqrt_n = A.advantage_normal(n, rho) if kind == "normal" else math.sqrt(n)
    return ev.Analysis(d, n, 0.95, rho, v, 10.0, sbn if sbn is not None else (n * 10.0 / v if v and math.isfinite(v) else float("nan")), sqrt_n, reliable, 0.5)


def test_success_within_five_percent_of_sqrt_n_inclusive():
    kind, text = ev.verdict_kernfrage(_a(4.0))
    assert kind == "success" and "**V = 4,00**" in text and "√n = 4,00" in text and "höchstens 5 %" in text
    assert ev.verdict_kernfrage(_a(4.0 * 0.95))[0] == "success"
    assert ev.verdict_kernfrage(_a(4.0 * 1.05))[0] == "success"
    assert ev.verdict_kernfrage(_a(4.0 * 0.949))[0] == "warning"
    assert ev.verdict_kernfrage(_a(4.0 * 1.051))[0] == "info"


def test_below_and_above_sqrt_text_names_percent_of_sqrt():
    kind, text = ev.verdict_kernfrage(_a(2.0))
    assert kind == "warning" and "weniger als die Wurzel-Regel" in text and "50 % davon" in text and "Einzelbestände zusammen 160,0 Stück" in text and "gemeinsamer Bestand 80,0 Stück" in text
    kind, text = ev.verdict_kernfrage(_a(5.0))
    assert kind == "info" and "mehr als die Wurzel-Regel" in text and "125 % davon" in text


def test_advantage_below_one_is_a_warning_about_the_definition():
    kind, text = ev.verdict_kernfrage(_a(0.8))
    assert kind == "warning" and "kleiner als 1" in text and "Quantil minus Mittelwert" in text
    assert ev.verdict_kernfrage(_a(1.0))[0] == "warning"       # 1,0 liegt weit unter sqrt(16); kein Sonderzweig für genau 1


def test_undefined_and_unreliable_cases_show_no_number():
    kind, text = ev.verdict_kernfrage(_a(float("nan"), sbn=-1.0))
    assert kind == "warning" and "nicht definiert" in text and "V = " not in text
    kind, text = ev.verdict_kernfrage(_a(3.0, reliable=False))
    assert kind == "warning" and "Gitter" in text and "V = " not in text


def test_correlation_verdict_names_closed_form_and_sqrt():
    a = _a(A.advantage_normal(8, 0.5), n=8, kind="normal", rho=0.5)
    kind, text = ev.verdict_kernfrage(a)
    assert kind == "info" and "**V = 1,33**" in text and "Gleichkorrelation 0,50" in text and "√n = 2,83" in text


def _rows(vs):
    return [{"label": f"X{i}", "v": {8: v, 32: v * 2}, "dist": A.Dist("normal")} for i, v in enumerate(vs)]


def test_exp1_counts_distributions_more_than_five_percent_below_sqrt():
    s8 = math.sqrt(8)
    kind, text = ev.verdict_exp1(_rows([s8, s8 * 0.96, s8 * 0.94]), 8)
    assert kind == "warning" and "bei 1 von 3 Verteilungen" in text and "X2" in text and "94 % von √n" in text
    assert ev.verdict_exp1(_rows([s8, s8 * 0.96]), 8)[0] == "success"


def test_exp2_orders_ascending():
    kind, text = ev.verdict_exp2(_rows([3.0, 1.0, 2.0]), 8, 5)
    assert kind == "info" and text.index("X1 (1,00)") < text.index("X2 (2,00)") < text.index("X0 (3,00)") and "5 Läufe" in text


def test_exp3_counts_rising_flat_and_falling():
    rows = [{"label": "a", "v": {90.0: 2.0, 99.9: 3.0}}, {"label": "b", "v": {90.0: 2.0, 99.9: 2.0}}, {"label": "c", "v": {90.0: 2.0, 99.9: 1.5}}]
    kind, text = ev.verdict_exp3(rows, qs=(90.0, 99.9))
    assert kind == "info" and "bei 1 von 3 Verteilungen, bleibt gleich bei 1 (Normal), sinkt bei 1" in text


def test_exp4_text_has_closed_form_numbers_and_limit():
    kind, text = ev.verdict_exp4(8, 0.5)
    assert "V = 1,33" in text and "√n = 2,83" in text and "1/√ρ = 1,41" in text
    assert "Ohne Korrelation ist V = √n = 2,83" in ev.verdict_exp4(8, 0.0)[1]


def _ex(e_high, asym, q=0.95, kind="pareto", param=1.5):
    return {"dist": A.Dist(kind, param), "label": ev.dist_label(A.Dist(kind, param)), "q": q, "e_high": e_high, "asymptote": asym}


def test_exp5_counts_cells_within_tolerance_inclusive_and_names_the_worst():
    kind, text = ev.verdict_exp5([_ex(0.30, 0.35), _ex(0.20, 0.35, q=0.99)])
    assert kind == "info" and "in 1 von 2 Zellen" in text and "0,05" in text and "Servicegrad 99,0 %" in text and "0,200 statt 0,350" in text
    kind, text = ev.verdict_exp5([_ex(0.30, 0.35)])
    assert kind == "success" and "in 1 von 1 Zellen" in text
    assert "Keine Verteilung" in ev.verdict_exp5([_ex(0.5, 0.5, kind="pareto", param=3.0)])[1]


@pytest.mark.parametrize("name, expected", [
    ("Normal, 8 Standorte", "success"),
    ("Schwerer Tail (Pareto 1,5), 32 Standorte", "warning"),
    ("Student-t mit 3 Freiheitsgraden", "warning"),
    ("Hoher Servicegrad 99 %", "info"),
    ("Gleichkorrelation 0,5", "info"),
])
def test_every_preset_runs_through_with_the_expected_verdict(name, expected):
    a = ev.analyse(ev.Settings(**{k: v for k, v in C.PRESETS[name].items()}))
    kind, text = ev.verdict_kernfrage(a)
    assert kind == expected and a.reliable, text


def test_presets_differ_in_their_advantage():
    vs = {name: ev.analyse(ev.Settings(**p)).v for name, p in C.PRESETS.items()}
    assert len({round(v, 3) for v in vs.values()}) == len(vs)
