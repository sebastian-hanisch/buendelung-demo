"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzt und rastet ein."""

import bue_constants as C
import bue_evaluation as ev
import bue_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert p["kind"] in C.KIND_ORDER and p["q"] in C.Q_OPTIONS, name
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.NU_MIN <= p["nu"] <= C.NU_MAX and C.ALPHA_MIN <= p["alpha"] <= C.ALPHA_MAX, name
        assert C.SIGMA_MIN <= p["sigma"] <= C.SIGMA_MAX and C.RHO_MIN <= p["rho"] <= C.RHO_MAX and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Normal, 8 Standorte"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_values_are_clamped_snapped_and_rejected():
    spec = P.SETTING_SPECS
    assert P.clean_value(spec["n_slider"], "100000") == C.N_MAX and P.clean_value(spec["n_slider"], "-5") == C.N_MIN
    assert P.clean_value(spec["alpha_slider"], "0.2") == C.ALPHA_MIN and P.clean_value(spec["rho_slider"], "7") == C.RHO_MAX
    assert P.clean_value(spec["q_select"], "97.3") == 99.0 and P.clean_value(spec["q_select"], "99.8") == 99.9
    assert P.clean_value(spec["kind_select"], "pareto") == "pareto" and P.clean_value(spec["kind_select"], "cauchy") is None
    assert P.clean_value(spec["n_slider"], "abc") is None and P.clean_value(spec["alpha_slider"], "nan") is None
