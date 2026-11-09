"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            if slider.disabled:
                continue          # ein Nutzer kann einen ausgegrauten Regler nicht bedienen; die Kombinationen unten decken ihn ab
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_selectbox_and_select_slider_option_does_not_raise():
    at0 = _fresh_app()
    for key, options in (("kind_select", list(at0.selectbox(key="kind_select").options)), ("q_select", list(at0.select_slider(key="q_select").options))):
        for option in options:
            at = _fresh_app()
            widget = at.selectbox(key=key) if key == "kind_select" else at.select_slider(key=key)
            widget.set_value(widget.options[options.index(option)] if key == "kind_select" else option).run(timeout=TIMEOUT)
            assert not at.exception, (key, option, [str(e) for e in at.exception])


def test_all_minima_and_all_maxima_at_once_do_not_raise():
    for kind in ("normal", "t", "pareto", "lognormal"):
        for edge in ("min", "max"):
            at = _fresh_app()
            at.selectbox(key="kind_select").set_value(kind).run(timeout=TIMEOUT)
            for s in at.slider:
                if not s.disabled:
                    s.set_value(s.min if edge == "min" else s.max)
            at.run(timeout=TIMEOUT)
            assert not at.exception, (kind, edge, [str(e) for e in at.exception])


def test_permalink_is_clamped_and_snapped():
    at = AppTest.from_file(APP_PATH)
    at.query_params["n"] = "100000"
    at.query_params["q"] = "97.3"
    at.query_params["kind"] = "pareto"
    at.query_params["alpha"] = "0.1"
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert at.slider(key="n_slider").value == 256 and at.select_slider(key="q_select").value == 99.0
    assert at.selectbox(key="kind_select").value == "pareto" and at.slider(key="alpha_slider").value == 1.2


def test_monte_carlo_button_shows_a_comparison_with_the_convolution():
    at = _fresh_app()
    at.button[[b.label for b in at.button].index(next(b.label for b in at.button if b.label.startswith("🎲 Monte Carlo")))].click().run(timeout=TIMEOUT)
    assert not at.exception
    assert any("Monte Carlo: V = " in i.value and "Faltung: V = " in i.value for i in at.info)


def test_disabled_controls_follow_the_distribution():
    at = _fresh_app()
    assert at.slider(key="nu_slider").disabled and at.slider(key="alpha_slider").disabled and not at.slider(key="rho_slider").disabled
    at.selectbox(key="kind_select").set_value("t").run(timeout=TIMEOUT)
    assert not at.slider(key="nu_slider").disabled and at.slider(key="rho_slider").disabled


def test_unreliable_grid_shows_no_number_in_the_metrics(monkeypatch):
    import dataclasses

    import bue_evaluation as ev
    import streamlit as st

    real = ev.analyse
    monkeypatch.setattr(ev, "analyse", lambda s: dataclasses.replace(real(s), reliable=False))
    st.cache_data.clear()  # sonst liefert der Cache früherer AppTests die echte Analyse
    try:
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=TIMEOUT)
    finally:
        st.cache_data.clear()
    assert not at.exception, [str(e) for e in at.exception]
    values = {m.label: m.value for m in at.metric}
    assert values["Bündelungsvorteil V"] == "–" and values["Gemeinsames Lager"] == "–"
