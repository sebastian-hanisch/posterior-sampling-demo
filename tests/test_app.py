"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden
Regler an seine Grenzen und prüft, dass kein Python-Fehler auftritt (insbesondere StreamlitDuplicateElementId bei st.plotly_chart ohne eindeutiges key=)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
AUTOPLAY_MARKER = "▶"


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
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_world_prior_sampler_and_start_does_not_raise():
    import smp_constants as C

    for world in C.WORLDS:
        for prior in C.PRIORS:
            for sampler in C.SAMPLERS:
                at = _fresh_app()
                at.selectbox(key="world_select").set_value(world)
                at.selectbox(key="prior_select").set_value(prior)
                at.selectbox(key="sampler_select").set_value(sampler)
                at.select_slider(key="steps_select").set_value(1000)
                at.run(timeout=TIMEOUT)
                assert not at.exception, (world, prior, sampler, [str(e) for e in at.exception])
    at = _fresh_app()
    at.selectbox(key="start_select").set_value("prior")
    at.select_slider(key="steps_select").set_value(1000)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_every_noise_level_scale_and_eps_does_not_raise():
    import smp_constants as C

    for level in C.NOISE_CHOICES:
        at = _fresh_app()
        at.select_slider(key="noise_select").set_value(level)
        at.select_slider(key="steps_select").set_value(1000)
        at.run(timeout=TIMEOUT)
        assert not at.exception, (level, [str(e) for e in at.exception])
    for scale in (C.SCALE_CHOICES[0], C.SCALE_CHOICES[-1]):
        for eps in C.EPS_CHOICES:
            at = _fresh_app()
            at.selectbox(key="prior_select").set_value("huber")
            at.select_slider(key="scale_select").set_value(scale)
            at.select_slider(key="eps_select").set_value(eps)
            at.select_slider(key="steps_select").set_value(1000)
            at.run(timeout=TIMEOUT)
            assert not at.exception, (scale, eps, [str(e) for e in at.exception])


def test_the_trace_radio_does_not_raise():
    at = _fresh_app()
    for i in (1, 2, 0):
        at.radio(key="trace_f").set_value(i).run(timeout=TIMEOUT)
        assert not at.exception, (i, [str(e) for e in at.exception])


def test_the_most_singular_corner_does_not_raise_and_warns():
    at = _fresh_app()
    at.slider(key="n_slider").set_value(96)
    at.slider(key="w_slider").set_value(0.05)
    at.select_slider(key="noise_select").set_value(1e-4)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("numerisch singulär" in w.value for w in at.warning)


def test_every_preset_is_applied_to_the_widgets():
    import smp_constants as C

    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.slider(key="n_slider").value == preset["n"]
        assert abs(at.slider(key="w_slider").value - preset["w"]) < 1e-9
        assert at.select_slider(key="noise_select").value == preset["noise"]
        assert at.selectbox(key="world_select").value == preset["world"]
        assert at.selectbox(key="prior_select").value == preset["prior"]
        assert at.selectbox(key="sampler_select").value == preset["sampler"]
        assert at.select_slider(key="steps_select").value == preset["steps"]
        assert at.slider(key="chains_slider").value == preset["chains"]
        assert at.selectbox(key="start_select").value == preset["start"]
        assert at.select_slider(key="scale_select").value == preset["scale"]
        assert at.select_slider(key="eps_select").value == preset["eps"]
        assert at.number_input(key="seed_input").value == preset["seed"]


def test_the_standard_instance_shows_the_success_verdict_and_four_experiments():
    at = _fresh_app()
    assert any("Die Diagnostik ist unauffällig" in s.value for s in at.success)
    headers = [s.value for s in at.subheader]
    assert sum("🔬" in h for h in headers) == 4


def test_the_failing_presets_show_a_warning_and_the_fallback_note():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name in ("pCN bei informativen Daten", "Isotropes RWM", "Start aus dem Prior"):
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert any("Nicht konvergiert" in w.value for w in at.warning), name
    at = _fresh_app()
    at.button[labels["Laplace-Prior"]].click().run(timeout=TIMEOUT)
    at.selectbox(key="sampler_select").set_value("pcn").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("pCN braucht einen Gaußschen Prior" in i.value for i in at.info)


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_permalink_values_are_clipped_and_unknown_choices_ignored():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "999999"
    at.query_params["w"] = "-3"
    at.query_params["r"] = "0.5"
    at.query_params["world"] = "zufall"
    at.query_params["prior"] = "l0"
    at.query_params["sampler"] = "hmc"
    at.query_params["steps"] = "500"
    at.query_params["chains"] = "99"
    at.query_params["start"] = "irgendwo"
    at.query_params["b"] = "0.5"
    at.query_params["eps"] = "0.5"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="n_slider").value == 96
    assert abs(at.slider(key="w_slider").value - 0.01) < 1e-9
    assert at.select_slider(key="noise_select").value == 1e-3
    assert at.selectbox(key="world_select").value == "phantom"
    assert at.selectbox(key="prior_select").value == "gauss"
    assert at.selectbox(key="sampler_select").value == "rwm_cov"
    assert at.select_slider(key="steps_select").value == 10000
    assert at.slider(key="chains_slider").value == 20
    assert at.selectbox(key="start_select").value == "map"
    assert at.select_slider(key="scale_select").value == 0.1
    assert at.select_slider(key="eps_select").value == 0.02


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    at.selectbox(key="sampler_select").set_value("pcn").run(timeout=TIMEOUT)
    assert at.query_params["sampler"] == ["pcn"] or at.query_params["sampler"] == "pcn"
