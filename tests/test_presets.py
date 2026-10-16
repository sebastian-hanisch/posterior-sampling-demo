"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Reglergrenzen."""

import pytest

import smp_constants as C
import smp_evaluation as ev
import smp_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds_and_on_the_step_grid():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0, name
        assert C.W_MIN <= p["w"] <= C.W_MAX and (p["w"] - C.W_MIN) / C.W_STEP == pytest.approx(round((p["w"] - C.W_MIN) / C.W_STEP)), name
        assert p["noise"] in C.NOISE_CHOICES and p["world"] in C.WORLDS and p["prior"] in C.PRIORS and p["sampler"] in C.SAMPLERS and p["start"] in C.STARTS, name
        assert p["steps"] in C.STEPS_CHOICES and C.CHAINS_MIN <= p["chains"] <= C.CHAINS_MAX and p["scale"] in C.SCALE_CHOICES and p["eps"] in C.EPS_CHOICES, name
        assert 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_default_is_a_valid_widget_value():
    for key, spec in P.SETTING_SPECS.items():
        if spec.choices is not None:
            assert spec.default in spec.choices, key
        else:
            assert spec.lo <= spec.default <= spec.hi, key


def test_the_presets_show_the_documented_outcomes():
    res = {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}
    std, pcn, iso, prior_start, lap, hub = (res[k] for k in C.PRESETS)
    assert ev.verdict_instance(std)[0] == "success" and max(std.rhats) < 1.05 and std.sd_err < 0.2
    assert pcn.sampler_used == "pcn" and pcn.sd_err > 1.0 and max(pcn.rhats) > 100 and ev.verdict_instance(pcn)[0] == "warning"
    assert iso.sampler_used == "rwm_iso" and iso.acc < 0.2 and max(iso.rhats) > 30 and ev.verdict_instance(iso)[0] == "warning"
    assert prior_start.settings.start == "prior" and max(prior_start.rhats) > 10 and ev.verdict_instance(prior_start)[0] == "warning"
    assert lap.ref_kind == "" and lap.nnz_map < lap.nnz_mean and lap.err_map < lap.err_mean and ev.verdict_instance(lap)[0] == "success"
    assert hub.ref_kind == "Laplace-Näherung" and hub.sd_ratio_median < 0.5 and "Laplace-Näherung" in ev.verdict_instance(hub)[1]


def test_experiment_constants_are_consistent():
    assert C.EXP_STEPS >= 20000 and C.NG_CHAINS >= 20 and C.RHAT_OK == 1.1 and C.ESS_OK >= 100 and C.EXP_M == 24 and C.EXP_DIMS[0] == C.EXP_M
    assert C.BETAS[0] < 1e-4 and C.BETAS[-1] > 0.5 and C.HS[0] <= 1e-4 and len(C.BETAS) == len(C.HS) == 14 and C.THIN_TARGET > 0 and 0 < C.BURN_FRACTION < 0.5
