"""Auswertung: Formatierung, Welten, Instanz, Verdikte (Art und Text, Schwellen an handgesetzten Werten), Form der Experimente und ihre Reproduzierbarkeit."""

import dataclasses
import math

import numpy as np
import pytest

import smp_constants as C
import smp_evaluation as ev


# ----------------------------------------------------------------------------- Formatierung

def test_de_sci_share_and_noise_label():
    assert ev.de(-1.234, 2) == "−1,23" and ev.sci(2.5e9) == "2,5·10⁹" and ev.sci(3.0e-12) == "3,0·10⁻¹²" and ev.share(0.1525) == "15 %"
    assert ev.sci(0.0) == "0,0" and ev.sci(float("inf")) == "unendlich" and ev.sci(9.96) == "1,0·10¹"
    assert [ev.noise_label(v) for v in C.NOISE_CHOICES] == ["0,01 %", "0,1 %", "1 %", "3 %", "10 %"]


def test_err_pct_digits_and_boundaries():
    assert [ev.err_pct(v) for v in (2.58, 0.1188, 0.0399, 0.0087, 8.7e-4, 4471.0, 0.57)] == ["258 %", "11,9 %", "4,0 %", "0,87 %", "0,09 %", "4,5·10³-fach", "57,0 %"]
    assert ev.err_pct(10.0) == "1,0·10¹-fach" and ev.err_pct(9.99) == "999 %" and ev.err_pct(0.995) == "100 %" and ev.err_pct(0.994) == "99,4 %"
    assert ev.err_pct(0.0095) == "0,9 %" and ev.err_pct(0.0094) == "0,94 %" and ev.err_pct(0.0001) == "0,01 %" and ev.err_pct(0.00009) == "9,0·10⁻³ %"


def test_num_text_iat_text_and_step_text_boundaries():
    assert ev.num_text(1000.0) == "1,0·10³" and ev.num_text(999.99) == "1000,0" and ev.num_text(0.01) == "0,01" and ev.num_text(0.0099) == "9,9·10⁻³"
    assert ev.num_text(10.0) == "10,0" and ev.num_text(9.99) == "9,99" and ev.num_text(0.0) == "0,00" and ev.num_text(float("inf")) == "unendlich"
    assert ev.iat_text(float("inf")) == "steht" and ev.iat_text(1234.0) == "1.234" and ev.iat_text(5.0) == "5"
    assert ev.step_text(0.034) == "0,034" and ev.step_text(0.01) == "0,010" and ev.step_text(0.0031) == "3,1·10⁻³"


# ----------------------------------------------------------------------------- Welten und Instanz

def test_world_truths_and_instance_are_consistent():
    x, g = ev.world_truth("spitzen", 48, 0.03)
    assert (x > 0).sum() == 3 and x.sum() / 48 == pytest.approx(0.12)
    x, g = ev.world_truth("phantom", 48, 0.03)
    assert x.shape == g.shape == (48,)
    a, s, x, b, sigma = ev.instance("phantom", 48, 0.03, 1e-3, 5)
    assert a.shape == (48, 48) and sigma == pytest.approx(1e-3 * float(np.sqrt(np.mean(g ** 2)))) and np.array_equal(b, ev.instance("phantom", 48, 0.03, 1e-3, 5)[3])
    assert not np.array_equal(b, ev.instance("phantom", 48, 0.03, 1e-3, 6)[3])


def test_the_default_settings_are_the_standard_preset():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_analysis_fields_for_the_gaussian_prior():
    a = ev.analyse(ev.Settings(steps=3000))
    assert a.sampler_used == "rwm_cov" and not a.fallback and a.ref_kind == "exakt" and a.burn == 600
    assert a.traces.shape[1:] == (4, 3) and a.trace_steps[0] == 0 and a.thinned.shape[1:] == (4, 48) and a.mean.shape == a.sd.shape == a.lo.shape == a.hi.shape == (48,)
    assert np.all(a.lo <= a.mean) and np.all(a.mean <= a.hi) and len(a.iats) == len(a.ess) == len(a.rhats) == 3
    assert a.step == pytest.approx(2.38 / math.sqrt(48)) and 0.1 < a.acc < 0.5 and a.err_map == pytest.approx(a.err_mean, abs=0.05)


def test_pcn_falls_back_for_a_non_gaussian_prior_and_keeps_the_choice_for_the_gaussian_one():
    a = ev.analyse(ev.Settings(prior="laplace", world="spitzen", sampler="pcn", steps=1000, noise=1e-2, scale=0.03))
    assert a.fallback and a.sampler_used == "rwm_cov" and a.ref_kind == "" and math.isnan(a.mean_err_sd) and a.ref_mean is None
    b = ev.analyse(ev.Settings(sampler="pcn", steps=1000))
    assert not b.fallback and b.sampler_used == "pcn"


def test_reference_kinds_by_prior():
    assert ev.analyse(ev.Settings(prior="huber", world="spitzen", steps=1000, noise=3e-2, scale=0.03, eps=0.002)).ref_kind == "Laplace-Näherung"
    assert ev.analyse(ev.Settings(prior="gauss", steps=1000)).ref_kind == "exakt"


def test_analysis_is_deterministic_and_the_seed_matters():
    a, b, c = ev.analyse(ev.Settings(steps=1000)), ev.analyse(ev.Settings(steps=1000)), ev.analyse(ev.Settings(steps=1000, seed=12))
    assert np.array_equal(a.mean, b.mean) and a.rhats == b.rhats and not np.array_equal(a.mean, c.mean)


def test_prior_start_differs_from_the_map_start():
    a, b = ev.analyse(ev.Settings(steps=1000)), ev.analyse(ev.Settings(steps=1000, start="prior"))
    assert max(b.rhats) > 5 * max(a.rhats)


def test_the_singular_flag_follows_the_condition_number():
    assert ev.analyse(ev.Settings(n=96, w=0.05, steps=1000)).singular and not ev.analyse(ev.Settings(steps=1000)).singular


def test_small_grids_and_all_samplers_are_finite():
    for n in (12, 24):
        for prior in C.PRIORS:
            for sampler in C.SAMPLERS:
                a = ev.analyse(ev.Settings(n=n, prior=prior, sampler=sampler, steps=1000, chains=4, world="spitzen", noise=1e-2, scale=0.1))
                assert np.isfinite(a.mean).all() and np.isfinite(a.sd).all() and all(np.isfinite(v) or v == math.inf for v in a.rhats)


# ----------------------------------------------------------------------------- Verdikte

def _with(**kw):
    a = ev.analyse(ev.Settings(steps=1000))
    base = dict(rhats=[1.0, 1.0, 1.0], ess=[500.0, 500.0, 500.0], mean_err_sd=0.1, sd_err=0.05)
    base.update(kw)
    return dataclasses.replace(a, **base)


def test_verdict_thresholds_at_the_exact_boundaries():
    assert ev.verdict_instance(_with())[0] == "success" and "gegen den exakten Posterior" in ev.verdict_instance(_with())[1]
    assert ev.verdict_instance(_with(rhats=[1.0, 1.1, 1.0]))[0] == "success" and ev.verdict_instance(_with(rhats=[1.0, 1.1001, 1.0]))[0] == "warning"
    assert ev.verdict_instance(_with(ess=[500.0, 100.0, 500.0]))[0] == "success" and ev.verdict_instance(_with(ess=[500.0, 99.9, 500.0]))[0] == "warning"
    assert ev.verdict_instance(_with(sd_err=0.2))[0] == "success" and ev.verdict_instance(_with(sd_err=0.21))[0] == "warning"
    assert ev.verdict_instance(_with(mean_err_sd=0.5))[0] == "success" and ev.verdict_instance(_with(mean_err_sd=0.51))[0] == "warning"
    kind, text = ev.verdict_instance(_with(rhats=[2.0, 1.0, 1.0], sd_err=1.0, mean_err_sd=3.0))
    assert kind == "warning" and "R̂ = 2,00" in text and "100 % falsch" in text and "3,00 sd falsch" in text


def test_verdict_for_a_non_exact_reference_ignores_the_oracle_errors():
    a = dataclasses.replace(ev.analyse(ev.Settings(steps=1000)), ref_kind="Laplace-Näherung", sd_err=9.0, mean_err_sd=9.0, sd_ratio_median=0.4, rhats=[1.0, 1.0, 1.0], ess=[500.0] * 3)
    kind, text = ev.verdict_instance(a)
    assert kind == "success" and "0,40-Fache" in text and "Laplace-Näherung" in text
    b = dataclasses.replace(a, ref_kind="")
    assert "Laplace" not in ev.verdict_instance(b)[1] and "exakten Posterior" not in ev.verdict_instance(b)[1]
    assert "exakten Posterior" not in ev.verdict_instance(a)[1]


# ----------------------------------------------------------------------------- Experimente: Form und Reproduzierbarkeit

def test_sampler_experiment_structure_and_reproducibility(monkeypatch):
    monkeypatch.setattr(C, "EXP_DIMS", (24, 48))
    r1, r2 = ev.sampler_experiment(steps=2000), ev.sampler_experiment(steps=2000)
    assert len(r1["info"]) == len(C.EXP_NOISE) * len(C.SAMPLERS) and [d["n"] for d in r1["dims"]] == [24, 48]
    assert r1["info"][0]["acc"] == r2["info"][0]["acc"] and r1["dims"][1]["fixed_pcn_acc"] == r2["dims"][1]["fixed_pcn_acc"]
    assert all(len(r["iats"]) == 3 for r in r1["info"])


def test_burnin_experiment_structure():
    r = ev.burnin_experiment(steps=5000)
    assert set(r["rows"]) == {"map", "prior"} and [x["discard"] for x in r["rows"]["map"]] == list(C.BURN_DISCARD)
    assert r["rows"]["prior"][0]["rhat_mean"] > 5 * r["rows"]["map"][0]["rhat_mean"]


def test_identification_experiment_structure():
    r = ev.identification_experiment()
    assert [x["key"] for x in r["rows"]] == ["ou", "se", "white"] and all(len(x["counts"]) == len(C.ID_NOISE) for x in r["rows"])
    assert all(c1 <= c2 for x in r["rows"] for c1, c2 in zip(x["counts"], x["counts"][1:]))


def test_nongauss_experiment_structure(monkeypatch):
    monkeypatch.setattr(C, "NG_LAPLACE", ((1e-2, 0.1),))
    monkeypatch.setattr(C, "NG_HUBER", ((0.02, 0.1, 3e-2),))
    r = ev.nongauss_experiment(chains=4)
    assert len(r["laplace"]) == 1 and len(r["huber"]) == 1 and r["laplace"][0]["nnz_mean"] >= r["laplace"][0]["nnz_map"] - 2 and r["huber"][0]["ratio_min"] <= r["huber"][0]["ratio_max"]


# ----------------------------------------------------------------------------- Tikhonov-Maßstab und Anzeigeraster

def test_tikhonov_best_matches_an_independent_normal_equation_route(monkeypatch):
    a, s, x, b, sigma = ev.instance("spitzen", 24, 0.03, 3e-2, 31)
    xn = float(np.linalg.norm(x))
    ref = min(float(np.linalg.norm(np.linalg.solve(a.T @ a + lam ** 2 * np.eye(24), a.T @ b) - x)) / xn for lam in np.logspace(-6, 0, 61))
    assert ev._tikhonov_best(a, b, x) == pytest.approx(ref, rel=1e-6)


def test_the_displayed_traces_are_thinned_to_about_1500_points():
    a = ev.analyse(ev.Settings(steps=3000))
    assert a.traces.shape[0] == 1500 and a.trace_steps[1] - a.trace_steps[0] == 2
    b = ev.analyse(ev.Settings(steps=1000))
    assert b.traces.shape[0] == 1000 and b.trace_steps[1] - b.trace_steps[0] == 1


def test_iats_are_the_mean_over_all_chains_and_the_trace_stride_follows_floor_division():
    a = ev.analyse(ev.Settings(steps=1000, chains=4))
    import smp_model as m

    expected = [float(np.mean([m.iat(a.traces[a.burn:, c, f]) for c in range(4)])) for f in range(3)]
    assert a.iats == pytest.approx(expected, rel=1e-9)
    assert ev.analyse(ev.Settings(steps=2999)).traces.shape[0] == 2999
