"""Aussagen aus README, Verdikten und Hilfetexten, einzeln an den festen Strömen gemessen (Seeds ab SEED_BASE). Ändert sich eine Zahl, muss der Text mit."""

import math

import numpy as np
import pytest

import smp_constants as C
import smp_evaluation as ev


@pytest.fixture(scope="module")
def sampler():
    return ev.sampler_experiment()


@pytest.fixture(scope="module")
def burnin():
    return ev.burnin_experiment()


@pytest.fixture(scope="module")
def ident():
    return ev.identification_experiment()


@pytest.fixture(scope="module")
def nongauss():
    return ev.nongauss_experiment()


# ----------------------------------------------------------------------------- Instanz und Presets

def test_standard_preset_numbers():
    a = ev.analyse(ev.Settings(**C.PRESETS["Standard"]))
    assert a.sampler_used == "rwm_cov" and a.step == pytest.approx(2.38 / math.sqrt(48)) and a.acc == pytest.approx(0.24, abs=0.03)
    assert max(a.rhats) < 1.05 and min(a.ess) > 150.0 and a.mean_err_sd == pytest.approx(0.22, abs=0.1) and a.sd_err == pytest.approx(0.11, abs=0.06)
    assert a.err_map == pytest.approx(0.136, abs=0.01) and a.err_mean == pytest.approx(0.135, abs=0.01) and 1.2e4 < a.cond < 1.3e4


def test_pcn_and_isotropic_presets_show_the_standing_chain():
    pcn = ev.analyse(ev.Settings(**C.PRESETS["pCN bei informativen Daten"]))
    assert pcn.step == pytest.approx(3.1e-4, rel=0.05) and pcn.acc == pytest.approx(0.22, abs=0.03) and max(pcn.rhats) > 100.0
    assert pcn.mean_err_sd == pytest.approx(4.0, abs=0.7) and pcn.sd_err == pytest.approx(2.7, abs=0.5)
    iso = ev.analyse(ev.Settings(**C.PRESETS["Isotropes RWM"]))
    assert iso.step == pytest.approx(3.7e-4, rel=0.05) and iso.acc == pytest.approx(0.12, abs=0.03) and 30.0 < max(iso.rhats) < 200.0
    assert iso.mean_err_sd == pytest.approx(4.0, abs=0.7) and iso.sd_err == pytest.approx(2.6, abs=0.5)


def test_prior_start_preset_numbers():
    a = ev.analyse(ev.Settings(**C.PRESETS["Start aus dem Prior"]))
    assert max(a.rhats) == pytest.approx(38.6, rel=0.2) and a.mean_err_sd == pytest.approx(21.6, rel=0.3) and ev.verdict_instance(a)[0] == "warning"


def test_laplace_and_huber_preset_numbers():
    lap = ev.analyse(ev.Settings(**C.PRESETS["Laplace-Prior"]))
    assert max(lap.rhats) < 1.1 and min(lap.ess) > 100.0 and lap.err_map == pytest.approx(0.0274, abs=0.003) and lap.err_mean == pytest.approx(0.0544, abs=0.006)
    assert lap.nnz_map == 24 and 34 <= lap.nnz_mean <= 42
    hub = ev.analyse(ev.Settings(**C.PRESETS["Dünner Pseudo-Huber-Prior"]))
    assert hub.sd_ratio_median == pytest.approx(0.38, abs=0.03) and hub.err_map == pytest.approx(0.0386, abs=0.004) and hub.err_mean == pytest.approx(0.0902, abs=0.01)
    assert hub.sd_err == pytest.approx(3.7, abs=0.6) and max(hub.rhats) < 1.1


# ----------------------------------------------------------------------------- Experiment 1

def test_information_numbers(sampler):
    rows = {(r["noise"], r["sampler"]): r for r in sampler["info"]}
    for noise, beta in ((0.1, 0.030), (0.01, 0.0031), (0.001, 0.00031)):
        assert rows[(noise, "pcn")]["step"] == pytest.approx(beta, rel=0.1) and rows[(noise, "pcn")]["acc"] == pytest.approx(0.23, abs=0.04)
    for noise, h in ((0.1, 0.0189), (0.01, 0.00265), (0.001, 0.00037)):
        assert rows[(noise, "rwm_iso")]["step"] == pytest.approx(h, rel=0.1)
    assert rows[(0.001, "pcn")]["sd_err"] > 0.9 and rows[(0.01, "pcn")]["sd_err"] > 0.9 and rows[(0.001, "rwm_iso")]["sd_err"] > 0.9
    for noise in C.EXP_NOISE:
        r = rows[(noise, "rwm_cov")]
        assert r["acc"] == pytest.approx(0.24, abs=0.03) and max(r["iats"]) < 400 and min(r["iats"]) > 90 and r["sd_err"] < 0.15 and r["mean_err"] < 0.25
    assert rows[(0.001, "pcn")]["iats"][1] > 1000 and rows[(0.001, "rwm_cov")]["iats"][1] < 120 and rows[(0.1, "pcn")]["iats"][0] < 40


def test_dimension_numbers(sampler):
    dims = {d["n"]: d for d in sampler["dims"]}
    for d in dims.values():
        assert d["pcn"]["step"] == pytest.approx(0.03, rel=0.05) and 0.36 <= d["fixed_pcn_acc"] <= 0.42 and d["pcn"]["iats"][0] < 40
    assert dims[24]["fixed_rwm_acc"] == pytest.approx(0.12, abs=0.03) and dims[192]["fixed_rwm_acc"] < 0.01 and dims[384]["fixed_rwm_acc"] < 0.01
    assert dims[384]["rwm"]["step"] < dims[48]["rwm"]["step"] / 3.0 and dims[384]["rwm"]["iats"][0] > 20 * dims[24]["rwm"]["iats"][0] / 2.0
    assert dims[384]["rwm"]["iats"][0] > 800 and dims[384]["rwm"]["step"] == pytest.approx(0.0098, rel=0.1)


# ----------------------------------------------------------------------------- Experiment 2

def test_burnin_numbers(burnin):
    m, p = burnin["rows"]["map"], burnin["rows"]["prior"]
    assert m[0]["rhat_point"] == pytest.approx(1.0075, abs=0.02) and m[0]["rhat_mean"] == pytest.approx(1.005, abs=0.02) and all(r["rhat_point"] < 1.1 for r in m)
    assert p[0]["rhat_point"] == pytest.approx(2.19, rel=0.15) and p[0]["rhat_mean"] == pytest.approx(22.4, rel=0.15)
    assert p[-1]["rhat_point"] == pytest.approx(3.49, rel=0.15) and p[-1]["rhat_mean"] == pytest.approx(43.5, rel=0.15)
    assert all(r["rhat_mean"] > 10.0 for r in p) and p[-1]["rhat_point"] > p[0]["rhat_point"]


# ----------------------------------------------------------------------------- Experiment 3

def test_identification_numbers(ident):
    rows = {r["key"]: r for r in ident["rows"]}
    assert rows["ou"]["counts"] == [21, 30, 38, 44] and rows["se"]["counts"] == [19, 24, 29, 33] and rows["white"]["counts"] == [25, 34, 41, 48]
    assert rows["ou"]["sums"][0] == pytest.approx(20.9, abs=0.2) and rows["ou"]["sums"][3] == pytest.approx(44.2, abs=0.2) and rows["white"]["sums"][3] == pytest.approx(46.9, abs=0.2)


# ----------------------------------------------------------------------------- Experiment 4

def test_laplace_prior_numbers(nongauss):
    lap = {(r["noise"], r["scale"]): r for r in nongauss["laplace"]}
    a, b, c = lap[(1e-2, 0.1)], lap[(3e-2, 0.03)], lap[(3e-2, 0.1)]
    assert (a["map"], a["mean"], a["tik"]) == pytest.approx((0.0136, 0.0144, 0.0152), abs=0.0015)
    assert (b["map"], b["mean"], b["tik"]) == pytest.approx((0.0214, 0.0296, 0.0457), abs=0.003)
    assert (c["map"], c["mean"], c["tik"]) == pytest.approx((0.0331, 0.0385, 0.0457), abs=0.004)
    assert b["nnz_map"] == 14 and b["nnz_mean"] >= 21 and all(r["map"] <= r["mean"] + 1e-3 for r in lap.values()) and all(0.2 < r["acc"] < 0.4 for r in lap.values())


def test_laplace_approximation_numbers(nongauss):
    h = nongauss["huber"]
    assert h[0]["ratio_median"] == pytest.approx(0.98, abs=0.03) and h[0]["ratio_min"] > 0.9 and h[0]["gap_sd"] < 0.15
    assert h[1]["ratio_median"] == pytest.approx(0.85, abs=0.05) and h[1]["ratio_min"] == pytest.approx(0.50, abs=0.06)
    assert h[2]["ratio_median"] == pytest.approx(0.46, abs=0.05) and h[3]["ratio_median"] == pytest.approx(0.37, abs=0.04) and h[3]["gap_sd"] == pytest.approx(0.85, abs=0.1)
    assert all(r["rhat"] < 1.1 for r in h) and all(r["map"] < r["mean"] for r in h[1:])
    assert [r["map"] for r in h] == pytest.approx([0.0387, 0.0218, 0.0575, 0.1299], abs=0.003) and [r["mean"] for r in h] == pytest.approx([0.0407, 0.0292, 0.0838, 0.1474], abs=0.004)
    assert [r["rhat"] for r in h] == pytest.approx([1.003, 1.010, 1.029, 1.012], abs=0.012)


def test_verdict_texts_carry_the_numbers(sampler, burnin, ident, nongauss):
    s = ev.verdict_sampler(sampler)[1]
    assert "3,1·10⁻⁴" in s and "0,030" in s and "0,00" in s and "0,39" in s
    b = ev.verdict_burnin(burnin)[1]
    assert "R̂ = 1,01" in b and "2,19 / 22,4" in b and "8000 Schritten" in b
    i = ev.verdict_identification(ident)[1]
    assert "21 / 30 / 38 / 44 von 48" in i
    n = ev.verdict_nongauss(nongauss)[1]
    assert "MAP 2,1 %" in n and "Posterior-Mittel 3,0 %" in n and "Tikhonov 4,6 %" in n and "0,37" in n and "0,98" in n
