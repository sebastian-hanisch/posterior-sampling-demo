"""Kernroutinen an der 3 x 3-Mini-Instanz von Hand (DEMO-PLAYBOOK §1b); jede Einheit einzeln."""

import math

import numpy as np
import pytest

import smp_constants as C
import smp_model as m

S3 = np.array([0.0, 10.0, 20.0])


def _target(mini, prior, **kw):
    a, x, b = mini
    return m.Target(a, b, 0.1, prior, S3, tau=1.0, ell=0.12, **kw)


# ----------------------------------------------------------------------------- Ziel, Prior, MAP

def test_exp_kernel_by_hand():
    k = m.exp_kernel(np.array([0.0, 1.0]), 2.0)
    assert k == pytest.approx(np.array([[1.0, math.exp(-0.5)], [math.exp(-0.5), 1.0]]))


def test_gauss_posterior_by_hand(mini):
    a, x, b = mini
    mean, cov = m.gauss_posterior(a, np.eye(3), 1.0, 0.1, b)
    assert mean == pytest.approx([1.153465, 1.653465, 1.153465], abs=2e-6)
    assert cov[0, 0] == pytest.approx(0.105864, abs=2e-6) and cov[1, 1] == pytest.approx(0.336634, abs=2e-6) and cov[0, 2] == pytest.approx(0.067403, abs=2e-6)
    assert sorted(np.linalg.eigvalsh(cov)) == pytest.approx([1.0 / 101.0, 1.0 / 26.0, 0.5])


def test_potentials_by_hand(mini):
    a, x, b = mini
    t = _target(mini, "gauss")
    assert t.phi(x)[0] == pytest.approx(0.0, abs=1e-20) and t.phi(np.zeros(3))[0] == pytest.approx(267.0)
    assert t.neg_log_prior(x)[0] == pytest.approx(3.0, abs=1e-8) and t.neg_log_post(x)[0] == pytest.approx(3.0, abs=1e-8)
    assert _target(mini, "laplace", scale=0.5).neg_log_prior(x)[0] == pytest.approx(8.0)
    assert _target(mini, "huber", scale=0.5, eps=0.1).neg_log_prior(x)[0] == pytest.approx(7.424948, abs=1e-6)
    assert _target(mini, "huber", scale=0.5, eps=0.1).neg_log_prior(np.zeros(3))[0] == 0.0
    both = np.vstack([x, np.zeros(3)])
    assert t.phi(both) == pytest.approx([0.0, 267.0]) and t.neg_log_post(both).shape == (2,)


def test_prior_draws_have_the_right_scale(mini):
    t = _target(mini, "laplace", scale=0.5)
    d = t.prior_draws(100000, 3)
    assert d.shape == (100000, 3) and float(np.mean(np.abs(d))) == pytest.approx(0.5, rel=0.02) and abs(float(d.mean())) < 0.01
    g = _target(mini, "gauss").prior_draws(100000, 4)
    assert float(np.std(g)) == pytest.approx(1.0, rel=0.02)
    assert np.array_equal(t.prior_draws(5, 9), t.prior_draws(5, 9)) and not np.array_equal(t.prior_draws(5, 9), t.prior_draws(5, 10))


def test_map_gauss_is_the_posterior_mean_and_has_zero_gradient(mini):
    t = _target(mini, "gauss")
    x_map = m.map_estimate(t)
    assert x_map == pytest.approx([1.153465, 1.653465, 1.153465], abs=2e-6)
    assert np.linalg.norm(t.ata @ x_map + t.prec @ x_map - t.atb) < 1e-8


def test_lasso_by_hand(mini):
    a, x, b = mini
    assert not m.lasso_active_set(a, b, 1.34 + 1e-9).any()
    w = m.lasso_active_set(a, b, 1.3)
    assert w[0] == 0.0 and w[2] == 0.0 and w[1] == pytest.approx(0.04 / 0.34)
    assert m.lasso_active_set(np.eye(3), b, 0.5) == pytest.approx([0.8, 0.9, 0.8])
    assert m.lasso_active_set(a, b, 1e-12) == pytest.approx(x, abs=1e-6)


def test_map_laplace_is_the_l1_solution_with_lambda_sigma_squared_over_b(mini):
    a, x, b = mini
    t = _target(mini, "laplace", scale=0.1 ** 2 / 1.3)             # lambda = sigma^2 / b = 1,3
    assert m.map_estimate(t) == pytest.approx(m.lasso_active_set(a, b, 1.3))
    assert m.map_estimate(t)[1] == pytest.approx(0.04 / 0.34)


def test_map_huber_has_zero_gradient_and_approaches_l1_for_small_eps(mini):
    t = _target(mini, "huber", scale=0.1 ** 2 / 0.5, eps=1e-6)
    x_map = m.map_estimate(t)
    grad = t.ata @ x_map - t.atb + x_map / np.sqrt(x_map ** 2 + t.eps ** 2) / t.scale
    assert np.linalg.norm(grad) < 1e-6
    assert x_map == pytest.approx(m.lasso_active_set(t.a, t.b, 0.5), abs=1e-3)


def test_hessian_by_hand(mini):
    t = _target(mini, "huber", scale=1.0, eps=1.0)
    assert m.hessian(t, np.zeros(3)) == pytest.approx(t.ata + np.eye(3))                # eps^2 / (0 + eps^2)^1,5 = 1
    assert m.hessian(t, np.array([1.0, 0.0, 0.0]))[0, 0] == pytest.approx(t.ata[0, 0] + 1.0 / 2.0 ** 1.5)
    assert m.hessian(_target(mini, "laplace"), np.zeros(3)) is None
    assert m.hessian(_target(mini, "gauss"), np.zeros(3)) == pytest.approx(_target(mini, "gauss").ata + _target(mini, "gauss").prec)


def test_laplace_approximation_is_exact_for_the_gaussian_posterior_and_missing_for_laplace(mini):
    a, x, b = mini
    t = _target(mini, "gauss")
    mean, cov = m.laplace_approximation(t, m.map_estimate(t))
    mean_ref, cov_ref = m.gauss_posterior(a, np.eye(3), 1.0, 0.1, b)
    assert mean == pytest.approx(mean_ref, abs=1e-8) and cov == pytest.approx(cov_ref, abs=1e-8)
    assert m.laplace_approximation(_target(mini, "laplace"), np.zeros(3)) is None


def test_proposal_covariance_by_hand(mini):
    t = _target(mini, "laplace", scale=0.4)
    assert m.proposal_covariance(t, np.zeros(3)) == pytest.approx(np.linalg.inv(t.ata + np.eye(3) / 0.04))     # (b/2)^2 = 0,04
    g = _target(mini, "gauss")
    assert m.proposal_covariance(g, m.map_estimate(g)) == pytest.approx(m.gauss_posterior(g.a, np.eye(3), 1.0, 0.1, g.b)[1], abs=1e-8)


def test_functionals_by_hand():
    f = m.functionals(np.array([0.1, 0.5, 0.9]))
    assert f.shape == (3, 3) and f[0] == pytest.approx([1 / 3] * 3) and f[1].tolist() == [0.0, 1.0, 0.0] and f[2] == pytest.approx([1 / 3, -1 / 3, 1 / 3])
    assert m.functionals((np.arange(48) + 0.5) / 48)[1].argmax() == 27                # nächstgelegen zu 0,58: (27 + 0,5)/48 = 0,573 (Abstand 0,007) gegen 0,594 (0,014)


# ----------------------------------------------------------------------------- Sampler

def test_samplers_with_zero_step_always_accept_and_do_not_move(mini):
    t = _target(mini, "gauss")
    x0 = np.array([[1.0, 2.0, 1.0], [0.5, 0.5, 0.5]])
    fm = m.functionals(S3)
    for kind in ("pcn", "rwm"):
        r = m.sample(t, kind, x0, 1000, 5, 0.0, None, fm)
        assert r["acc"].tolist() == [1.0, 1.0] and r["count"] == 800
        assert r["sum1"] == pytest.approx(800 * x0) and r["sum2"] == pytest.approx(800 * x0 ** 2)
        assert r["traces"].shape == (1000, 2, 3) and r["traces"][-1] == pytest.approx(x0 @ fm.T)


def test_samplers_with_a_huge_step_almost_never_accept(mini):
    t = _target(mini, "gauss")
    x0 = np.array([[1.15, 1.65, 1.15]])
    assert m.sample(t, "rwm", x0, 2000, 5, 50.0)["acc"][0] < 0.01
    assert m.sample(t, "pcn", x0, 2000, 5, 0.99)["acc"][0] < 0.05


def test_sample_shapes_burn_and_thinning(mini):
    t = _target(mini, "gauss")
    r = m.sample(t, "rwm", np.zeros((3, 3)), 1000, 5, 0.05, None, m.functionals(S3))
    assert r["count"] == 800 and r["thinned"].shape == (400, 3, 3) and r["traces"].shape == (1000, 3, 3) and r["sum1"].shape == (3, 3)
    r2 = m.sample(t, "rwm", np.zeros((2, 3)), 300, 5, 0.05)
    assert r2["count"] == 240 and r2["thinned"].shape == (240, 2, 3) and r2["traces"].shape == (300, 2, 0)


def test_sample_is_deterministic_and_the_seed_matters(mini):
    t = _target(mini, "gauss")
    x0 = np.zeros((2, 3))
    a1 = m.sample(t, "rwm", x0, 500, 7, 0.1)
    a2 = m.sample(t, "rwm", x0, 500, 7, 0.1)
    a3 = m.sample(t, "rwm", x0, 500, 8, 0.1)
    assert np.array_equal(a1["sum1"], a2["sum1"]) and not np.array_equal(a1["sum1"], a3["sum1"])


def test_pcn_acceptance_rule_by_hand_from_the_prior_side(mini):
    """Bei x = (1, 2, 1) (Phi = 0) wird ein Vorschlag mit Phi' > 0 nur mit Wahrscheinlichkeit exp(-Phi') angenommen; mit beta ~ 1 kommen die Vorschläge aus dem Prior, deren Phi' um
    einige hundert liegt: die Annahme ist praktisch null. Bei x = 0 (Phi = 267) wird jeder Vorschlag mit Phi' < 267 sicher angenommen, das sind einige Prozent."""
    t = _target(mini, "gauss")
    hi = m.sample(t, "pcn", np.array([[1.0, 2.0, 1.0]]), 2000, 5, 0.999)
    lo = m.sample(t, "pcn", np.zeros((1, 3)), 200, 5, 0.999)
    assert hi["acc"][0] < 0.005 and lo["acc"][0] > 0.02 and lo["acc"][0] > 5 * hi["acc"][0]


def test_tune_step_picks_the_step_closest_to_the_target_acceptance(mini):
    t = _target(mini, "gauss")
    x0 = np.tile(m.map_estimate(t), (4, 1))
    h, acc = m.tune_step(t, "rwm", x0, 3)
    assert h in C.HS and abs(acc - C.TARGET_ACC_RWM) < 0.1
    h2, acc2 = m.tune_step(t, "pcn", x0, 3)
    assert h2 in C.BETAS and abs(acc2 - C.TARGET_ACC_PCN) < 0.1
    assert m.tune_step(t, "rwm", x0, 3, grid=(1e-6, 100.0))[0] in (1e-6, 100.0)


# ----------------------------------------------------------------------------- Diagnostik und Identifikation

def test_iat_basic_cases():
    assert m.iat(np.ones(100)) == math.inf
    rng = np.random.default_rng(0)
    assert m.iat(rng.normal(size=20000)) == pytest.approx(1.0, abs=0.2)
    assert m.iat(np.array([1.0, -1.0] * 5000)) == 1.0                                  # negative Autokorrelation wird auf 1 begrenzt


def test_rhat_by_hand():
    chains = np.array([[0.0, 2.0, 0.0, 2.0], [1.0, 3.0, 1.0, 3.0]])
    assert m.rhat(chains) == pytest.approx(math.sqrt(1.125))                           # W = 4/3, B = 2
    same = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    assert m.rhat(same) == pytest.approx(math.sqrt(3.0 / 4.0))                         # B = 0: sqrt((n - 1)/n)
    assert m.rhat(np.ones((2, 5))) == math.inf


def test_effective_sample_size_by_hand():
    rng = np.random.default_rng(1)
    tr = rng.normal(size=(5000, 3))
    assert m.effective_sample_size(tr) == pytest.approx(15000.0, rel=0.1)              # iid: N je Kette
    assert m.effective_sample_size(tr[:, :1]) == pytest.approx(5000.0, rel=0.1)


def test_identification_eigenvalues_by_hand(mini):
    a, x, b = mini
    lam = m.identification_eigenvalues(a, np.eye(3), 1.0, 0.2)
    assert lam == pytest.approx([25.0, 6.25, 0.25], rel=1e-9) and int((lam > 1.0).sum()) == 2
    assert m.identification_eigenvalues(a, np.eye(3), 2.0, 0.2) == pytest.approx([100.0, 25.0, 1.0], rel=1e-9)
    assert m.identification_eigenvalues(a, np.eye(3), 1.0, 0.1) == pytest.approx([100.0, 25.0, 1.0], rel=1e-9)


# ----------------------------------------------------------------------------- weitere Einheiten: IAT direkt, Blockrand, Ausdünnung

def test_iat_matches_a_direct_autocorrelation_computation():
    rng = np.random.default_rng(5)
    e = rng.normal(size=3000)
    x = np.zeros(3000)
    for i in range(1, 3000):
        x[i] = 0.8 * x[i - 1] + e[i]
    y = x - x.mean()
    acf = np.correlate(y, y, mode="full")[len(y) - 1:]
    acf = acf / acf[0]
    tau = 1.0
    for k in range(1, len(y)):
        tau = 1.0 + 2.0 * float(np.sum(acf[1:k + 1]))
        if k >= 5.0 * tau:
            break
    assert m.iat(x) == pytest.approx(max(tau, 1.0), rel=1e-9)


def test_sample_handles_steps_that_are_not_a_multiple_of_the_block(mini):
    t = _target(mini, "gauss")
    r = m.sample(t, "rwm", np.zeros((2, 3)), 1300, 5, 0.1, None, m.functionals(S3))
    assert r["traces"].shape == (1300, 2, 3) and r["count"] == 1040 and r["acc"].shape == (2,)
    assert np.isfinite(r["traces"]).all()


def test_thinned_samples_are_the_chain_states_at_burn_plus_multiples_of_the_thinning(mini):
    t = _target(mini, "gauss")
    r = m.sample(t, "rwm", np.zeros((2, 3)), 1000, 5, 0.1, None, np.eye(3))
    burn, thin = 200, 2
    for j in (0, 1, 5, 399):
        assert np.array_equal(r["thinned"][j], r["traces"][burn + j * thin])


def test_thinning_with_an_odd_interval_and_a_burn_that_is_not_a_multiple(mini):
    """steps = 1700: burn = 340, thin = (1700 - 340) // 400 = 3, 340 mod 3 = 1: das j-te gespeicherte x ist der Zustand bei Schritt 340 + 3 j."""
    t = _target(mini, "gauss")
    r = m.sample(t, "rwm", np.zeros((2, 3)), 1700, 5, 0.1, None, np.eye(3))
    assert r["thinned"].shape == (454, 2, 3)
    for j in (0, 1, 2, 100, 453):
        assert np.array_equal(r["thinned"][j], r["traces"][340 + 3 * j])
