"""Orakel (DEMO-PLAYBOOK §12): Sampler gegen den exakten Gauß-Posterior (Mittel, Kovarianz, Randverteilungen), das exakte Gibbs-Verfahren (Laplace-Prior), AR(1)-Ketten, die
Reversibilität der Annahmeregeln und die numerische Integration in einer Dimension. Jeder Test unter etwa 10 s.

Fehlt scipy im Test-venv, überspringen sich die scipy-Tests (pytest.importorskip), sie raten nie."""

import math

import numpy as np
import pytest

import smp_evaluation as ev
import smp_model as m
from inverse_generator import SplitMix64


def _gauss_instance(n=12, noise=1e-1, seed=5):
    a, s, x, b, sigma = ev.instance("phantom", n, 0.03, noise, seed)
    return m.Target(a, b, sigma, "gauss", s), a, b, sigma


def test_preconditioned_rwm_reproduces_the_exact_gaussian_posterior():
    t, a, b, sigma = _gauss_instance()
    mean, cov = m.gauss_posterior(a, t.c, t.tau, sigma, b)
    chol = np.linalg.cholesky(cov + 1e-14 * np.eye(12))
    x0 = mean[None, :] + 2.0 * (SplitMix64(3).normal(4 * 12).reshape(4, 12) @ chol.T)
    r = m.sample(t, "rwm", x0, 40000, 5, 2.38 / math.sqrt(12), chol, m.functionals(t.s))
    total = r["count"] * 4
    mc = r["sum1"].sum(axis=0) / total
    sd = np.sqrt(r["sum2"].sum(axis=0) / total - mc ** 2)
    ref = np.sqrt(np.diag(cov))
    assert np.max(np.abs(mc - mean) / ref) < 0.2 and np.max(np.abs(sd / ref - 1.0)) < 0.1
    assert float(np.mean(r["acc"])) == pytest.approx(0.3, abs=0.12)
    samples = r["thinned"].reshape(-1, 12)
    assert np.cov(samples.T) == pytest.approx(cov, rel=0.15, abs=0.15 * float(np.abs(np.diag(cov)).max()))


def test_pcn_reproduces_the_exact_gaussian_posterior_for_weak_data():
    t, a, b, sigma = _gauss_instance(noise=3e-1)
    mean, cov = m.gauss_posterior(a, t.c, t.tau, sigma, b)
    x0 = np.tile(mean, (4, 1))
    beta, acc = m.tune_step(t, "pcn", x0, 3)
    r = m.sample(t, "pcn", x0, 60000, 7, beta)
    total = r["count"] * 4
    mc = r["sum1"].sum(axis=0) / total
    sd = np.sqrt(r["sum2"].sum(axis=0) / total - mc ** 2)
    ref = np.sqrt(np.diag(cov))
    assert np.max(np.abs(mc - mean) / ref) < 0.25 and np.max(np.abs(sd / ref - 1.0)) < 0.15 and 0.1 < acc < 0.5


def test_marginals_match_the_exact_normal_distributions_in_the_kolmogorov_distance():
    scipy = pytest.importorskip("scipy")
    from scipy import stats

    t, a, b, sigma = _gauss_instance()
    mean, cov = m.gauss_posterior(a, t.c, t.tau, sigma, b)
    chol = np.linalg.cholesky(cov + 1e-14 * np.eye(12))
    r = m.sample(t, "rwm", np.tile(mean, (4, 1)), 40000, 11, 2.38 / math.sqrt(12), chol)
    pooled = r["thinned"].reshape(-1, 12)                                    # 1.600 ausgedünnte Werte
    ref = np.sqrt(np.diag(cov))
    for i in (0, 5, 11):
        assert stats.kstest((pooled[:, i] - mean[i]) / ref[i], "norm").statistic < 0.08


def test_identical_seeds_and_starts_give_identical_chains_and_different_starts_do_not():
    t, a, b, sigma = _gauss_instance()
    mean, cov = m.gauss_posterior(a, t.c, t.tau, sigma, b)
    x0 = np.tile(mean, (2, 1))
    r1 = m.sample(t, "rwm", x0, 2000, 3, 0.05)
    r2 = m.sample(t, "rwm", x0, 2000, 3, 0.05)
    r3 = m.sample(t, "rwm", x0 + 0.01, 2000, 3, 0.05)
    assert np.array_equal(r1["thinned"], r2["thinned"]) and not np.array_equal(r1["thinned"], r3["thinned"])


def test_laplace_prior_mean_matches_the_exact_gibbs_sampler():
    scipy = pytest.importorskip("scipy")
    from scipy import special, stats

    n = 6
    a, s, x, b, sigma = ev.instance("spitzen", n, 0.03, 3e-2, 31)
    scale = 0.1
    t = m.Target(a, b, sigma, "laplace", s, scale=scale)
    x_map = m.map_estimate(t)
    chol = np.linalg.cholesky(m.proposal_covariance(t, x_map) + 1e-14 * np.eye(n))
    r = m.sample(t, "rwm", np.tile(x_map, (8, 1)), 20000, 5, 2.38 / math.sqrt(n), chol)
    mean_rwm = r["sum1"].sum(axis=0) / (r["count"] * 8)
    # exaktes Gibbs-Verfahren: je Koordinate eine Mischung aus zwei gestutzten Normalverteilungen
    ata, atb = t.ata, t.atb
    rng = SplitMix64(7)
    sweeps = 4000
    u1 = rng.uniform(sweeps * n).reshape(sweeps, n)
    u2 = rng.uniform(sweeps * n).reshape(sweeps, n)
    cur = x_map.copy()
    draws = []
    for it in range(sweeps):
        for i in range(n):
            p = ata[i, i]
            mu = (atb[i] - ata[i] @ cur + p * cur[i]) / p
            sd = 1.0 / math.sqrt(p)
            mp, mm = mu - 1.0 / (scale * p), mu + 1.0 / (scale * p)
            lw_p = 0.5 * p * (mp ** 2 - mu ** 2) + special.log_ndtr(mp * math.sqrt(p))
            lw_m = 0.5 * p * (mm ** 2 - mu ** 2) + special.log_ndtr(-mm * math.sqrt(p))
            pp = 1.0 / (1.0 + math.exp(lw_m - lw_p))
            if u1[it, i] < pp:
                cur[i] = stats.truncnorm.ppf(u2[it, i], (0.0 - mp) / sd, np.inf, loc=mp, scale=sd)
            else:
                cur[i] = stats.truncnorm.ppf(u2[it, i], -np.inf, (0.0 - mm) / sd, loc=mm, scale=sd)
        draws.append(cur.copy())
    mean_gibbs = np.mean(draws[500:], axis=0)
    sd_post = np.std(draws[500:], axis=0)
    assert np.max(np.abs(mean_rwm - mean_gibbs) / sd_post) < 0.25


def _ar1(rho, n, seed):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


@pytest.mark.parametrize("rho", [0.5, 0.9])
def test_iat_matches_the_ar1_value(rho):
    est = np.mean([m.iat(_ar1(rho, 100000, seed)) for seed in (1, 2, 3)])
    assert est == pytest.approx((1.0 + rho) / (1.0 - rho), rel=0.15)


def test_rhat_detects_unconverged_and_accepts_converged_chains():
    rng = np.random.default_rng(2)
    same = rng.normal(size=(4, 2000))
    apart = same + np.array([[0.0], [1.0], [2.0], [3.0]])
    assert m.rhat(same) == pytest.approx(1.0, abs=0.01) and m.rhat(apart) > 1.5


def test_the_acceptance_rules_satisfy_detailed_balance_at_the_mini_instance(mini):
    scipy = pytest.importorskip("scipy")
    from scipy import stats

    a, x0, b = mini
    t = m.Target(a, b, 0.1, "gauss", np.array([0.0, 10.0, 20.0]), tau=1.0, ell=0.12)
    sigma_prior = t.tau ** 2 * np.eye(3)
    prior = stats.multivariate_normal(mean=np.zeros(3), cov=sigma_prior)
    x, y = np.array([1.2, 1.6, 1.1]), np.array([1.1, 1.7, 1.3])
    beta = 0.3

    def pi(v):
        return math.exp(-t.phi(v)[0]) * prior.pdf(v)

    def q_pcn(u, v):
        return stats.multivariate_normal(mean=math.sqrt(1 - beta ** 2) * u, cov=beta ** 2 * sigma_prior).pdf(v)

    def alpha_pcn(u, v):
        return min(1.0, math.exp(t.phi(u)[0] - t.phi(v)[0]))

    assert pi(x) * q_pcn(x, y) * alpha_pcn(x, y) == pytest.approx(pi(y) * q_pcn(y, x) * alpha_pcn(y, x), rel=1e-9)

    def alpha_rwm(u, v):
        return min(1.0, math.exp(-t.neg_log_post(v)[0] + t.neg_log_post(u)[0]))

    assert math.exp(-t.neg_log_post(x)[0]) * alpha_rwm(x, y) == pytest.approx(math.exp(-t.neg_log_post(y)[0]) * alpha_rwm(y, x), rel=1e-9)
    assert pi(x) == pytest.approx(math.exp(-t.neg_log_post(x)[0]) / (2 * math.pi) ** 1.5, rel=1e-6)


def test_samplers_leave_a_known_one_dimensional_non_gaussian_posterior_invariant():
    """n = 1, pseudo-Huber-Prior: Mittel und Standardabweichung der Kette gegen die numerische Integration der Dichte."""
    a = np.array([[0.8]])
    b = np.array([0.5])
    sigma, scale, eps = 0.3, 0.2, 0.05
    t = m.Target(a, b, sigma, "huber", np.array([0.5]), scale=scale, eps=eps)
    xs = np.linspace(-3.0, 3.0, 60001)
    dens = np.exp(-0.5 * (0.8 * xs - 0.5) ** 2 / sigma ** 2 - (np.sqrt(xs ** 2 + eps ** 2) - eps) / scale)
    dens /= np.trapezoid(dens, xs) if hasattr(np, "trapezoid") else np.trapz(dens, xs)
    mean_q = float(np.sum(xs * dens) * (xs[1] - xs[0]))
    sd_q = math.sqrt(float(np.sum((xs - mean_q) ** 2 * dens) * (xs[1] - xs[0])))
    x_map = m.map_estimate(t)
    r = m.sample(t, "rwm", np.tile(x_map, (8, 1)), 40000, 3, 1.5 * sd_q / 1.0, None)
    total = r["count"] * 8
    mc = float(r["sum1"].sum() / total)
    sd = math.sqrt(float(r["sum2"].sum() / total) - mc ** 2)
    assert mc == pytest.approx(mean_q, abs=0.03 * sd_q + 0.01) and sd == pytest.approx(sd_q, rel=0.05)
    # die MAP-Stelle ist das Maximum der Dichte auf dem Gitter
    assert x_map[0] == pytest.approx(float(xs[int(np.argmax(dens))]), abs=2e-4)
    # die Laplace-Näherung am MAP trifft die Standardabweichung der Dichte auf wenige Prozent (milder Prior)
    cov = m.laplace_approximation(t, x_map)[1]
    assert math.sqrt(cov[0, 0]) == pytest.approx(sd_q, rel=0.25)


def test_a_chain_started_far_away_shows_a_large_rhat_that_falls_with_more_steps():
    t, a, b, sigma = _gauss_instance(noise=1e-1)
    mean, cov = m.gauss_posterior(a, t.c, t.tau, sigma, b)
    chol = np.linalg.cholesky(cov + 1e-14 * np.eye(12))
    far = np.array([mean + 40.0 * (SplitMix64(30 + i).normal(12) @ chol.T) for i in range(4)])
    fm = m.functionals(t.s)[:1]
    short = m.sample(t, "rwm", far, 300, 3, 2.38 / math.sqrt(12), chol, fm)
    long = m.sample(t, "rwm", far, 20000, 3, 2.38 / math.sqrt(12), chol, fm)
    assert m.rhat(short["traces"][:, :, 0].T) > 1.5 and m.rhat(long["traces"][5000:, :, 0].T) < 1.1
