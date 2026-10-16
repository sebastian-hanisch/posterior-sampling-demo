"""Kern: Ziele (Gauß-, Laplace-, pseudo-Huber-Prior), MAP, Laplace-Näherung, Sampler (pCN, Zufallsschritt-Metropolis), Diagnostik (IAT, ESS, R-Dach),
Identifikation. Nur numpy, kein Streamlit.

Jede Funktion ist eine kleine, einzeln prüfbare Einheit (DEMO-PLAYBOOK §1b): `tests/test_model.py` rechnet sie an der 3 x 3-Mini-Instanz von Hand und
`tests/test_oracle_smp.py` gegen den exakten Gauß-Posterior, das exakte Gibbs-Verfahren (Laplace-Prior), AR(1)-Ketten und die Reversibilität der Annahmeregel.

Modell: b = A x + e, e ~ N(0, sigma^2 I), Prior eines von
    gauss:   x ~ N(0, tau^2 C) mit C_ij = exp(-|s_i - s_j| / ell)      -log pi = 1/2 x^T P x,  P = (tau^2 C)^{-1}
    laplace: x_i ~ Laplace(0, b) unabhängig                             -log pi = |x|_1 / b
    huber:   -log pi = sum (sqrt(x_i^2 + eps^2) - eps) / b               (glatte Näherung des Laplace-Priors)
Die negative Log-Posterior ist Phi(x) + (-log pi) mit Phi(x) = |A x - b|^2 / (2 sigma^2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import smp_constants as C
from inverse_generator import SplitMix64

_TINY = 1e-300


# ----------------------------------------------------------------------------- Prior und Ziel

def exp_kernel(s: np.ndarray, ell: float) -> np.ndarray:
    """Korrelationsmatrix exp(-|s_i - s_j| / ell) (Ornstein-Uhlenbeck); die Präzisionsmatrix ist gut konditioniert."""
    s = np.asarray(s, dtype=float)
    return np.exp(-np.abs(s[:, None] - s[None, :]) / ell)


def gauss_posterior(a: np.ndarray, c: np.ndarray, tau: float, sigma: float, b: np.ndarray):
    """Exakter Posterior N(m, G) für den Gaußschen Prior N(0, tau^2 C) in Datenraumformeln: (Mittel (n,), Kovarianz (n, n))."""
    k = tau ** 2 * (a @ c @ a.T) + sigma ** 2 * np.eye(a.shape[0])
    gain = tau ** 2 * (c @ a.T)
    mean = gain @ np.linalg.solve(k, b)
    cov = tau ** 2 * c - gain @ np.linalg.solve(k, gain.T)
    return mean, 0.5 * (cov + cov.T)


@dataclass
class Target:
    """Ein Posterior: Vorwärtsmatrix, Daten, Rauschen und Prior (`prior` in C.PRIORS); `s` ist das Gitter für den Gaußschen Prior."""
    a: np.ndarray
    b: np.ndarray
    sigma: float
    prior: str
    s: np.ndarray
    tau: float = C.PRIOR_TAU
    ell: float = C.PRIOR_ELL
    scale: float = C.DEFAULT_SCALE
    eps: float = C.DEFAULT_EPS

    def __post_init__(self):
        self.n = self.a.shape[1]
        self.ata = self.a.T @ self.a / self.sigma ** 2
        self.atb = self.a.T @ self.b / self.sigma ** 2
        if self.prior == "gauss":
            self.c = exp_kernel(self.s, self.ell)
            self.chol_prior = np.linalg.cholesky(self.tau ** 2 * self.c + 1e-10 * np.eye(self.n))
            self.prec = np.linalg.inv(self.tau ** 2 * self.c + 1e-10 * np.eye(self.n))

    def phi(self, x: np.ndarray) -> np.ndarray:
        """Likelihood-Potential |A x - b|^2 / (2 sigma^2) für jede Zeile von x."""
        r = np.atleast_2d(x) @ self.a.T - self.b[None, :]
        return 0.5 * np.sum(r ** 2, axis=1) / self.sigma ** 2

    def neg_log_prior(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        if self.prior == "gauss":
            return 0.5 * np.einsum("ij,jk,ik->i", x, self.prec, x)
        if self.prior == "laplace":
            return np.sum(np.abs(x), axis=1) / self.scale
        return np.sum(np.sqrt(x ** 2 + self.eps ** 2) - self.eps, axis=1) / self.scale

    def neg_log_post(self, x: np.ndarray) -> np.ndarray:
        return self.phi(x) + self.neg_log_prior(x)

    def prior_draws(self, count: int, seed: int) -> np.ndarray:
        """Ziehungen aus dem Prior (Laplace-Prior und pseudo-Huber: unabhängige Laplace-Werte), Form (count, n)."""
        rng = SplitMix64(seed)
        if self.prior == "gauss":
            return rng.normal(count * self.n).reshape(count, self.n) @ self.chol_prior.T
        u = rng.uniform(count * self.n).reshape(count, self.n) - 0.5
        return -self.scale * np.sign(u) * np.log(np.maximum(1.0 - 2.0 * np.abs(u), 1e-300))


# ----------------------------------------------------------------------------- MAP und Laplace-Näherung

def lasso_active_set(m: np.ndarray, y: np.ndarray, lam: float, maxit: int = 5000, tol: float = 1e-10) -> np.ndarray:
    """Exakte Lösung von min 1/2 |M w - y|^2 + lam |w|_1 per Aktivmenge (wie in Stück 6): auf der aktiven Menge mit festen Vorzeichen das lineare System lösen,
    bei Vorzeichenwechsel die Koordinate mit dem frühesten Wechsel entfernen, sonst die am stärksten verletzte Bedingung |g_j| <= lam aufnehmen."""
    p = m.shape[1]
    mtm, mty = m.T @ m, m.T @ y
    w = np.zeros(p)
    active = np.zeros(p, dtype=bool)
    sgn = np.zeros(p)
    for _ in range(maxit):
        idx = np.flatnonzero(active)
        if idx.size:
            s = sgn[idx]
            sol = np.linalg.solve(mtm[idx][:, idx], mty[idx] - lam * s)
            bad = sol * s < 0.0
            if bad.any():
                w_old = w[idx]
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(bad, w_old / (w_old - sol), np.inf)
                j = int(np.argmin(t))
                w[idx[j]] = 0.0
                active[idx[j]] = False
                continue
            w[idx] = sol
        g = mtm @ w - mty
        viol = np.where(active, 0.0, np.abs(g) - lam)
        j = int(np.argmax(viol))
        if viol[j] <= tol * max(lam, _TINY):
            return w
        active[j] = True
        sgn[j] = -np.sign(g[j])
        w[j] = 0.0
    return w


def hessian(t: Target, x: np.ndarray):
    """Hesse-Matrix der negativen Log-Posterior bei x; None für den Laplace-Prior (der Betrag ist bei null nicht differenzierbar)."""
    if t.prior == "gauss":
        return t.ata + t.prec
    if t.prior == "huber":
        return t.ata + np.diag(t.eps ** 2 / (x ** 2 + t.eps ** 2) ** 1.5 / t.scale)
    return None


def map_estimate(t: Target) -> np.ndarray:
    """MAP: Gauß: Posterior-Mittel; Laplace: L1 mit lambda = sigma^2 / b (Stück 6); pseudo-Huber: Newton-Verfahren mit Schrittweitensuche."""
    if t.prior == "gauss":
        return np.linalg.solve(t.ata + t.prec, t.atb)
    if t.prior == "laplace":
        return lasso_active_set(t.a, t.b, t.sigma ** 2 / t.scale)
    x = np.zeros(t.n)
    for _ in range(300):
        grad = t.ata @ x - t.atb + x / np.sqrt(x ** 2 + t.eps ** 2) / t.scale
        step = np.linalg.solve(hessian(t, x), grad)
        h = 1.0
        while t.neg_log_post(x - h * step)[0] > t.neg_log_post(x)[0] and h > 1e-10:
            h *= 0.5
        x = x - h * step
        if np.linalg.norm(step) < 1e-12:
            break
    return x


def laplace_approximation(t: Target, x_map: np.ndarray):
    """(Mittel, Kovarianz) der Laplace-Näherung N(x_map, H^-1); None für den Laplace-Prior."""
    h = hessian(t, x_map)
    if h is None:
        return None
    cov = np.linalg.inv(h)
    return x_map, 0.5 * (cov + cov.T)


def proposal_covariance(t: Target, x_map: np.ndarray) -> np.ndarray:
    """Kovarianz der Vorschläge des vorkonditionierten Samplers: exakter Posterior (Gauß), inverse Hesse-Matrix (pseudo-Huber), (A^T A / sigma^2 + 4 I / b^2)^-1 (Laplace)."""
    if t.prior == "laplace":
        return np.linalg.inv(t.ata + np.eye(t.n) / (t.scale / 2.0) ** 2)
    return np.linalg.inv(hessian(t, x_map))


def functionals(s: np.ndarray) -> np.ndarray:
    """Drei Funktionale als Zeilen (3, n): Mittelwert, Wert bei s = 0,58, raue Richtung (alternierende Vorzeichen)."""
    n = len(s)
    return np.vstack([np.full(n, 1.0 / n), np.eye(n)[int(np.argmin(np.abs(s - 0.58)))], ((-1.0) ** np.arange(n)) / n])


FUNCTIONAL_NAMES = ("Mittelwert", "Wert bei s = 0,58", "raue Richtung")


# ----------------------------------------------------------------------------- Sampler

def sample(t: Target, kind: str, x0: np.ndarray, steps: int, seed: int, step: float, proposal_chol: np.ndarray | None = None, fmat: np.ndarray | None = None) -> dict:
    """Vektorisiert über die Ketten (Zeilen von x0). kind `pcn`: x' = sqrt(1 - step^2) x + step xi, xi ~ N(0, tau^2 C), Annahme exp(Phi(x) - Phi(x')) (nur Gauß-Prior).
    kind `rwm`: x' = x + step L z, Annahme exp(-U(x') + U(x)) mit der negativen Log-Posterior U; L = proposal_chol (Einheitsmatrix, wenn None).
    Gibt zurück: traces (steps, chains, F) der Funktionale, sum1 und sum2 (nach dem Verwerfen), count, thinned (m, chains, n), acc (chains,)."""
    x = np.array(np.atleast_2d(x0), dtype=float)
    chains, n = x.shape
    fmat = np.zeros((0, n)) if fmat is None else fmat
    rng = SplitMix64(seed)
    burn = int(C.BURN_FRACTION * steps)
    thin = max(1, (steps - burn) // C.THIN_TARGET)
    traces = np.empty((steps, chains, fmat.shape[0]))
    s1, s2 = np.zeros((chains, n)), np.zeros((chains, n))
    kept, thinned = 0, []
    acc = np.zeros(chains)
    if kind == "pcn":
        cur = t.phi(x)
        shrink = math.sqrt(1.0 - step ** 2)
    else:
        cur = -t.neg_log_post(x)
    done = 0
    while done < steps:
        blk = min(C.BLOCK, steps - done)
        z = rng.normal(blk * chains * n).reshape(blk, chains, n)
        u = np.log(np.maximum(rng.uniform(blk * chains).reshape(blk, chains), _TINY))
        for i in range(blk):
            if kind == "pcn":
                prop = shrink * x + step * (z[i] @ t.chol_prior.T)
                new = t.phi(prop)
                ok = u[i] < cur - new
            else:
                prop = x + step * (z[i] if proposal_chol is None else z[i] @ proposal_chol.T)
                new = -t.neg_log_post(prop)
                ok = u[i] < new - cur
            x = np.where(ok[:, None], prop, x)
            cur = np.where(ok, new, cur)
            acc += ok
            traces[done + i] = x @ fmat.T
            if done + i >= burn:
                s1 += x
                s2 += x ** 2
                kept += 1
                if (done + i - burn) % thin == 0:
                    thinned.append(x.copy())
        done += blk
    return {"traces": traces, "sum1": s1, "sum2": s2, "count": kept, "thinned": np.array(thinned), "acc": acc / steps}


def tune_step(t: Target, kind: str, x0: np.ndarray, seed: int, proposal_chol: np.ndarray | None = None, grid=None) -> tuple[float, float]:
    """Wählt aus dem Gitter (Standard pCN: BETAS, isotropes RWM: HS) den Schritt, dessen Pilot-Annahmerate dem Ziel am nächsten liegt; gibt (Schritt, Annahmerate) zurück."""
    default, target = (C.BETAS, C.TARGET_ACC_PCN) if kind == "pcn" else (C.HS, C.TARGET_ACC_RWM)
    grid = default if grid is None else grid
    start = np.array(np.atleast_2d(x0)[: C.PILOT_CHAINS])
    best = None
    for h in grid:
        acc = float(np.mean(sample(t, kind, start, C.PILOT_STEPS, seed, h, proposal_chol)["acc"]))
        if best is None or abs(acc - target) < best[0]:
            best = (abs(acc - target), h, acc)
    return best[1], best[2]


# ----------------------------------------------------------------------------- Diagnostik

def iat(x: np.ndarray) -> float:
    """Integrierte Autokorrelationszeit einer Kette (Sokal-Fenster: erstes M mit M >= 5 tau(M)); inf bei konstanter Kette."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if float(x @ x) <= 0.0:
        return math.inf
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf = acf / acf[0]
    tau = 1.0
    for m in range(1, n):
        tau = 1.0 + 2.0 * float(np.sum(acf[1:m + 1]))
        if m >= 5.0 * tau:
            break
    return max(tau, 1.0)


def rhat(chains: np.ndarray) -> float:
    """Gelman-Rubin R-Dach für Ketten der Form (chains, steps)."""
    chains = np.asarray(chains, dtype=float)
    m, n = chains.shape
    w = float(chains.var(axis=1, ddof=1).mean())
    bvar = n * float(chains.mean(axis=1).var(ddof=1))
    if w <= 0.0:
        return math.inf
    return math.sqrt(((n - 1) / n * w + bvar / n) / w)


def effective_sample_size(traces: np.ndarray) -> float:
    """Summe über die Ketten von N / IAT für Spuren (steps, chains) nach dem Verwerfen."""
    n = traces.shape[0]
    taus = [iat(traces[:, c]) for c in range(traces.shape[1])]
    return float(sum(n / t for t in taus))


# ----------------------------------------------------------------------------- Identifikation

def identification_eigenvalues(a: np.ndarray, c: np.ndarray, tau: float, sigma: float) -> np.ndarray:
    """Eigenwerte lambda_i (absteigend) der prior-gewichteten Hesse-Matrix tau^2 R^T A^T A R / sigma^2 mit R R^T = C; lambda_i > 1: die Daten bestimmen die Richtung
    stärker als der Prior (die Posterior-Varianz fällt auf weniger als die Hälfte), Posterior-Varianz der Richtung = tau^2 / (1 + lambda_i)."""
    ev, evec = np.linalg.eigh(0.5 * (c + c.T))
    r = evec * np.sqrt(np.maximum(ev, 0.0))
    h = tau ** 2 * r.T @ a.T @ a @ r / sigma ** 2
    return np.sort(np.linalg.eigvalsh(0.5 * (h + h.T)))[::-1]
