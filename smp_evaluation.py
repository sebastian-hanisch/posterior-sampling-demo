"""Auswertung: Welten, Kennzahlen der Instanz, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import smp_constants as C
import smp_model as model
from inverse_generator import SplitMix64, blurred_exact, grid, normal_cdf, operator, phantom

_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,5·10⁹."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def num_text(x: float) -> str:
    """Zahl mit drei gültigen Stellen, ab 1000 und unter 0,01 als Zehnerpotenz."""
    if not math.isfinite(x):
        return "unendlich"
    if abs(x) >= 1000.0 or (x != 0.0 and abs(x) < 0.01):
        return sci(x, 1)
    return de(x, 1 if abs(x) >= 10 else 2)


def iat_text(x: float) -> str:
    """Autokorrelationszeit als ganze Zahl; inf als „steht“."""
    return "steht" if not math.isfinite(x) else f"{x:,.0f}".replace(",", ".")


def err_pct(err: float) -> str:
    """Fehler als Prozent der Signalnorm: ab 100 Prozent ohne, ab 1 Prozent mit einer, ab 0,01 Prozent mit zwei Stellen, darunter als Zehnerpotenz;
    ab dem Zehnfachen der Signalnorm als Vielfaches."""
    if not math.isfinite(err):
        return "unendlich"
    if err >= 10.0:
        return f"{sci(err)}-fach"
    x = 100.0 * err
    if x >= 99.5:
        return f"{de(x, 0)} %"
    if x >= 0.95:
        return f"{de(x, 1)} %"
    if x >= 0.01:
        return f"{de(x, 2)} %"
    return f"{sci(x)} %"


def noise_label(v: float) -> str:
    """Rauschstufe als Prozent des Effektivwerts, z. B. 0,1 %."""
    return de(100.0 * v, 4).rstrip("0").rstrip(",") + " %" if v > 0.0 else "0 %"


def share(x: float) -> str:
    """Anteil (0 bis 1) als ganze Prozent."""
    return f"{de(100.0 * x, 0)} %"


def step_text(h: float) -> str:
    return de(h, 3) if h >= 0.01 else sci(h, 1)


# ----------------------------------------------------------------------------- Welten

def world_truth(world: str, n: int, w: float):
    """(x, g): Wahrheit am Gitter und ihre analytische unscharfe Antwort (vom Gitter unabhängig)."""
    s = grid(n)
    if world == "phantom":
        return phantom(s), blurred_exact(s, w)
    x = np.zeros(n)
    g = np.zeros(n)
    for pos, mass in ((0.25, 0.04), (0.50, 0.03), (0.70, 0.05)):
        j = int(round(pos * n - 0.5))
        x[j] += mass * n
        g += mass * np.exp(-((s - s[j]) ** 2) / (2.0 * w * w)) / (w * math.sqrt(2.0 * math.pi))
    return x, g


@lru_cache(maxsize=32)
def instance(world: str, n: int, w: float, noise: float, seed: int):
    """(a, s, x, b, sigma): Matrix, Gitter, Wahrheit, Daten (analytische Antwort plus festes Rauschen), absolutes Rauschen."""
    a = operator(n, w)
    x, g = world_truth(world, n, w)
    sigma = float(noise) * float(np.sqrt(np.mean(g ** 2)))
    b = g + SplitMix64(seed).normal(n) * sigma
    return a, grid(n), x, b, sigma


# ----------------------------------------------------------------------------- Instanz

@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    w: float = C.DEFAULT_W
    noise: float = C.DEFAULT_NOISE
    world: str = "phantom"
    prior: str = "gauss"
    sampler: str = "rwm_cov"
    steps: int = C.DEFAULT_STEPS
    chains: int = C.DEFAULT_CHAINS
    start: str = "map"
    scale: float = C.DEFAULT_SCALE
    eps: float = C.DEFAULT_EPS
    seed: int = C.DEFAULT_SEED


@dataclass
class Analysis:
    settings: Settings
    s: np.ndarray
    x: np.ndarray
    b: np.ndarray
    sampler_used: str
    fallback: bool            # pCN verlangt einen Gaußschen Prior; bei anderem Prior läuft RWM mit Kovarianz
    step: float
    x_map: np.ndarray
    mean: np.ndarray          # Kettenmittel (alle Ketten, nach dem Verwerfen)
    sd: np.ndarray
    lo: np.ndarray            # 2,5-%-Quantil der ausgedünnten Ketten
    hi: np.ndarray
    ref_mean: object          # exakter (Gauß) oder Laplace-genäherter Mittelwert, sonst None
    ref_sd: object
    ref_kind: str             # "exakt", "Laplace-Näherung" oder ""
    acc: float
    iats: list                # je Funktional, Mittel über die Ketten
    ess: list
    rhats: list
    traces: np.ndarray        # (Punkte, Ketten, Funktionale), ausgedünnt für die Anzeige
    trace_steps: np.ndarray
    burn: int
    mean_err_sd: float        # größter |Kettenmittel - Referenz| in sd der Referenz (nan ohne Referenz)
    sd_err: float             # größter |Ketten-sd / Referenz-sd - 1|
    sd_ratio_median: float    # Median Referenz-sd / Ketten-sd
    err_map: float
    err_mean: float
    nnz_map: int
    nnz_mean: int
    cond: float
    singular: bool
    sigma: float
    thinned: np.ndarray


def _starts(t: model.Target, st: Settings, x_map: np.ndarray, pcov: np.ndarray) -> np.ndarray:
    if st.start == "prior":
        return t.prior_draws(st.chains, st.seed + 5)
    z = SplitMix64(st.seed + 9).normal(st.chains * t.n).reshape(st.chains, t.n)
    return x_map[None, :] + 2.0 * z @ np.linalg.cholesky(pcov + 1e-14 * np.eye(t.n)).T


def analyse(settings: Settings) -> Analysis:
    st = settings
    a, s, x, b, sigma = instance(st.world, st.n, st.w, st.noise, st.seed)
    t = model.Target(a, b, sigma, st.prior, s, scale=st.scale, eps=st.eps)
    x_map = model.map_estimate(t)
    pcov = model.proposal_covariance(t, x_map)
    pchol = np.linalg.cholesky(pcov + 1e-14 * np.eye(st.n))
    fmat = model.functionals(s)
    used = st.sampler if not (st.sampler == "pcn" and st.prior != "gauss") else "rwm_cov"
    x0 = _starts(t, st, x_map, pcov)
    if used == "pcn":
        step, _ = model.tune_step(t, "pcn", x0, st.seed + 1)
        res = model.sample(t, "pcn", x0, st.steps, st.seed + 2, step, None, fmat)
    elif used == "rwm_iso":
        step, _ = model.tune_step(t, "rwm", x0, st.seed + 1)
        res = model.sample(t, "rwm", x0, st.steps, st.seed + 2, step, None, fmat)
    else:
        step = C.RWM_COV_SCALE_FACTOR / math.sqrt(st.n)
        res = model.sample(t, "rwm", x0, st.steps, st.seed + 2, step, pchol, fmat)
    burn = int(C.BURN_FRACTION * st.steps)
    total = res["count"] * st.chains
    mean = res["sum1"].sum(axis=0) / total
    sd = np.sqrt(np.maximum(res["sum2"].sum(axis=0) / total - mean ** 2, 0.0))
    pooled = res["thinned"].reshape(-1, st.n)
    lo, hi = np.quantile(pooled, [0.025, 0.975], axis=0)
    after = res["traces"][burn:]
    iats = [float(np.mean([model.iat(after[:, c, f]) for c in range(st.chains)])) for f in range(fmat.shape[0])]
    ess = [model.effective_sample_size(after[:, :, f]) for f in range(fmat.shape[0])]
    rhats = [model.rhat(after[:, :, f].T) for f in range(fmat.shape[0])]
    ref_mean = ref_sd = None
    ref_kind = ""
    if st.prior == "gauss":
        ref_mean, cov = model.gauss_posterior(a, t.c, t.tau, sigma, b)
        ref_sd, ref_kind = np.sqrt(np.diag(cov)), "exakt"
    elif st.prior == "huber":
        mu, cov = model.laplace_approximation(t, x_map)
        ref_mean, ref_sd, ref_kind = mu, np.sqrt(np.diag(cov)), "Laplace-Näherung"
    if ref_mean is not None:
        mean_err = float(np.max(np.abs(mean - ref_mean) / ref_sd))
        sd_err = float(np.max(np.abs(sd / ref_sd - 1.0)))
        ratio = float(np.median(ref_sd / np.maximum(sd, 1e-300)))
    else:
        mean_err = sd_err = ratio = math.nan
    xn = float(np.linalg.norm(x))
    stride = max(1, st.steps // 1500)
    cond = float(np.linalg.cond(a))
    return Analysis(
        settings=st, s=s, x=x, b=b, sampler_used=used, fallback=(used != st.sampler), step=float(step), x_map=x_map, mean=mean, sd=sd, lo=lo, hi=hi, ref_mean=ref_mean,
        ref_sd=ref_sd, ref_kind=ref_kind, acc=float(np.mean(res["acc"])), iats=iats, ess=ess, rhats=rhats, traces=res["traces"][::stride], trace_steps=np.arange(0, st.steps, stride),
        burn=burn, mean_err_sd=mean_err, sd_err=sd_err, sd_ratio_median=ratio, err_map=float(np.linalg.norm(x_map - x) / xn), err_mean=float(np.linalg.norm(mean - x) / xn),
        nnz_map=int((np.abs(x_map) > 1e-9).sum()), nnz_mean=int((np.abs(mean) > 1e-3 * max(float(np.abs(mean).max()), 1e-300)).sum()), cond=cond,
        singular=bool(cond >= C.SINGULAR_COND), sigma=sigma, thinned=res["thinned"],
    )


# ----------------------------------------------------------------------------- Experiment 1: Sampler gegen Information und Dimension

def _iats(trace: np.ndarray) -> list:
    """IAT je Funktional für eine Spur (steps, F)."""
    return [model.iat(trace[:, f]) for f in range(trace.shape[1])]


def _oracle_errors(res: dict, mean: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    total = res["count"] * res["sum1"].shape[0]
    m = res["sum1"].sum(axis=0) / total
    sd = np.sqrt(np.maximum(res["sum2"].sum(axis=0) / total - m ** 2, 0.0))
    ref = np.sqrt(np.diag(cov))
    return float(np.max(np.abs(m - mean) / ref)), float(np.max(np.abs(sd / ref - 1.0)))


def sampler_experiment(steps: int = C.EXP_STEPS, w: float = C.DEFAULT_W) -> dict:
    """Teil 1: Information (Rauschen) bei n = 48; Teil 2: Dimension bei m = 24 Messwerten und Rauschen 10 %."""
    info = []
    n = C.DEFAULT_N
    for noise in C.EXP_NOISE:
        a, s, x, b, sigma = instance("phantom", n, w, noise, C.SEED_BASE + 2)
        t = model.Target(a, b, sigma, "gauss", s)
        mean, cov = model.gauss_posterior(a, t.c, t.tau, sigma, b)
        x0 = mean[None, :]
        fmat = model.functionals(s)
        chol = np.linalg.cholesky(cov + 1e-14 * np.eye(n))
        for kind in C.SAMPLERS:
            if kind == "pcn":
                h, _ = model.tune_step(t, "pcn", x0, 5)
                res = model.sample(t, "pcn", x0, steps, 11, h, None, fmat)
            elif kind == "rwm_iso":
                h, _ = model.tune_step(t, "rwm", x0, 5)
                res = model.sample(t, "rwm", x0, steps, 13, h, None, fmat)
            else:
                h = C.RWM_COV_SCALE_FACTOR / math.sqrt(n)
                res = model.sample(t, "rwm", x0, steps, 15, h, chol, fmat)
            merr, serr = _oracle_errors(res, mean, cov)
            info.append({"noise": noise, "sampler": kind, "step": float(h), "acc": float(res["acc"][0]), "iats": _iats(res["traces"][int(C.BURN_FRACTION * steps):, 0, :]),
                         "mean_err": merr, "sd_err": serr})
    # Dimension: feste Messwerte, wachsendes Gitter der Unbekannten
    m = C.EXP_M
    t_obs = (np.arange(m) + 0.5) / m
    g_obs = blurred_exact(t_obs, w)
    sigma = C.EXP_DIM_NOISE * float(np.sqrt(np.mean(g_obs ** 2)))
    b = g_obs + SplitMix64(C.SEED_BASE + 1).normal(m) * sigma
    dims = []
    fixed = {}
    for nn in C.EXP_DIMS:
        s = (np.arange(nn) + 0.5) / nn
        a = np.exp(-((t_obs[:, None] - s[None, :]) ** 2) / (2 * w * w)) / (w * math.sqrt(2 * math.pi)) / nn
        t = model.Target(a, b, sigma, "gauss", s)
        mean, _ = model.gauss_posterior(a, t.c, t.tau, sigma, b)
        x0 = mean[None, :]
        fmat = model.functionals(s)
        beta, _ = model.tune_step(t, "pcn", x0, 5, grid=C.BETAS[8:12])
        rp = model.sample(t, "pcn", x0, steps, 11, beta, None, fmat)
        h, _ = model.tune_step(t, "rwm", x0, 5, grid=C.HS[7:12])
        rr = model.sample(t, "rwm", x0, steps, 13, h, None, fmat)
        if nn == C.EXP_DIMS[0]:
            fixed = {"beta": beta, "h": h}
        fp = model.sample(t, "pcn", x0, 3000, 17, fixed["beta"], None, fmat)["acc"][0]
        fr = model.sample(t, "rwm", x0, 3000, 19, fixed["h"], None, fmat)["acc"][0]
        burn = int(C.BURN_FRACTION * steps)
        dims.append({"n": nn, "pcn": {"step": float(beta), "acc": float(rp["acc"][0]), "iats": _iats(rp["traces"][burn:, 0, :])},
                     "rwm": {"step": float(h), "acc": float(rr["acc"][0]), "iats": _iats(rr["traces"][burn:, 0, :])}, "fixed_pcn_acc": float(fp), "fixed_rwm_acc": float(fr)})
    return {"info": info, "dims": dims, "steps": steps, "n": n, "m": m}


# ----------------------------------------------------------------------------- Experiment 2: Burn-in und Start

def burnin_experiment(steps: int = C.BURN_STEPS, w: float = C.DEFAULT_W) -> dict:
    n = C.DEFAULT_N
    a, s, x, b, sigma = instance("phantom", n, w, 1e-3, C.SEED_BASE + 2)
    t = model.Target(a, b, sigma, "gauss", s)
    mean, cov = model.gauss_posterior(a, t.c, t.tau, sigma, b)
    chol = np.linalg.cholesky(cov + 1e-14 * np.eye(n))
    fmat = model.functionals(s)[:2]
    starts = {
        "map": np.array([mean + 2.0 * chol @ SplitMix64(40 + i).normal(n) for i in range(4)]),
        "prior": np.array([t.chol_prior @ SplitMix64(60 + i).normal(n) for i in range(4)]),
    }
    rows = {}
    for key, x0 in starts.items():
        res = model.sample(t, "rwm", x0, steps, 100, C.RWM_COV_SCALE_FACTOR / math.sqrt(n), chol, fmat)
        rows[key] = [{"discard": d, "rhat_point": model.rhat(res["traces"][d:, :, 1].T), "rhat_mean": model.rhat(res["traces"][d:, :, 0].T)} for d in C.BURN_DISCARD]
    return {"rows": rows, "steps": steps, "n": n}


# ----------------------------------------------------------------------------- Experiment 3: Identifikation

def identification_experiment(n: int = C.DEFAULT_N, w: float = C.DEFAULT_W) -> dict:
    s = grid(n)
    a = operator(n, w)
    g = blurred_exact(s, w)
    kernels = {
        "ou": ("exponentiell (ℓ = 0,12), der Prior der App", model.exp_kernel(s, C.PRIOR_ELL)),
        "se": ("quadratisch-exponentiell (ℓ = 0,05)", np.exp(-0.5 * (np.abs(s[:, None] - s[None, :]) / 0.05) ** 2)),
        "white": ("weiß", np.eye(n)),
    }
    rows = []
    for key, (label, c) in kernels.items():
        counts, sums = [], []
        for noise in C.ID_NOISE:
            sg = noise * float(np.sqrt(np.mean(g ** 2)))
            lam = model.identification_eigenvalues(a, c, C.PRIOR_TAU, sg)
            counts.append(int((lam > 1.0).sum()))
            sums.append(float(np.sum(lam / (1.0 + lam))))
        rows.append({"key": key, "label": label, "counts": counts, "sums": sums})
    return {"rows": rows, "n": n, "noise": C.ID_NOISE}


# ----------------------------------------------------------------------------- Experiment 4: nicht-Gaußsch

def _tikhonov_best(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    u, sv, vt = np.linalg.svd(a)
    xn = float(np.linalg.norm(x))
    return min(float(np.linalg.norm(((b @ u) * (sv ** 2 / (sv ** 2 + lam ** 2)) / sv) @ vt - x)) / xn for lam in np.logspace(-6, 0, 61))


def nongauss_experiment(chains: int = C.NG_CHAINS, w: float = C.DEFAULT_W) -> dict:
    n = C.NG_N
    a = operator(n, w)
    x, g = world_truth("spitzen", n, w)
    xn = float(np.linalg.norm(x))
    s = grid(n)
    lap, hub = [], []
    for noise, scale in C.NG_LAPLACE:
        sigma = noise * float(np.sqrt(np.mean(g ** 2)))
        b = g + SplitMix64(31).normal(n) * sigma
        t = model.Target(a, b, sigma, "laplace", s, scale=scale)
        x_map = model.map_estimate(t)
        chol = np.linalg.cholesky(model.proposal_covariance(t, x_map) + 1e-14 * np.eye(n))
        res = model.sample(t, "rwm", np.tile(x_map, (chains, 1)), 10000, 5, C.RWM_COV_SCALE_FACTOR / math.sqrt(n), chol, model.functionals(s))
        mean = res["sum1"].sum(axis=0) / (res["count"] * chains)
        lap.append({"noise": noise, "scale": scale, "map": float(np.linalg.norm(x_map - x) / xn), "mean": float(np.linalg.norm(mean - x) / xn), "tik": _tikhonov_best(a, b, x),
                    "nnz_map": int((np.abs(x_map) > 1e-9).sum()), "nnz_mean": int((np.abs(mean) > 1e-3 * float(np.abs(mean).max())).sum()), "acc": float(res["acc"].mean())})
    for eps, scale, noise in C.NG_HUBER:
        sigma = noise * float(np.sqrt(np.mean(g ** 2)))
        b = g + SplitMix64(31).normal(n) * sigma
        t = model.Target(a, b, sigma, "huber", s, scale=scale, eps=eps)
        x_map = model.map_estimate(t)
        mu, cov = model.laplace_approximation(t, x_map)
        chol = np.linalg.cholesky(cov + 1e-14 * np.eye(n))
        res = model.sample(t, "rwm", np.tile(x_map, (chains, 1)), 20000, 7, C.RWM_COV_SCALE_FACTOR / math.sqrt(n), chol, model.functionals(s))
        total = res["count"] * chains
        mean = res["sum1"].sum(axis=0) / total
        sd = np.sqrt(np.maximum(res["sum2"].sum(axis=0) / total - mean ** 2, 0.0))
        ratio = np.sqrt(np.diag(cov)) / sd
        after = res["traces"][int(C.BURN_FRACTION * 20000):]
        hub.append({"eps": eps, "scale": scale, "noise": noise, "map": float(np.linalg.norm(x_map - x) / xn), "mean": float(np.linalg.norm(mean - x) / xn),
                    "ratio_median": float(np.median(ratio)), "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()), "gap_sd": float(np.max(np.abs(x_map - mean) / sd)),
                    "acc": float(res["acc"].mean()), "rhat": float(max(model.rhat(after[:, :, f].T) for f in range(after.shape[2])))})
    return {"laplace": lap, "huber": hub, "chains": chains, "n": n}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    st = a.settings
    ess_min = min(a.ess)
    bad = []
    if max(a.rhats) > C.RHAT_OK:
        bad.append(f"R̂ = {num_text(max(a.rhats))} (über {de(C.RHAT_OK, 1)})")
    if ess_min < C.ESS_OK:
        bad.append(f"effektive Stichprobengröße {de(ess_min, 0)} (unter {de(C.ESS_OK, 0)})")
    if a.ref_kind == "exakt":
        if a.sd_err > C.SD_ERR_OK:
            bad.append(f"Standardabweichung um bis zu {share(a.sd_err)} falsch")
        if a.mean_err_sd > C.MEAN_ERR_OK:
            bad.append(f"Mittelwert um bis zu {num_text(a.mean_err_sd)} sd falsch")
    lead = (f"{C.SAMPLER_SHORT[a.sampler_used]}, {st.chains} Ketten à {st.steps} Schritte, Annahme {de(a.acc, 2)}: R̂ höchstens {num_text(max(a.rhats))}, "
            f"effektive Stichprobengröße mindestens {de(ess_min, 0)}")
    if a.ref_kind == "exakt":
        lead += f", größter Mittelwertfehler {num_text(a.mean_err_sd)} sd, größter sd-Fehler {share(a.sd_err)} gegen den exakten Posterior"
    lead += "."
    if a.ref_kind == "Laplace-Näherung":
        lead += f" Laplace-Näherung: Standardabweichung im Median das {de(a.sd_ratio_median, 2)}-Fache der Kette."
    if bad:
        return "warning", f"{lead} Nicht konvergiert: " + "; ".join(bad) + "."
    return "success", f"{lead} Die Diagnostik ist unauffällig."


def verdict_sampler(res: dict) -> tuple[str, str]:
    rows = {(r["noise"], r["sampler"]): r for r in res["info"]}
    last = res["dims"][-1]
    return "warning", (f"Bei Rauschen {noise_label(1e-3)} braucht pCN den Schritt {step_text(rows[(1e-3, 'pcn')]['step'])} (bei {noise_label(1e-1)}: {step_text(rows[(1e-1, 'pcn')]['step'])}); "
                       f"der Sampler mit der Kovarianz des Posteriors hat die Autokorrelationszeit {iat_text(rows[(1e-3, 'rwm_cov')]['iats'][1])} für den Punktwert, pCN {iat_text(rows[(1e-3, 'pcn')]['iats'][1])}. "
                       f"Bei n = {last['n']} fällt die Annahme des isotropen Samplers mit dem Schritt von n = {res['dims'][0]['n']} auf {de(last['fixed_rwm_acc'], 2)}, die von pCN bleibt bei {de(last['fixed_pcn_acc'], 2)}.")


def verdict_burnin(res: dict) -> tuple[str, str]:
    m, p = {r["discard"]: r for r in res["rows"]["map"]}, {r["discard"]: r for r in res["rows"]["prior"]}
    last = C.BURN_DISCARD[-1]
    return "warning", (f"Start um das MAP: R̂ = {de(m[0]['rhat_point'], 2)} (Punktwert) und {de(m[0]['rhat_mean'], 2)} (Mittelwert) schon ohne Verwerfen. Start aus dem Prior: "
                       f"R̂ = {num_text(p[0]['rhat_point'])} / {num_text(p[0]['rhat_mean'])} ohne Verwerfen und nach {last} verworfenen Schritten noch {num_text(p[last]['rhat_point'])} / "
                       f"{num_text(p[last]['rhat_mean'])}: Die Ketten erreichen den Posterior in {res['steps']} Schritten nicht.")


def verdict_identification(res: dict) -> tuple[str, str]:
    ou = next(r for r in res["rows"] if r["key"] == "ou")
    return "info", (f"Beim Prior der App bestimmen die Daten bei Rauschen {noise_label(res['noise'][0])} / {noise_label(res['noise'][1])} / {noise_label(res['noise'][2])} / "
                    f"{noise_label(res['noise'][3])} {ou['counts'][0]} / {ou['counts'][1]} / {ou['counts'][2]} / {ou['counts'][3]} von {res['n']} Richtungen stärker als der Prior; "
                    f"der Rest ist Prior.")


def verdict_nongauss(res: dict) -> tuple[str, str]:
    lap = {(r["noise"], r["scale"]): r for r in res["laplace"]}
    hub = res["huber"]
    smooth, thin = hub[0], hub[-1]
    k = lap[(3e-2, 0.03)]
    return "warning", (f"Laplace-Prior (b = 0,03, Rauschen 3 %): Fehler MAP {err_pct(k['map'])}, Posterior-Mittel {err_pct(k['mean'])}, Tikhonov {err_pct(k['tik'])}; das Mittel ist "
                       f"mit {k['nnz_mean']} von {res['n']} Werten dicht besetzt. Laplace-Näherung beim glatten Prior: Standardabweichung im Median {de(smooth['ratio_median'], 2)} der Kette; "
                       f"beim dünnen Prior (ε = {de(thin['eps'], 3)}, b = {de(thin['scale'], 2)}, Rauschen {noise_label(thin['noise'])}): {de(thin['ratio_median'], 2)}, "
                       f"der Wert reicht von {de(thin['ratio_min'], 2)} bis {de(thin['ratio_max'], 2)}.")
