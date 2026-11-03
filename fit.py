"""Fit β and κ to observed regional tau instead of asserting them.

The app's APOE multipliers are assumptions. This module is the path from
assumption to measurement: given a regional tau-PET pattern (one number per
Desikan-Killiany region), find the (β, κ) that make the diffusion model
reproduce it best. Do that separately for ε4 carriers and non-carriers, and the
genotype effect becomes a fitted quantity with a confidence interval, not a
multiplier someone typed in.

What is and isn't identifiable
------------------------------
The model is linear with a constant seed. A single cross-sectional snapshot
constrains the *shape* of the pattern, which depends mostly on the ratio β/κ
and on how far the cascade has run (steps). β and κ separately are only weakly
identified from one snapshot: a slow spread with slow clearance looks much like
a fast spread with fast clearance. Two time points (baseline + follow-up) break
that degeneracy, because the *rate* of change pins down the absolute scale.

`recover()` demonstrates this on synthetic data; `tests/test_fit.py` asserts it.
Fit real data with `fit_longitudinal` whenever a follow-up scan exists.

Loss
----
Patterns are compared after scaling both to unit sum, so the seed rate (which
only sets magnitude) drops out and the fit is about *where* tau is, not how much.
Loss is 1 − Pearson r between log-scaled simulated and observed patterns; log
scaling stops the seed regions, which are orders of magnitude above the rest in
the simulation, from dominating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from model import Params, simulate

EPS = 1e-9


@dataclass
class Fit:
    beta: float
    kappa: float
    loss: float
    r: float
    steps: int
    n_evals: int

    @property
    def ratio(self) -> float:
        return self.beta / max(self.kappa, EPS)


def _shape(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0) + EPS
    return np.log(x / x.sum())


def pattern_loss(sim: np.ndarray, obs: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    """1 − Pearson r between log-normalised patterns, over regions in `mask`."""
    if mask is None:
        mask = np.ones(len(obs), dtype=bool)
    a, b = _shape(sim[mask]), _shape(obs[mask])
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    r = float((a * b).sum() / denom) if denom > 0 else 0.0
    return 1.0 - r, r


def fit_cross_sectional(W: np.ndarray, seed_idx: list[int], observed: np.ndarray,
                        steps: int = 300, mask: np.ndarray | None = None,
                        x0: tuple[float, float] = (0.05, 0.01)) -> Fit:
    """Fit (β, κ) to one regional pattern. Optimises in log space so both stay
    positive. Returns the best of a coarse grid + Nelder-Mead polish."""
    n_evals = 0

    def run(beta: float, kappa: float) -> np.ndarray:
        nonlocal n_evals
        n_evals += 1
        p = Params(beta=beta, kappa=kappa, steps=steps)
        return simulate(W, seed_idx, "ε3/ε3", p)[-1]

    def objective(theta: np.ndarray) -> float:
        beta, kappa = np.exp(theta)
        if beta > 0.5 or kappa > 0.5:
            return 2.0
        return pattern_loss(run(beta, kappa), observed, mask)[0]

    # coarse grid so Nelder-Mead starts near the basin
    best = (np.inf, np.log(x0))
    for b in np.geomspace(0.005, 0.3, 7):
        for k in np.geomspace(0.001, 0.1, 7):
            v = objective(np.log([b, k]))
            if v < best[0]:
                best = (v, np.log([b, k]))
    res = minimize(objective, best[1], method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 400})
    beta, kappa = np.exp(res.x)
    loss, r = pattern_loss(run(beta, kappa), observed, mask)
    return Fit(float(beta), float(kappa), float(loss), float(r), steps, n_evals)


def fit_longitudinal(W: np.ndarray, seed_idx: list[int], baseline: np.ndarray,
                     followup: np.ndarray, steps: int = 300, gap: int = 60,
                     mask: np.ndarray | None = None) -> Fit:
    """Fit (β, κ) to two snapshots `gap` model-steps apart. The baseline pattern
    pins the shape; the change between snapshots pins the rate, which is what
    separates β from κ. `gap` is the follow-up interval in model steps — a free
    choice that sets the time unit (e.g. gap=60 steps ≡ 2 years makes one step
    ≈ 12 days)."""
    n_evals = 0

    def objective(theta: np.ndarray) -> float:
        nonlocal n_evals
        n_evals += 1
        beta, kappa = np.exp(theta)
        if beta > 0.5 or kappa > 0.5:
            return 4.0
        h = simulate(W, seed_idx, "ε3/ε3", Params(beta=beta, kappa=kappa, steps=steps + gap))
        l1, _ = pattern_loss(h[steps], baseline, mask)
        l2, _ = pattern_loss(h[steps + gap], followup, mask)
        # rate term: relative growth of total load must match
        g_sim = h[steps + gap].sum() / max(h[steps].sum(), EPS)
        g_obs = followup.sum() / max(baseline.sum(), EPS)
        return l1 + l2 + (np.log(g_sim) - np.log(g_obs)) ** 2

    best = (np.inf, np.log([0.05, 0.01]))
    for b in np.geomspace(0.005, 0.3, 7):
        for k in np.geomspace(0.001, 0.1, 7):
            v = objective(np.log([b, k]))
            if v < best[0]:
                best = (v, np.log([b, k]))
    res = minimize(objective, best[1], method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 400})
    beta, kappa = np.exp(res.x)
    h = simulate(W, seed_idx, "ε3/ε3", Params(beta=beta, kappa=kappa, steps=steps + gap))
    loss, r = pattern_loss(h[steps + gap], followup, mask)
    return Fit(float(beta), float(kappa), float(res.fun), float(r), steps, n_evals)


def recover(W: np.ndarray, seed_idx: list[int], true_beta: float, true_kappa: float,
            noise: float = 0.05, steps: int = 300, gap: int = 60, seed: int = 0) -> dict:
    """Simulate with known (β, κ), add multiplicative noise, fit both ways.
    Shows what each design can and cannot recover."""
    rng = np.random.default_rng(seed)
    h = simulate(W, seed_idx, "ε3/ε3", Params(beta=true_beta, kappa=true_kappa, steps=steps + gap))
    base = h[steps] * rng.lognormal(0, noise, len(h[steps]))
    foll = h[steps + gap] * rng.lognormal(0, noise, len(h[steps]))
    cs = fit_cross_sectional(W, seed_idx, base, steps=steps)
    lg = fit_longitudinal(W, seed_idx, base, foll, steps=steps, gap=gap)
    return {"true": (true_beta, true_kappa), "cross_sectional": cs, "longitudinal": lg}


def bootstrap_ci(fit_fn, n: int = 100, seed: int = 0, **kw) -> dict:
    """Region-resampling bootstrap for a fit. `fit_fn(mask=...)` must accept a
    boolean region mask. Returns 2.5/50/97.5 percentiles of β, κ and β/κ."""
    rng = np.random.default_rng(seed)
    n_regions = kw.pop("n_regions")
    out = []
    for _ in range(n):
        idx = rng.integers(0, n_regions, n_regions)
        mask = np.zeros(n_regions, dtype=bool)
        mask[np.unique(idx)] = True
        f = fit_fn(mask=mask, **kw)
        out.append((f.beta, f.kappa, f.ratio))
    out = np.array(out)
    q = lambda j: tuple(np.percentile(out[:, j], [2.5, 50, 97.5]))
    return {"beta": q(0), "kappa": q(1), "ratio": q(2), "samples": out}
