"""Network diffusion model of pathology spread.

Core equation (Raj, Kuceyeski & Weiner, Neuron 2012, simplified):

    dx/dt = -beta * L @ x  -  kappa * x  +  seed(t)

    x      : pathology load in each region (vector)
    L      : graph Laplacian of the connectome, L = D - W
    beta   : spread rate along connections
    kappa  : clearance rate (how fast the brain removes pathology)
    seed   : constant trickle of new pathology in the seed region(s)

The Laplacian term moves pathology from regions with more of it to connected
regions with less, in proportion to connection strength. That is the whole
"spread along the wiring" idea in one line of algebra.

APOE genotype changes beta and kappa. These multipliers are illustrative,
not fitted to data: APOE4 is modelled as faster spread + weaker clearance,
APOE2 as the reverse. Say so in the README.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Multipliers applied to the baseline (beta, kappa).
APOE_PROFILES = {
    "ε2/ε2": {"beta": 0.70, "kappa": 1.40, "label": "protective"},
    "ε2/ε3": {"beta": 0.85, "kappa": 1.20, "label": "mildly protective"},
    "ε3/ε3": {"beta": 1.00, "kappa": 1.00, "label": "reference"},
    "ε3/ε4": {"beta": 1.35, "kappa": 0.75, "label": "elevated risk"},
    "ε4/ε4": {"beta": 1.80, "kappa": 0.50, "label": "high risk"},
}


@dataclass
class Params:
    beta: float = 0.05      # baseline spread rate
    kappa: float = 0.01     # baseline clearance rate
    seed_rate: float = 0.05 # pathology produced per step in seed regions
    dt: float = 1.0
    steps: int = 300


def laplacian(W: np.ndarray, normalised: bool = True) -> np.ndarray:
    d = W.sum(axis=1)
    L = np.diag(d) - W
    if normalised:
        with np.errstate(divide="ignore"):
            inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        L = inv_sqrt[:, None] * L * inv_sqrt[None, :]
    return L


def simulate(W: np.ndarray, seed_idx: list[int], genotype: str,
             p: Params = Params()) -> np.ndarray:
    """Run the diffusion forward. Returns array of shape (steps + 1, n_regions)."""
    prof = APOE_PROFILES[genotype]
    beta = p.beta * prof["beta"]
    kappa = p.kappa * prof["kappa"]

    n = W.shape[0]
    L = laplacian(W)
    seed = np.zeros(n)
    seed[seed_idx] = p.seed_rate

    x = np.zeros(n)
    history = [x.copy()]
    for _ in range(p.steps):
        dx = -beta * (L @ x) - kappa * x + seed
        x = np.maximum(x + p.dt * dx, 0.0)
        history.append(x.copy())
    return np.array(history)


def arrival_order(history: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """For each region, the first time step its load crosses `threshold`.
    Regions that never cross get +inf. Lower = reached earlier."""
    crossed = history >= threshold
    first = np.where(crossed.any(axis=0), crossed.argmax(axis=0), np.inf)
    return first


# ---------------------------------------------------------------------------
# Braak staging reference
#
# Cortical Desikan-Killiany regions grouped into Braak tau stages, following the
# ROI scheme used for tau-PET staging (Schöll et al., Neuron 2016; Baker et al.,
# Data in Brief 2017). Stage II (hippocampus) is subcortical and not in the
# cortical parcellation, so cortical stages run I, III, IV, V, VI.
# ---------------------------------------------------------------------------
BRAAK_STAGE = {
    "entorhinal": 1,
    "parahippocampal": 3, "fusiform": 3, "lingual": 3,
    "middletemporal": 4, "caudalanteriorcingulate": 4, "rostralanteriorcingulate": 4,
    "posteriorcingulate": 4, "isthmuscingulate": 4, "insula": 4,
    "inferiortemporal": 4, "temporalpole": 4,
    "superiorfrontal": 5, "lateralorbitofrontal": 5, "medialorbitofrontal": 5,
    "frontalpole": 5, "caudalmiddlefrontal": 5, "rostralmiddlefrontal": 5,
    "parsopercularis": 5, "parsorbitalis": 5, "parstriangularis": 5,
    "lateraloccipital": 5, "supramarginal": 5, "inferiorparietal": 5,
    "superiortemporal": 5, "superiorparietal": 5, "precuneus": 5,
    "bankssts": 5, "transversetemporal": 5,
    "pericalcarine": 6, "postcentral": 6, "cuneus": 6, "precentral": 6,
    "paracentral": 6,
}


def braak_stages(names: list[str]) -> np.ndarray:
    """Braak stage per region name (e.g. 'L_fusiform'); NaN if unmapped."""
    out = []
    for n in names:
        base = n.split("_", 1)[1] if "_" in n else n
        out.append(BRAAK_STAGE.get(base, np.nan))
    return np.array(out, dtype=float)


def braak_concordance(arrival: np.ndarray, stages: np.ndarray) -> float:
    """Spearman rank correlation between simulated arrival step and Braak stage.

    1.0 = the simulation reaches regions in exactly Braak order;
    0.0 = no relationship. Regions that never arrive are ranked last (tied).
    """
    mask = ~np.isnan(stages)
    a = arrival[mask].copy()
    s = stages[mask]
    finite = np.isfinite(a)
    if finite.sum() < 3:
        return float("nan")
    a[~finite] = a[finite].max() + 1  # never-arrived regions tie for last
    ra = _rank(a)
    rs = _rank(s)
    ra -= ra.mean()
    rs -= rs.mean()
    denom = np.sqrt((ra ** 2).sum() * (rs ** 2).sum())
    return float((ra * rs).sum() / denom) if denom > 0 else float("nan")


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (handles ties), no scipy dependency."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def beta_sweep(W: np.ndarray, seed_idx: list[int], names: list[str],
               betas: np.ndarray, p: Params, threshold: float) -> dict[str, np.ndarray]:
    """Braak concordance as a function of baseline beta, for every genotype."""
    stages = braak_stages(names)
    out = {}
    for g in APOE_PROFILES:
        vals = []
        for b in betas:
            pp = Params(beta=float(b), kappa=p.kappa, seed_rate=p.seed_rate,
                        dt=p.dt, steps=p.steps)
            h = simulate(W, seed_idx, g, pp)
            vals.append(braak_concordance(arrival_order(h, threshold), stages))
        out[g] = np.array(vals)
    return out
