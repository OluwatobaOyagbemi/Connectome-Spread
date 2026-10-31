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
