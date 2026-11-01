"""Controls for the Braak-concordance result.

A Spearman ρ on its own answers "how ordered?", not "compared with what?".
Three controls answer the second question:

1. Shuffled-Braak null — keep the simulated arrival order, permute the Braak
   stage labels across regions. Tests whether ρ could arise from any random
   assignment of stages to the same arrival sequence.
2. Rewired-connectome null — keep the Braak labels, shuffle the edge weights
   among the connectome's existing edge slots (same density, same weight
   distribution, scrambled topology), re-simulate. Tests whether the specific
   wiring matters, or whether any graph of this density would do.
3. Seed control — keep everything, move the seed to other regions. Tests
   whether the entorhinal seed is special, as Braak staging says it should be.

All three use the same simulate → arrival_order → braak_concordance path as the
app, so there is no separate code path to drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model import Params, simulate, arrival_order, braak_stages, braak_concordance


@dataclass
class NullResult:
    observed: float
    samples: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.nanmean(self.samples))

    @property
    def p95(self) -> float:
        return float(np.nanpercentile(self.samples, 95))

    @property
    def p_value(self) -> float:
        """One-sided: fraction of null samples at least as large as observed
        (with the +1 correction so p is never reported as exactly 0)."""
        s = self.samples[~np.isnan(self.samples)]
        return float(((s >= self.observed).sum() + 1) / (len(s) + 1))


def shuffled_braak_null(arrival: np.ndarray, stages: np.ndarray,
                        n: int = 2000, seed: int = 0) -> NullResult:
    rng = np.random.default_rng(seed)
    observed = braak_concordance(arrival, stages)
    samples = np.array([braak_concordance(arrival, rng.permutation(stages))
                        for _ in range(n)])
    return NullResult(observed, samples)


def rewire(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle the upper-triangle weights among all region pairs (including
    zero slots), then symmetrise. Preserves density and the weight
    distribution; destroys which regions are connected to which."""
    iu = np.triu_indices_from(W, k=1)
    vals = W[iu].copy()
    rng.shuffle(vals)
    out = np.zeros_like(W)
    out[iu] = vals
    return out + out.T


def rewired_connectome_null(W: np.ndarray, seed_idx: list[int], names: list[str],
                            p: Params, threshold: float,
                            n: int = 200, seed: int = 0,
                            genotype: str = "ε3/ε3") -> NullResult:
    rng = np.random.default_rng(seed)
    stages = braak_stages(names)
    observed = braak_concordance(
        arrival_order(simulate(W, seed_idx, genotype, p), threshold), stages)
    samples = []
    for _ in range(n):
        h = simulate(rewire(W, rng), seed_idx, genotype, p)
        samples.append(braak_concordance(arrival_order(h, threshold), stages))
    return NullResult(observed, np.array(samples))


# Bilateral seeds that span the Braak sequence: stage I (entorhinal), III
# (fusiform), VI (pericalcarine, precentral), plus a stage-V association hub.
SEED_CONTROLS = ["entorhinal", "fusiform", "superiorparietal", "pericalcarine",
                 "precentral"]


def seed_control(W: np.ndarray, names: list[str], p: Params, threshold: float,
                 seeds: list[str] = SEED_CONTROLS,
                 genotype: str = "ε3/ε3") -> list[dict]:
    """Braak concordance when the simulation is seeded bilaterally in each
    region in `seeds`. Returns one row per seed."""
    stages = braak_stages(names)
    rows = []
    for s in seeds:
        idx = [i for i, n in enumerate(names) if n.endswith(s)]
        if not idx:
            continue
        a = arrival_order(simulate(W, idx, genotype, p), threshold)
        rows.append({
            "seed": s,
            "Braak stage of seed": _stage_label(stages[idx[0]]),
            "regions reached": int(np.isfinite(a).sum()),
            "ρ": braak_concordance(a, stages),
        })
    return rows


def threshold_control(W: np.ndarray, seed_idx: list[int], names: list[str],
                      p: Params, thresholds=(0.02, 0.05, 0.10),
                      genotype: str = "ε3/ε3") -> list[dict]:
    """Does the ordering survive when the arrival threshold is lowered so that
    every region arrives and nothing is tied for last?"""
    stages = braak_stages(names)
    h = simulate(W, seed_idx, genotype, p)
    rows = []
    for t in thresholds:
        a = arrival_order(h, t)
        rows.append({"threshold": t, "regions reached": int(np.isfinite(a).sum()),
                     "ρ": braak_concordance(a, stages)})
    return rows


def _stage_label(s: float) -> str:
    return "—" if np.isnan(s) else ["", "I", "II", "III", "IV", "V", "VI"][int(s)]
