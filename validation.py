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

from model import Params, simulate, arrival_order, braak_stages, braak_concordance, _rank


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


# ---------------------------------------------------------------------------
# Geometry controls
#
# Braak stages are spatially contiguous: they start in the medial temporal lobe
# and move outward. So anything that spreads smoothly away from entorhinal
# cortex will correlate with Braak order, wiring or not. The rewired null above
# destroys spatial structure as well as topology, which makes it easy to beat.
# These controls ask the harder question: does the connectome know anything
# about Braak order that straight-line distance does not?
# ---------------------------------------------------------------------------

def distance_from_seed(xyz: np.ndarray, seed_idx: list[int]) -> np.ndarray:
    """Euclidean distance (mm) from each region to its nearest seed region.
    Used directly as a parameter-free 'arrival order': nearer = earlier."""
    D = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    return D[:, seed_idx].min(axis=1)


def hemisphere(names: list[str]) -> np.ndarray:
    """'L' or 'R' per region, for DK-style ('L_fusiform'), Schaefer-style
    ('7Networks_LH_Vis_1') or Glasser-style ('L_V1') names."""
    out = []
    for n in names:
        if "_LH_" in n or n.startswith(("L_", "lh_", "LH_")):
            out.append("L")
        elif "_RH_" in n or n.startswith(("R_", "rh_", "RH_")):
            out.append("R")
        else:
            raise ValueError(f"cannot tell hemisphere of {n!r}")
    return np.array(out)


def geometry_preserving_rewire(W: np.ndarray, xyz: np.ndarray, names: list[str],
                               rng: np.random.Generator, n_bins: int = 10) -> np.ndarray:
    """Shuffle edge weights only among region pairs at similar distance, and
    separately within and between hemispheres. Keeps the weight-distance
    relationship (near regions stay strongly connected) while scrambling which
    specific regions are wired together. After Betzel & Bassett, PNAS 2018."""
    iu = np.triu_indices_from(W, k=1)
    d = np.linalg.norm(xyz[iu[0]] - xyz[iu[1]], axis=-1)
    hemi = hemisphere(names)
    same_hemi = hemi[iu[0]] == hemi[iu[1]]
    vals = W[iu].copy()
    out = vals.copy()
    for group in (same_hemi, ~same_hemi):
        edges = np.quantile(d[group], np.linspace(0, 1, n_bins + 1))
        b = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, n_bins - 1)
        for k in range(n_bins):
            sel = np.flatnonzero(group & (b == k))
            out[sel] = vals[rng.permutation(sel)]
    M = np.zeros_like(W)
    M[iu] = out
    return M + M.T


def geometry_preserving_null(W: np.ndarray, xyz: np.ndarray, seed_idx: list[int],
                             names: list[str], p: Params, threshold: float,
                             n: int = 500, seed: int = 0,
                             genotype: str = "ε3/ε3",
                             stages: np.ndarray | None = None,
                             ordering: str = "arrival") -> NullResult:
    rng = np.random.default_rng(seed)
    stages = braak_stages(names) if stages is None else stages
    order = lambda M: spread_order(M, seed_idx, p, threshold, genotype, ordering)
    observed = braak_concordance(order(W), stages)
    samples = [braak_concordance(order(geometry_preserving_rewire(W, xyz, names, rng)), stages)
               for _ in range(n)]
    return NullResult(observed, np.array(samples))


def spread_order(W: np.ndarray, seed_idx: list[int], p: Params, threshold: float,
                 genotype: str = "ε3/ε3", ordering: str = "arrival") -> np.ndarray:
    """Per-region 'when does it get hit' score; lower = earlier.

    arrival  first step the load crosses `threshold`; regions that never cross
             tie for last. The README's default, tuned on DK-68.
    load     minus the load at the final step (more load = earlier). No
             threshold and no ties, so it is comparable across parcellations,
             where a fixed threshold is not (finer parcels plateau lower).
    """
    h = simulate(W, seed_idx, genotype, p)
    if ordering == "load":
        return -h[-1]
    if ordering != "arrival":
        raise ValueError(f"unknown ordering {ordering!r}")
    a = arrival_order(h, threshold).copy()
    fin = np.isfinite(a)
    a[~fin] = a[fin].max() + 1 if fin.any() else 0
    return a


def _partial(r_xy: float, r_xz: float, r_yz: float) -> float:
    return (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))


def connectome_vs_distance(W: np.ndarray, xyz: np.ndarray, seed_idx: list[int],
                           names: list[str], p: Params, threshold: float,
                           n_boot: int = 5000, seed: int = 0,
                           genotype: str = "ε3/ε3",
                           stages: np.ndarray | None = None,
                           clusters: list[str] | None = None,
                           ordering: str = "arrival") -> dict:
    """Compare three predictors of Braak order from the same seed:

    connectome  arrival step from the diffusion model
    distance    Euclidean distance from the seed (no model, no parameters)
    combined    sum of the two rank orders, equally weighted (nothing fitted)

    Returns each ρ, the partial correlation of each with Braak stage controlling
    for the other, and bootstrap 95% CIs on the differences. The bootstrap
    resamples the 34 bilateral region pairs (left and right of a region are not
    independent observations). For other atlases pass `stages` (Braak stage per
    region) and `clusters` (the unit to resample, e.g. each parcel's parent DK
    region), so finer atlases do not get narrower CIs from correlated parcels.
    """
    stages = braak_stages(names) if stages is None else stages
    a = spread_order(W, seed_idx, p, threshold, genotype, ordering)
    dist = distance_from_seed(xyz, seed_idx)
    combined = _rank(a) + _rank(dist)

    rho = {"connectome": braak_concordance(a, stages),
           "distance": braak_concordance(dist, stages),
           "combined": braak_concordance(combined, stages)}
    r_cd = braak_concordance(a, dist)   # Spearman between the two orders
    partial = {"connectome | distance": _partial(rho["connectome"], r_cd, rho["distance"]),
               "distance | connectome": _partial(rho["distance"], r_cd, rho["connectome"])}

    # bootstrap over bilateral pairs
    base = clusters if clusters is not None else [n.split("_", 1)[1] if "_" in n else n
                                                  for n in names]
    pairs = sorted(set(base))
    members = [np.flatnonzero(np.array(base) == b) for b in pairs]
    rng = np.random.default_rng(seed)
    d_cd, d_comb = [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(pairs), len(pairs))])
        s = stages[idx]
        rd = braak_concordance(dist[idx], s)
        d_cd.append(braak_concordance(a[idx], s) - rd)
        d_comb.append(braak_concordance(combined[idx], s) - rd)
    ci = lambda x: tuple(float(v) for v in np.nanpercentile(x, [2.5, 97.5]))
    return {"rho": rho, "order_agreement": r_cd, "partial": partial,
            "ci_connectome_minus_distance": ci(d_cd),
            "ci_combined_minus_distance": ci(d_comb),
            "n_boot": n_boot}
