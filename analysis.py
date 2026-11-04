"""Two analyses that go beyond reproducing the published model.

1. Seed scan — seed every region in turn and ask how Braak concordance depends
   on where the cascade starts. If ρ falls off with the seed's own Braak stage,
   the model is saying the entorhinal origin is not just one good choice but the
   only region from which the wiring reproduces the clinical order.

2. Knockout scan — remove one region at a time (zero its row and column, i.e. a
   virtual lesion) and re-run from the entorhinal seed. Regions whose removal
   hurts concordance most are the bridges the cascade depends on; regions whose
   removal helps are shortcuts that let pathology jump the Braak queue.

Both reuse simulate → arrival_order → braak_concordance, so there is no second
model to drift from the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from connectome import region_lobe
from model import Params, simulate, arrival_order, braak_stages, braak_concordance

ROMAN = {1: "I", 3: "III", 4: "IV", 5: "V", 6: "VI"}


def _label(s: float) -> str:
    return "—" if np.isnan(s) else ROMAN.get(int(s), "?")


def seed_scan(W: np.ndarray, names: list[str], p: Params, threshold: float,
              genotype: str = "ε3/ε3", bilateral: bool = True) -> pd.DataFrame:
    """Braak concordance for every possible seed. With `bilateral`, each base
    region (e.g. 'entorhinal') is seeded in both hemispheres and reported once."""
    stages = braak_stages(names)
    rows = []
    if bilateral:
        bases = sorted({n.split("_", 1)[1] for n in names})
        groups = [(b, [i for i, n in enumerate(names) if n.endswith("_" + b)]) for b in bases]
    else:
        groups = [(n, [i]) for i, n in enumerate(names)]
    for label, idx in groups:
        a = arrival_order(simulate(W, idx, genotype, p), threshold)
        rows.append({
            "seed": label,
            "seed Braak stage": _label(stages[idx[0]]),
            "seed stage (num)": stages[idx[0]],
            "lobe": region_lobe(names[idx[0]]),
            "regions reached": int(np.isfinite(a).sum()),
            "ρ": braak_concordance(a, stages),
        })
    df = pd.DataFrame(rows).sort_values("ρ", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df


def seed_stage_trend(scan: pd.DataFrame) -> tuple[float, float]:
    """Spearman correlation between a seed's Braak stage and the ρ it produces,
    and the mean ρ for stage-I/III seeds vs stage-V/VI seeds."""
    d = scan.dropna(subset=["seed stage (num)"])
    from model import _rank
    x, y = _rank(d["seed stage (num)"].to_numpy()), _rank(d["ρ"].to_numpy())
    x, y = x - x.mean(), y - y.mean()
    r = float((x * y).sum() / np.sqrt((x ** 2).sum() * (y ** 2).sum()))
    early = d[d["seed stage (num)"] <= 3]["ρ"].mean()
    late = d[d["seed stage (num)"] >= 5]["ρ"].mean()
    return r, float(early - late)


def knockout_scan(W: np.ndarray, names: list[str], seed_idx: list[int], p: Params,
                  threshold: float, genotype: str = "ε3/ε3",
                  bilateral: bool = True) -> pd.DataFrame:
    """Δρ when each region (bilateral pair by default) is lesioned out of the
    connectome. Seed regions are skipped."""
    stages = braak_stages(names)
    base = braak_concordance(arrival_order(simulate(W, seed_idx, genotype, p), threshold), stages)
    if bilateral:
        bases = sorted({n.split("_", 1)[1] for n in names})
        groups = [(b, [i for i, n in enumerate(names) if n.endswith("_" + b)]) for b in bases]
    else:
        groups = [(n, [i]) for i, n in enumerate(names)]
    rows = []
    for label, idx in groups:
        if any(i in seed_idx for i in idx):
            continue
        Wk = W.copy()
        Wk[idx, :] = 0.0
        Wk[:, idx] = 0.0
        a = arrival_order(simulate(Wk, seed_idx, genotype, p), threshold)
        # score only the surviving regions: the lesioned one can never arrive,
        # and counting it as "last" would penalise early-stage removals by
        # construction rather than by their role in the cascade
        st = stages.copy()
        st[idx] = np.nan
        rho = braak_concordance(a, st)
        rows.append({
            "removed": label,
            "Braak stage": _label(stages[idx[0]]),
            "lobe": region_lobe(names[idx[0]]),
            "strength": float(W[idx].sum()),
            "regions reached": int(np.isfinite(a).sum()),
            "ρ": rho,
            "Δρ": rho - base,
        })
    df = pd.DataFrame(rows).sort_values("Δρ").reset_index(drop=True)
    df.index = df.index + 1
    df.attrs["baseline"] = base
    return df
