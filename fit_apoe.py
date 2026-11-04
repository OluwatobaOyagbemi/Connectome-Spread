"""Fit β and κ per APOE group from a regional tau-PET table.

    python fit_apoe.py data/tau_pet.csv

Input CSV (see data/tau_pet_template.csv): one row per Desikan-Killiany region
named exactly as in data/hcp_dk68_sc.csv (L_entorhinal, R_fusiform, …), and one
column per group/timepoint. Column names: <group>_baseline and, if available,
<group>_followup, e.g. e4neg_baseline, e4neg_followup, e4pos_baseline,
e4pos_followup. Values are tau-PET SUVR (any consistent reference region).
Regions you don't have can be left blank; they are excluded from the fit.

Output: fitted β, κ, β/κ per group with bootstrap 95% CIs, the fitted-vs-
observed correlation, and the implied genotype multipliers relative to the
reference group — the numbers that would replace the assumed ones in
model.APOE_PROFILES.

If you don't yet have data, run with --demo to see the pipeline on synthetic
"observations" generated from the model's own ε3/ε3 and ε4/ε4 profiles.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from connectome import hcp_connectome
from fit import fit_cross_sectional, fit_longitudinal, bootstrap_ci
from model import APOE_PROFILES, Params, simulate


def load_table(path: str, names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    missing = [n for n in names if n not in df.index]
    if missing:
        sys.exit(f"{len(missing)} regions missing from {path}, e.g. {missing[:3]}")
    return df.loc[names]


def demo_table(W, seed_idx, names) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = {}
    for label, g in (("e4neg", "ε3/ε3"), ("e4pos", "ε4/ε4")):
        h = simulate(W, seed_idx, g, Params(steps=360))
        cols[f"{label}_baseline"] = h[300] * rng.lognormal(0, 0.08, len(names))
        cols[f"{label}_followup"] = h[360] * rng.lognormal(0, 0.08, len(names))
    return pd.DataFrame(cols, index=names)


def fit_group(W, seed_idx, df, group, gap, n_boot):
    base = df[f"{group}_baseline"].to_numpy(dtype=float)
    have_follow = f"{group}_followup" in df.columns
    present = ~np.isnan(base)
    if have_follow:
        foll = df[f"{group}_followup"].to_numpy(dtype=float)
        present &= ~np.isnan(foll)
        base, foll = np.nan_to_num(base), np.nan_to_num(foll)
        fn = lambda mask: fit_longitudinal(W, seed_idx, base, foll, gap=gap, mask=mask & present)
    else:
        base = np.nan_to_num(base)
        fn = lambda mask: fit_cross_sectional(W, seed_idx, base, mask=mask & present)
    point = fn(np.ones(len(base), dtype=bool))
    ci = bootstrap_ci(lambda mask: fn(mask), n=n_boot, n_regions=len(base))
    return point, ci, have_follow, int(present.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--gap", type=int, default=60,
                    help="follow-up interval in model steps (default 60)")
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--reference", default="e4neg")
    a = ap.parse_args()

    conn = hcp_connectome()
    names, W = list(conn.index), conn.values
    seed_idx = [i for i, n in enumerate(names) if n.endswith("entorhinal")]

    if a.demo:
        df = demo_table(W, seed_idx, names)
        print("DEMO: synthetic observations from the model's own ε3/ε3 (e4neg) and "
              "ε4/ε4 (e4pos) profiles. True multipliers: β ×1.80, κ ×0.50.\n")
    elif a.csv:
        df = load_table(a.csv, names)
    else:
        sys.exit("give a CSV or --demo")

    groups = sorted({c.rsplit("_", 1)[0] for c in df.columns})
    results = {}
    for g in groups:
        point, ci, longit, n = fit_group(W, seed_idx, df, g, a.gap, a.boot)
        results[g] = (point, ci)
        kind = "longitudinal" if longit else "cross-sectional (β/κ ratio only is reliable)"
        print(f"{g}  [{kind}, {n} regions]")
        print(f"  β = {point.beta:.4f}  95% CI [{ci['beta'][0]:.4f}, {ci['beta'][2]:.4f}]")
        print(f"  κ = {point.kappa:.4f}  95% CI [{ci['kappa'][0]:.4f}, {ci['kappa'][2]:.4f}]")
        print(f"  β/κ = {point.ratio:.1f}  95% CI [{ci['ratio'][0]:.1f}, {ci['ratio'][2]:.1f}]")
        print(f"  fit: r = {point.r:.3f} between simulated and observed pattern\n")

    if a.reference in results and len(results) > 1:
        ref_pt, ref_ci = results[a.reference]
        print(f"Implied multipliers relative to {a.reference} "
              f"(compare with model.APOE_PROFILES):")
        for g, (pt, ci) in results.items():
            if g == a.reference:
                continue
            # bootstrap ratio of samples for a CI on the multiplier
            mb = ci["samples"][:, 0] / ref_ci["samples"][:, 0]
            mk = ci["samples"][:, 1] / ref_ci["samples"][:, 1]
            print(f"  {g}: β ×{pt.beta / ref_pt.beta:.2f} "
                  f"[{np.percentile(mb, 2.5):.2f}, {np.percentile(mb, 97.5):.2f}]   "
                  f"κ ×{pt.kappa / ref_pt.kappa:.2f} "
                  f"[{np.percentile(mk, 2.5):.2f}, {np.percentile(mk, 97.5):.2f}]")
        print("\nAssumed in the app:",
              {g: (v["beta"], v["kappa"]) for g, v in APOE_PROFILES.items()})


if __name__ == "__main__":
    main()
