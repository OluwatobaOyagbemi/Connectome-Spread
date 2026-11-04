"""Reproduce every number in the README from scratch.

    python reproduce.py            # prints the tables, writes docs/figures/
    python reproduce.py --quick    # fewer null samples (CI / smoke test)

Writes:
  docs/figures/validation.png   the README figure (arrival order + nulls)
  docs/figures/results.json     every reported number, machine-readable
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from connectome import hcp_connectome
from model import (APOE_PROFILES, Params, simulate, arrival_order, braak_stages,
                   braak_concordance, beta_sweep)
from validation import (shuffled_braak_null, rewired_connectome_null, seed_control,
                        threshold_control)

ROOT = Path(__file__).parent
FIG_DIR = ROOT / "docs" / "figures"
THRESHOLD = 0.10


def main(quick: bool = False) -> dict:
    conn = hcp_connectome()
    names, W = list(conn.index), conn.values
    seed_idx = [i for i, n in enumerate(names) if n.endswith("entorhinal")]
    p = Params()
    stages = braak_stages(names)

    # ── 1. Genotype table ────────────────────────────────────────────────────
    hists = {g: simulate(W, seed_idx, g, p) for g in APOE_PROFILES}
    table = []
    for g, h in hists.items():
        a = arrival_order(h, THRESHOLD)
        table.append({"genotype": g, "total load": round(float(h[-1].sum()), 1),
                      "regions reached": int(np.isfinite(a).sum()),
                      "rho": round(braak_concordance(a, stages), 2)})
    print("\nGenotype table (entorhinal seed, β 0.05, κ 0.01, 300 steps, threshold 0.10)")
    print(f"{'genotype':10} {'load':>6} {'reached':>8} {'ρ':>6}")
    for r in table:
        print(f"{r['genotype']:10} {r['total load']:6.1f} {r['regions reached']:8d} {r['rho']:6.2f}")

    # ── 2. Nulls ─────────────────────────────────────────────────────────────
    a_ref = arrival_order(hists["ε3/ε3"], THRESHOLD)
    n_shuf, n_rew = (200, 30) if quick else (2000, 200)
    shuf = shuffled_braak_null(a_ref, stages, n=n_shuf)
    rew = rewired_connectome_null(W, seed_idx, names, p, THRESHOLD, n=n_rew)
    print(f"\nObserved ρ (ε3/ε3): {shuf.observed:.2f}")
    print(f"Shuffled-Braak null   ({n_shuf} perms): mean {shuf.mean:.2f}, "
          f"95th pct {shuf.p95:.2f}, p = {shuf.p_value:.4f}")
    print(f"Rewired-connectome null ({n_rew} graphs): mean {rew.mean:.2f}, "
          f"95th pct {rew.p95:.2f}, p = {rew.p_value:.4f}")

    # ── 3. Seed + threshold controls ─────────────────────────────────────────
    seeds = seed_control(W, names, p, THRESHOLD)
    print("\nSeed control (bilateral seed, ε3/ε3)")
    for r in seeds:
        print(f"  {r['seed']:18} stage {r['Braak stage of seed']:4} "
              f"reached {r['regions reached']:3d}   ρ = {r['ρ']:+.2f}")
    thr = threshold_control(W, seed_idx, names, p)
    print("\nThreshold control (ε3/ε3)")
    for r in thr:
        print(f"  threshold {r['threshold']:.2f}  reached {r['regions reached']:3d}   ρ = {r['ρ']:.2f}")

    # ── 4. β window ──────────────────────────────────────────────────────────
    betas = np.round(np.linspace(0.01, 0.20, 20), 3)
    sweep = beta_sweep(W, seed_idx, names, betas, p, THRESHOLD)
    win = betas[np.nan_to_num(sweep["ε3/ε3"]) >= 0.5]
    peak = betas[np.nanargmax(sweep["ε3/ε3"])]
    print(f"\nβ window (ε3/ε3): ρ ≥ 0.5 for β ∈ [{win.min():.2f}, {win.max():.2f}], "
          f"peak at β = {peak:.2f} (ρ = {np.nanmax(sweep['ε3/ε3']):.2f})")

    results = {
        "defaults": {"beta": p.beta, "kappa": p.kappa, "steps": p.steps,
                     "threshold": THRESHOLD, "seed": "bilateral entorhinal"},
        "genotype_table": table,
        "shuffled_braak_null": {"n": n_shuf, "mean": round(shuf.mean, 2),
                                "p95": round(shuf.p95, 2), "p": round(shuf.p_value, 4)},
        "rewired_connectome_null": {"n": n_rew, "mean": round(rew.mean, 2),
                                    "p95": round(rew.p95, 2), "p": round(rew.p_value, 4)},
        "seed_control": [{**r, "ρ": round(r["ρ"], 2)} for r in seeds],
        "threshold_control": [{**r, "ρ": round(r["ρ"], 2)} for r in thr],
        "beta_window": {"min": float(win.min()), "max": float(win.max()),
                        "peak": float(peak), "peak_rho": round(float(np.nanmax(sweep["ε3/ε3"])), 2)},
    }
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    make_figure(names, hists["ε3/ε3"], a_ref, stages, shuf, rew, seeds, FIG_DIR / "validation.png")
    print(f"\nWrote {FIG_DIR / 'results.json'} and {FIG_DIR / 'validation.png'}")
    return results


def make_figure(names, hist, arrival, stages, shuf, rew, seeds, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, grid = "#52514e", "#d5dce5"
    stage_color = {1: "#e34948", 3: "#eb6834", 4: "#eda100", 5: "#2a78d6", 6: "#6250d6"}
    roman = {1: "I", 3: "III", 4: "IV", 5: "V", 6: "VI"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    fig.patch.set_facecolor("white")

    # A: arrival step vs Braak stage, one dot per reached region
    ax = axes[0]
    fin = np.isfinite(arrival) & ~np.isnan(stages)
    rng = np.random.default_rng(1)
    for s in sorted(set(stages[fin].astype(int))):
        m = fin & (stages == s)
        ax.scatter(stages[m] + rng.uniform(-0.18, 0.18, m.sum()), arrival[m],
                   s=42, color=stage_color[s], alpha=0.9, edgecolor="white", linewidth=0.6,
                   label=f"Braak {roman[s]}")
    not_reached = (~np.isfinite(arrival)) & ~np.isnan(stages)
    ax.scatter(stages[not_reached] + rng.uniform(-0.18, 0.18, not_reached.sum()),
               np.full(not_reached.sum(), hist.shape[0] + 15), s=18, color="#c5cad2",
               marker="v", label="not reached")
    ax.set_xticks([1, 3, 4, 5, 6]); ax.set_xticklabels(["I", "III", "IV", "V", "VI"])
    ax.set_xlabel("Braak tau stage of region"); ax.set_ylabel("simulated arrival step")
    ax.set_title(f"Arrival order follows Braak order   ρ = {shuf.observed:.2f}", loc="left",
                 fontsize=11, color=ink)
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)

    # B, C: null distributions
    for ax, null, title in ((axes[1], shuf, "Shuffled Braak labels"),
                            (axes[2], rew, "Rewired connectome")):
        ax.hist(null.samples[~np.isnan(null.samples)], bins=30, color="#cfe0f7",
                edgecolor="white")
        ax.axvline(null.observed, color="#e34948", lw=2)
        ax.text(null.observed - 0.02, ax.get_ylim()[1] * 0.92, "observed", color="#e34948",
                ha="right", fontsize=9)
        ax.axvline(null.p95, color=ink, lw=1, ls=":")
        ax.text(null.p95 + 0.02, ax.get_ylim()[1] * 0.75, "null 95th pct", color=ink,
                fontsize=8)
        ax.set_xlim(-0.6, 0.8); ax.set_xlabel("Braak concordance ρ")
        ptxt = "p < 0.001" if null.p_value < 0.001 else f"p = {null.p_value:.3f}"
        ax.set_title(f"{title}   {ptxt}", loc="left", fontsize=11, color=ink)

    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.tick_params(colors=ink, labelsize=9)
        ax.yaxis.label.set_color(ink); ax.xaxis.label.set_color(ink)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(quick=ap.parse_args().quick)
