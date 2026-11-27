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

from connectome import hcp_connectome, hcp_centroids
from model import (APOE_PROFILES, Params, simulate, arrival_order, braak_stages,
                   braak_concordance, beta_sweep)
from validation import (shuffled_braak_null, rewired_connectome_null, seed_control,
                        threshold_control, geometry_preserving_null, connectome_vs_distance)
from analysis import seed_scan, seed_stage_trend, knockout_scan
from resolution import resolution_series, seed_sensitivity
from vulnerability import (load_expression, predictors, spin_permutations, genome_scan,
                           gene_vs_braak)

ROOT = Path(__file__).parent
FIG_DIR = ROOT / "docs" / "figures"
THRESHOLD = 0.10


def main(quick: bool = False) -> dict:
    conn = hcp_connectome()
    names, W = list(conn.index), conn.values
    seed_idx = [i for i, n in enumerate(names) if n.endswith("entorhinal")]
    p = Params()
    stages = braak_stages(names)
    xyz = hcp_centroids(names)

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

    # ── 2b. Geometry: does the wiring beat distance? ─────────────────────────
    n_geo = 50 if quick else 500
    geo = geometry_preserving_null(W, xyz, seed_idx, names, p, THRESHOLD, n=n_geo)
    print(f"Geometry-preserving null ({n_geo} graphs): mean {geo.mean:.2f}, "
          f"95th pct {geo.p95:.2f}, p = {geo.p_value:.4f}")
    n_boot = 500 if quick else 5000
    cvd = {t: connectome_vs_distance(W, xyz, seed_idx, names, p, t, n_boot=n_boot)
           for t in (THRESHOLD, 0.02)}
    print("\nConnectome vs straight-line distance from entorhinal (ε3/ε3)")
    for t, c in cvd.items():
        r, pc = c["rho"], c["partial"]
        lo, hi = c["ci_connectome_minus_distance"]
        clo, chi = c["ci_combined_minus_distance"]
        print(f"  threshold {t:.2f}: connectome {r['connectome']:.2f}  distance {r['distance']:.2f}  "
              f"combined {r['combined']:.2f}   orders agree ρ = {c['order_agreement']:.2f}")
        print(f"      partial ρ: connectome|distance {pc['connectome | distance']:.2f}, "
              f"distance|connectome {pc['distance | connectome']:.2f}")
        print(f"      95% CI connectome − distance [{lo:+.2f}, {hi:+.2f}]; "
              f"combined − distance [{clo:+.2f}, {chi:+.2f}]")

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

    # ── 5. Seed scan + lesion scan ───────────────────────────────────────────
    scan = seed_scan(W, names, p, THRESHOLD, xyz=xyz)
    trend_r, early_minus_late = seed_stage_trend(scan)
    dtrend_r, dearly_minus_late = seed_stage_trend(scan, "ρ (distance)")
    dist_best = scan.sort_values("ρ (distance)", ascending=False).seed.iloc[0]
    print(f"\nSeed scan ({len(scan)} bilateral seeds): ρ vs seed's Braak stage, "
          f"Spearman {trend_r:+.2f}; early (I/III) minus late (V/VI) mean ρ = {early_minus_late:+.2f}")
    print("  top 5:   " + ", ".join(f"{r.seed} {r['ρ']:+.2f}" for _, r in scan.head(5).iterrows()))
    print("  bottom 5:" + ", ".join(f" {r.seed} {r['ρ']:+.2f}" for _, r in scan.tail(5).iterrows()))
    print(f"  distance-only baseline: Spearman {dtrend_r:+.2f}, early minus late {dearly_minus_late:+.2f}, "
          f"best seed {dist_best}")
    ko = knockout_scan(W, names, seed_idx, p, THRESHOLD)
    print(f"\nLesion scan (baseline ρ = {ko.attrs['baseline']:.3f}), largest losses:")
    for _, r in ko.head(5).iterrows():
        print(f"  remove {r.removed:18} stage {r['Braak stage']:4} Δρ = {r['Δρ']:+.3f}")
    print("  largest gains: " + ", ".join(f"{r.removed} {r['Δρ']:+.3f}" for _, r in ko.tail(3).iterrows()))

    # ── 6. Resolution: does a finer parcellation help the connectome? ───────
    res = resolution_series(n_null=40 if quick else 200, n_boot=300 if quick else 2000)
    print("\nResolution series (final-load ordering, entorhinal seed, ε3/ε3)")
    print(f"  {'atlas':13} {'N':>4} {'conn':>5} {'dist':>5} {'both':>5} {'partial':>8} "
          f"{'conn−dist 95% CI':>18} {'both−dist 95% CI':>18} {'geo p':>6}")
    for r in res:
        lo, hi = r["ci_connectome_minus_distance"]
        clo, chi = r["ci_combined_minus_distance"]
        print(f"  {r['atlas']:13} {r['n_parcels']:4d} {r['rho']['connectome']:5.2f} {r['rho']['distance']:5.2f} "
              f"{r['rho']['combined']:5.2f} {r['partial']['connectome | distance']:8.2f} "
              f"   [{lo:+.2f}, {hi:+.2f}]    [{clo:+.2f}, {chi:+.2f}] {r['geometry_null']['p']:6.3f}")
    sens = {a: seed_sensitivity(a) for a in ("schaefer_100", "schaefer_400", "glasser_360")}
    for a, rows in sens.items():
        print(f"  seed sensitivity {a}: " + "; ".join(
            f"{r['seed definition']} ({r['n seeds']}): conn {r['connectome']:.2f} dist {r['distance']:.2f} "
            f"partial {r['partial']:.2f}" for r in rows))

    # ── 7. Regional vulnerability: gene expression beyond wiring + distance ──
    expr, Pv = load_expression(), predictors()
    perms = spin_permutations(n=500 if quick else 5000)
    genome = genome_scan(expr, Pv)
    genes = {g: gene_vs_braak(g, expr, Pv, perms, genome, n_boot=500 if quick else 5000)
             for g in ("MAPT", "APOE")}
    print(f"\nGene expression vs Braak order (34 left-hemisphere regions, {len(perms)} spins, "
          f"{len(genome['genes']):,} genes)")
    print(f"  connectome + distance alone: ρ = {genes['MAPT']['connectome_plus_distance']:.2f}")
    for g, r in genes.items():
        lo, hi = r["ci_gain"]
        jlo, jhi = r["jackknife_partial"]
        print(f"  {g:5} alone ρ {r['rho']:+.2f} (spin p {r['spin_p']:.3f}, genome pct {r['genome_percentile']:.0f}) | "
              f"beyond wiring+distance ρ {r['partial']:+.2f} (spin p {r['spin_p_partial']:.3f}, "
              f"genome pct {r['genome_percentile_partial']:.0f}, jackknife [{jlo:.2f}, {jhi:.2f}])")
        print(f"        adding {g}: {r['connectome_plus_distance']:.2f} → {r['with_gene']:.2f}, "
              f"gain 95% CI [{lo:+.2f}, {hi:+.2f}]")

    results = {
        "defaults": {"beta": p.beta, "kappa": p.kappa, "steps": p.steps,
                     "threshold": THRESHOLD, "seed": "bilateral entorhinal"},
        "genotype_table": table,
        "shuffled_braak_null": {"n": n_shuf, "mean": round(shuf.mean, 2),
                                "p95": round(shuf.p95, 2), "p": round(shuf.p_value, 4)},
        "rewired_connectome_null": {"n": n_rew, "mean": round(rew.mean, 2),
                                    "p95": round(rew.p95, 2), "p": round(rew.p_value, 4)},
        "geometry_preserving_null": {"n": n_geo, "mean": round(geo.mean, 2),
                                     "p95": round(geo.p95, 2), "p": round(geo.p_value, 4)},
        "connectome_vs_distance": {str(t): _round(c) for t, c in cvd.items()},
        "seed_control": [{**r, "ρ": round(r["ρ"], 2)} for r in seeds],
        "threshold_control": [{**r, "ρ": round(r["ρ"], 2)} for r in thr],
        "seed_scan": {"trend_spearman": round(trend_r, 2),
                      "early_minus_late": round(early_minus_late, 2),
                      "distance_trend_spearman": round(dtrend_r, 2),
                      "distance_early_minus_late": round(dearly_minus_late, 2),
                      "distance_best_seed": dist_best,
                      "top": [(r.seed, round(r["ρ"], 2)) for _, r in scan.head(5).iterrows()],
                      "bottom": [(r.seed, round(r["ρ"], 2)) for _, r in scan.tail(5).iterrows()]},
        "lesion_scan": {"baseline": round(float(ko.attrs["baseline"]), 3),
                        "largest_losses": [(r.removed, r["Braak stage"], round(r["Δρ"], 3)) for _, r in ko.head(5).iterrows()],
                        "largest_gains": [(r.removed, r["Braak stage"], round(r["Δρ"], 3)) for _, r in ko.tail(3).iterrows()]},
        "resolution": [_round({k: v for k, v in r.items()}) for r in res],
        "resolution_seed_sensitivity": {a: [_round(r) for r in rows] for a, rows in sens.items()},
        "vulnerability": {g: _round(r) for g, r in genes.items()},
        "beta_window": {"min": float(win.min()), "max": float(win.max()),
                        "peak": float(peak), "peak_rho": round(float(np.nanmax(sweep["ε3/ε3"])), 2)},
    }
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    make_figure(names, hists["ε3/ε3"], a_ref, stages, shuf, rew, geo, cvd[THRESHOLD],
                FIG_DIR / "validation.png")
    make_resolution_figure(res, FIG_DIR / "resolution.png")
    make_vulnerability_figure(genome, genes, perms, expr, Pv, FIG_DIR / "vulnerability.png")
    print(f"\nWrote {FIG_DIR / 'results.json'}, {FIG_DIR / 'validation.png'} and "
          f"{FIG_DIR / 'resolution.png'} and vulnerability.png")
    return results


def _round(x):
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [_round(v) for v in x]
    return round(float(x), 3) if isinstance(x, (float, np.floating)) else x


def make_figure(names, hist, arrival, stages, shuf, rew, geo, cvd, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, grid = "#52514e", "#d5dce5"
    stage_color = {1: "#e34948", 3: "#eb6834", 4: "#eda100", 5: "#2a78d6", 6: "#6250d6"}
    roman = {1: "I", 3: "III", 4: "IV", 5: "V", 6: "VI"}

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), gridspec_kw={"width_ratios": [1.4, 1, 1, 1]})
    fig.patch.set_facecolor("white")
    BINS = np.linspace(-0.6, 0.8, 50)

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
    ax.set_title(f"A  Arrival order vs Braak stage   ρ = {shuf.observed:.2f}", loc="left",
                 fontsize=11, color=ink)
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)

    # B, C: null distributions
    for ax, null, title in ((axes[1], shuf, "B  Shuffled Braak labels"),
                            (axes[2], geo, "C  Geometry-preserving rewire")):
        ax.hist(null.samples[~np.isnan(null.samples)], bins=BINS, color="#cfe0f7",
                edgecolor="white", label="null")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        ax.axvline(null.observed, color="#e34948", lw=2)
        ax.text(null.observed - 0.02, ax.get_ylim()[1] * 0.92, "observed", color="#e34948",
                ha="right", fontsize=9)
        ax.axvline(null.p95, color=ink, lw=1, ls=":")
        ax.text(null.p95 + 0.02, ax.get_ylim()[1] * 0.75, "null 95th pct", color=ink,
                fontsize=8)
        ax.set_xlim(-0.6, 0.8); ax.set_xlabel("Braak concordance ρ")
        ptxt = "p < 0.001" if null.p_value < 0.001 else f"p = {null.p_value:.3f}"
        ax.set_title(f"{title}   {ptxt}", loc="left", fontsize=11, color=ink)

    axes[2].hist(rew.samples[~np.isnan(rew.samples)], bins=BINS, color="#e6e8ec",
                 edgecolor="white", zorder=0)
    axes[2].text(-0.55, axes[2].get_ylim()[1] * 0.92,
                 f"grey: full rewire,\nno geometry (p95 {rew.p95:.2f})", fontsize=8, color=ink, va="top")

    # D: connectome vs distance vs combined
    ax = axes[3]
    r = cvd["rho"]
    labels = ["connectome\ndiffusion", "distance\nfrom seed", "both\n(rank sum)"]
    vals = [r["connectome"], r["distance"], r["combined"]]
    bars = ax.bar(labels, vals, color=["#2a78d6", "#a3a8b0", "#6250d6"], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center",
                fontsize=10, color=ink)
    ax.axhline(geo.p95, color=ink, lw=1, ls=":")
    ax.text(0, geo.p95 - 0.07, "geometry\nnull p95", fontsize=7.5, color="white", ha="center")
    ax.set_ylim(0, 0.9); ax.set_ylabel("Braak concordance ρ")
    lo, hi = cvd["ci_connectome_minus_distance"]
    ax.set_title("D  Wiring vs distance", loc="left", fontsize=11, color=ink)
    ax.text(0.5, 0.86, f"connectome − distance\n95% CI [{lo:+.2f}, {hi:+.2f}]", fontsize=8,
            color=ink, ha="center", transform=ax.transAxes)

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


def make_vulnerability_figure(genome, genes, perms, expr, P, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vulnerability import partial_spearman

    ink, grid = "#52514e", "#d5dce5"
    col = {"MAPT": "#e34948", "APOE": "#eda100"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.hist(genome["partial"], bins=60, color="#cfe0f7", edgecolor="white")
    for g, r in genes.items():
        ax.axvline(r["partial"], color=col[g], lw=2)
        ax.text(r["partial"] + 0.01, ax.get_ylim()[1] * (0.92 if g == "MAPT" else 0.78),
                f"{g}  {r['genome_percentile_partial']:.0f}th pct", color=col[g], fontsize=9)
    ax.set_xlabel("partial ρ with Braak order, controlling wiring + distance")
    ax.set_ylabel("genes")
    ax.set_title(f"A  All {len(genome['genes']):,} stable genes", loc="left", fontsize=11, color=ink)

    ax = axes[1]
    x = -expr.loc[P["names"], "MAPT"].to_numpy()
    null = [partial_spearman(x[q], P["stages"], [P["connectome"], P["distance"]]) for q in perms]
    r = genes["MAPT"]
    ax.hist(null, bins=50, color="#e6e8ec", edgecolor="white")
    ax.axvline(r["partial"], color=col["MAPT"], lw=2)
    ax.text(r["partial"] - 0.02, ax.get_ylim()[1] * 0.9, f"observed\np = {r['spin_p_partial']:.3f}",
            color=col["MAPT"], fontsize=9, ha="right")
    ax.set_xlabel("partial ρ of the rotated MAPT map")
    ax.set_title(f"B  MAPT vs {len(perms):,} spatial rotations of itself", loc="left", fontsize=11, color=ink)

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


def make_resolution_figure(res, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, grid = "#52514e", "#d5dce5"
    labels = [f"{r['atlas'].replace('_', ' ').replace('dk', 'DK').replace('schaefer', 'Schaefer').replace('glasser', 'Glasser')}"
              for r in res]
    x = np.arange(len(res))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    series = [("connectome", "connectome diffusion", "#2a78d6", "o"),
              ("distance", "distance from seed", "#8a8f98", "s"),
              ("combined", "both (rank sum)", "#6250d6", "D")]
    for key, lab, col, mk in series:
        ax.plot(x, [r["rho"][key] for r in res], marker=mk, color=col, lw=1.8, ms=7, label=lab)
    ax.plot(x, [r["geometry_null"]["p95"] for r in res], color=ink, lw=1, ls=":",
            label="geometry-preserving null, 95th pct")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 0.85); ax.set_ylabel("Braak concordance ρ")
    ax.set_title("A  Same HCP data, six parcellations", loc="left", fontsize=11, color=ink)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)

    ax = axes[1]
    for off, key, col, lab in ((-0.15, "ci_connectome_minus_distance", "#2a78d6", "connectome − distance"),
                               (0.15, "ci_combined_minus_distance", "#6250d6", "both − distance")):
        pt = [r["rho"]["connectome" if "connectome_minus" in key else "combined"] - r["rho"]["distance"]
              for r in res]
        lo = [r[key][0] for r in res]; hi = [r[key][1] for r in res]
        ax.errorbar(x + off, pt, yerr=[np.subtract(pt, lo), np.subtract(hi, pt)], fmt="o",
                    color=col, ms=6, capsize=3, lw=1.4, label=lab)
    ax.axhline(0, color=ink, lw=1)
    ax.set_ylim(-0.5, 0.6)
    ax.set_xticks(x); ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=8)
    ax.set_ylabel("Δρ vs distance alone (95% CI)")
    ax.set_title("B  Gain over distance, bootstrap 95% CI", loc="left", fontsize=11, color=ink)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)

    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.tick_params(colors=ink, labelsize=9)
        ax.yaxis.label.set_color(ink)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(quick=ap.parse_args().quick)
