"""Connectome Spread as an MCP server: the simulation as tools an AI agent can call.

    python mcp_server.py                 # stdio transport (Claude Desktop, Claude Code)
    python mcp_server.py --http          # streamable HTTP on :8000 (remote clients)

Claude Desktop config (claude_desktop_config.json):

    {"mcpServers": {"connectome-spread": {
        "command": "python", "args": ["/absolute/path/to/mcp_server.py"]}}}

Claude Code:   claude mcp add connectome-spread -- python /absolute/path/to/mcp_server.py

Every tool reuses the same simulate → arrival_order → braak_concordance path as
the app and the tests, so an agent gets the same numbers a person would.
"""

from __future__ import annotations

import sys

import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from analysis import knockout_scan, seed_scan, seed_stage_trend
from connectome import hcp_connectome, region_lobe
from model import (APOE_PROFILES, Params, arrival_order, beta_sweep, braak_concordance,
                   braak_stages, simulate)
from validation import rewired_connectome_null, shuffled_braak_null

server = MCPServer(
    "connectome-spread",
    instructions=(
        "Network-diffusion simulation of Alzheimer's tau spread on a real human "
        "connectome (68 Desikan-Killiany regions, HCP). Region names look like "
        "'L_entorhinal'; seeds are given by base name ('entorhinal' = both "
        "hemispheres). Braak concordance ρ is Spearman correlation between simulated "
        "arrival order and Braak tau stage (1 = perfect Braak order). Time steps are "
        "dimensionless. APOE multipliers are illustrative assumptions, not fitted."
    ),
)

_CONN = hcp_connectome()
NAMES: list[str] = list(_CONN.index)
W: np.ndarray = _CONN.values
STAGES = braak_stages(NAMES)
ROMAN = {1: "I", 3: "III", 4: "IV", 5: "V", 6: "VI"}


def _seed_idx(seed: str) -> list[int]:
    idx = [i for i, n in enumerate(NAMES) if n == seed or n.endswith("_" + seed)]
    if not idx:
        raise ToolError(f"unknown region '{seed}'; call list_regions()")
    return idx


def _genotype(g: str) -> str:
    g = g.replace("e", "ε").replace("E", "ε")
    if g not in APOE_PROFILES:
        raise ToolError(f"genotype must be one of {list(APOE_PROFILES)}")
    return g


def _stage(i: int) -> str:
    s = STAGES[i]
    return "—" if np.isnan(s) else ROMAN[int(s)]


@server.tool()
def list_regions() -> list[dict]:
    """The 68 cortical regions: name, lobe, Braak tau stage (I, III–VI; II is
    hippocampal and absent from the cortical atlas), and total connection strength."""
    return [{"region": n, "lobe": region_lobe(n), "braak_stage": _stage(i),
             "strength": round(float(W[i].sum()), 1)} for i, n in enumerate(NAMES)]


@server.tool()
def list_genotypes() -> dict:
    """APOE genotypes and the (illustrative) multipliers they apply to spread rate β
    and clearance rate κ."""
    return {g: {"effect": v["label"], "spread_x": v["beta"], "clearance_x": v["kappa"]}
            for g, v in APOE_PROFILES.items()}


@server.tool()
def simulate_spread(seed: str = "entorhinal", genotype: str = "e3/e3",
                    beta: float = 0.05, kappa: float = 0.01, steps: int = 300,
                    threshold: float = 0.10) -> dict:
    """Run the diffusion model from a seed region and return the arrival order of
    every region, its Braak stage, the final regional loads, and the Braak
    concordance ρ. Seed by base name ('entorhinal' seeds both hemispheres) or by
    full name ('L_entorhinal'). Steps are dimensionless model time."""
    idx = _seed_idx(seed)
    g = _genotype(genotype)
    h = simulate(W, idx, g, Params(beta=beta, kappa=kappa, steps=steps))
    arr = arrival_order(h, threshold)
    order = np.argsort(arr, kind="mergesort")
    arrivals = [{"region": NAMES[i], "arrival_step": int(arr[i]), "braak_stage": _stage(i),
                 "final_load": round(float(h[-1, i]), 4)}
                for i in order if np.isfinite(arr[i])]
    return {
        "seed": [NAMES[i] for i in idx], "genotype": g,
        "params": {"beta": beta, "kappa": kappa, "steps": steps, "threshold": threshold},
        "total_load": round(float(h[-1].sum()), 3),
        "regions_reached": len(arrivals),
        "braak_concordance_rho": round(braak_concordance(arr, STAGES), 3),
        "arrivals": arrivals,
        "not_reached": [NAMES[i] for i in range(len(NAMES)) if not np.isfinite(arr[i])],
    }


@server.tool()
def compare_genotypes(seed: str = "entorhinal", beta: float = 0.05, kappa: float = 0.01,
                      steps: int = 300, threshold: float = 0.10) -> list[dict]:
    """Total load, regions reached and Braak concordance for all five APOE genotypes
    from the same seed and parameters."""
    idx = _seed_idx(seed)
    out = []
    for g in APOE_PROFILES:
        h = simulate(W, idx, g, Params(beta=beta, kappa=kappa, steps=steps))
        arr = arrival_order(h, threshold)
        out.append({"genotype": g, "total_load": round(float(h[-1].sum()), 2),
                    "regions_reached": int(np.isfinite(arr).sum()),
                    "braak_concordance_rho": round(braak_concordance(arr, STAGES), 3)})
    return out


@server.tool()
def sweep_beta(seed: str = "entorhinal", kappa: float = 0.01, steps: int = 300,
               threshold: float = 0.10, n: int = 20) -> dict:
    """Braak concordance as a function of spread rate β (0.01–0.20) for every
    genotype, plus the β window in which ρ ≥ 0.5 for each."""
    idx = _seed_idx(seed)
    betas = np.round(np.linspace(0.01, 0.20, n), 3)
    sweep = beta_sweep(W, idx, NAMES, betas, Params(kappa=kappa, steps=steps), threshold)
    out = {"betas": betas.tolist(), "rho": {}, "window_rho_ge_0.5": {}}
    for g, vals in sweep.items():
        out["rho"][g] = [None if np.isnan(v) else round(float(v), 3) for v in vals]
        win = betas[np.nan_to_num(vals) >= 0.5]
        out["window_rho_ge_0.5"][g] = ([float(win.min()), float(win.max())] if len(win) else None)
    return out


@server.tool()
def run_controls(seed: str = "entorhinal", beta: float = 0.05, kappa: float = 0.01,
                 steps: int = 300, threshold: float = 0.10,
                 n_shuffle: int = 1000, n_rewire: int = 50) -> dict:
    """Is the Braak concordance better than chance? Shuffled-Braak-label null and
    rewired-connectome null (ε3/ε3), each with mean, 95th percentile and one-sided p."""
    idx = _seed_idx(seed)
    p = Params(beta=beta, kappa=kappa, steps=steps)
    arr = arrival_order(simulate(W, idx, "ε3/ε3", p), threshold)
    shuf = shuffled_braak_null(arr, STAGES, n=n_shuffle)
    rew = rewired_connectome_null(W, idx, NAMES, p, threshold, n=n_rewire)
    f = lambda r: {"mean": round(r.mean, 3), "p95": round(r.p95, 3), "p": round(r.p_value, 4)}
    return {"observed_rho": round(shuf.observed, 3),
            "shuffled_braak_null": {"n": n_shuffle, **f(shuf)},
            "rewired_connectome_null": {"n": n_rewire, **f(rew)}}


@server.tool()
def scan_seeds(genotype: str = "e3/e3", beta: float = 0.05, kappa: float = 0.01,
               steps: int = 300, threshold: float = 0.10) -> dict:
    """Seed every one of the 34 bilateral region pairs in turn and rank them by
    Braak concordance. Also returns the Spearman trend between a seed's own Braak
    stage and the ρ it produces."""
    scan = seed_scan(W, NAMES, Params(beta=beta, kappa=kappa, steps=steps), threshold,
                     _genotype(genotype))
    r, diff = seed_stage_trend(scan)
    return {"trend_seed_stage_vs_rho": round(r, 3), "early_minus_late_mean_rho": round(diff, 3),
            "ranking": [{"seed": s.seed, "braak_stage": s["seed Braak stage"], "lobe": s.lobe,
                         "regions_reached": int(s["regions reached"]), "rho": round(s["ρ"], 3)}
                        for _, s in scan.iterrows()]}


@server.tool()
def scan_lesions(seed: str = "entorhinal", genotype: str = "e3/e3", beta: float = 0.05,
                 kappa: float = 0.01, steps: int = 300, threshold: float = 0.10) -> dict:
    """Remove each region pair from the connectome in turn (virtual lesion), re-run
    from the seed, and report the change in Braak concordance scored over the
    surviving regions. Negative Δρ = the region is a bridge the Braak-like order
    depends on; positive = a shortcut."""
    ko = knockout_scan(W, NAMES, _seed_idx(seed), Params(beta=beta, kappa=kappa, steps=steps),
                       threshold, _genotype(genotype))
    return {"baseline_rho": round(float(ko.attrs["baseline"]), 3),
            "lesions_sorted_by_delta": [
                {"removed": r.removed, "braak_stage": r["Braak stage"], "lobe": r.lobe,
                 "strength": round(r.strength, 1), "rho": round(r["ρ"], 3),
                 "delta_rho": round(r["Δρ"], 3)} for _, r in ko.iterrows()]}


if __name__ == "__main__":
    if "--http" in sys.argv:
        server.run(transport="streamable-http")
    else:
        server.run()
