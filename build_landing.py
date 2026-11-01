"""Regenerate the numbers baked into docs/index.html from the model.

    python build_landing.py

The landing page's brain graph (node positions and edges) is static; this script
rewrites the per-region arrival annotations, the result table and the control
figures so they can never drift from what `reproduce.py` prints. Run it after
any change to the model, the defaults or the data, then commit docs/index.html.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from connectome import hcp_connectome
from model import APOE_PROFILES, Params, simulate, arrival_order, braak_stages, braak_concordance

ROOT = Path(__file__).parent
HTML = ROOT / "docs" / "index.html"
RESULTS = ROOT / "docs" / "figures" / "results.json"
THRESHOLD = 0.10
SECONDS_PER_STEP = 0.027    # animation pacing: 300 steps ≈ 8 s


def main() -> None:
    conn = hcp_connectome()
    names, W = list(conn.index), conn.values
    seed_idx = [i for i, n in enumerate(names) if n.endswith("entorhinal")]
    p = Params()
    stages = braak_stages(names)
    hists = {g: simulate(W, seed_idx, g, p) for g in APOE_PROFILES}
    h = hists["ε3/ε3"]
    arr = arrival_order(h, THRESHOLD)
    load = h[-1]
    rmax = np.sqrt(load.max())

    html = HTML.read_text()

    # ── circles ──────────────────────────────────────────────────────────────
    circle_re = re.compile(
        r'<circle class="n (?:dim|lit)"(?: style="animation-delay:[\d.]+s")? '
        r'cx="(?P<cx>[\d.]+)" cy="(?P<cy>[\d.]+)" r="(?P<r>[\d.]+)">'
        r'<title>(?P<name>[LR]_\w+): [^<]*</title></circle>')
    seen = set()

    def repl(m: re.Match) -> str:
        name = m.group("name")
        seen.add(name)
        i = names.index(name)
        if np.isfinite(arr[i]):
            step = int(arr[i])
            r = 6.0 + 14.5 * np.sqrt(load[i]) / rmax
            return (f'<circle class="n lit" style="animation-delay:{step * SECONDS_PER_STEP:.2f}s" '
                    f'cx="{m.group("cx")}" cy="{m.group("cy")}" r="{r:.1f}">'
                    f'<title>{name}: arrives at step {step}</title></circle>')
        return (f'<circle class="n dim" cx="{m.group("cx")}" cy="{m.group("cy")}" '
                f'r="{m.group("r")}"><title>{name}: not reached in {p.steps} steps</title></circle>')

    html, n_circ = circle_re.subn(repl, html)
    assert n_circ == len(names) and seen == set(names), (n_circ, set(names) - seen)

    # ── headline number + table ──────────────────────────────────────────────
    rho = braak_concordance(arr, stages)
    html = re.sub(r'<div class="num">[\d.]+<small>', f'<div class="num">{rho:.2f}<small>', html)
    sw = {"ε2/ε2": "#2a78d6", "ε3/ε3": "#eda100", "ε4/ε4": "#e34948"}
    for g, color in sw.items():
        a = arrival_order(hists[g], THRESHOLD)
        row = (f'<tr><td><span class="sw" style="background:{color}"></span>{g}</td>'
               f'<td>{hists[g][-1].sum():.1f}</td><td>{int(np.isfinite(a).sum())}</td>'
               f'<td>{braak_concordance(a, stages):.2f}</td></tr>')
        html, n = re.subn(rf'<tr><td><span class="sw" style="background:#[0-9a-f]{{6}}"></span>{re.escape(g)}</td>.*?</tr>',
                          row, html)
        assert n == 1, g

    # ── controls paragraph (from reproduce.py output) ────────────────────────
    if RESULTS.exists():
        R = json.loads(RESULTS.read_text())
        s, w = R["shuffled_braak_null"], R["rewired_connectome_null"]
        seeds = {r["seed"]: r["ρ"] for r in R["seed_control"]}
        para = (f'<p id="controls">Against chance: shuffling the Braak labels gives a 95th-percentile '
                f'ρ of {s["p95"]:.2f} ({s["n"]:,} permutations), and rewiring the connectome with the same '
                f'density and weights gives {w["p95"]:.2f} ({w["n"]} graphs). Moving the seed breaks it: '
                f'fusiform {seeds["fusiform"]:+.2f}, superior parietal {seeds["superiorparietal"]:+.2f}, '
                f'precentral {seeds["precentral"]:+.2f}. Only the entorhinal start, where Braak staging '
                f'begins, produces Braak order. <a href="https://github.com/OluwatobaOyagbemi/connectome-spread/blob/main/reproduce.py">'
                f'<code>reproduce.py</code></a> regenerates every number on this page.</p>')
        if '<p id="controls">' in html:
            html = re.sub(r'<p id="controls">.*?</p>', para, html, flags=re.S)
        else:
            anchor = "so the claim comes with its own range of validity.</p>"
            assert anchor in html
            html = html.replace(anchor, anchor + "\n        " + para)

    HTML.write_text(html)
    print(f"Updated {HTML}: {n_circ} regions, ρ = {rho:.2f}")


if __name__ == "__main__":
    main()
