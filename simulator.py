from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from connectome import (hcp_connectome, synthetic_connectome, load_csv_connectome,
                        region_lobe)
from model import (APOE_PROFILES, Params, simulate, arrival_order, braak_stages,
                   braak_concordance, beta_sweep)
from validation import (shuffled_braak_null, rewired_connectome_null, seed_control,
                        threshold_control)

# ─────────────────────────────────────────────────────────────────────────────
# Page + theme
# ─────────────────────────────────────────────────────────────────────────────

GENOTYPES = list(APOE_PROFILES)
# Validated categorical palette (light / dark), fixed order protective → risk.
GENO_COLOR = {
    "ε2/ε2": "#2a78d6", "ε2/ε3": "#1baf7a", "ε3/ε3": "#eda100",
    "ε3/ε4": "#6250d6", "ε4/ε4": "#e34948",
}
LOBE_ORDER = ["temporal", "frontal", "parietal", "occipital", "cingulate", "insula", "other"]
LOBE_COLOR = dict(zip(LOBE_ORDER,
                      ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                       "#008300", "#6250d6"]))
INK = "#52514e"
GRID = "rgba(128,128,128,0.18)"

DEFAULTS = dict(beta=0.05, kappa=0.01, steps=300, threshold=0.10)

st.markdown("""
<style>
  /* typography & spacing */
  .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }
  h1 { font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.1rem; }
  .cs-sub { color: #6b6a66; font-size: 1.02rem; margin-bottom: 1.2rem; }
  .cs-eyebrow { font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
                color: #8a8985; font-weight: 600; margin-bottom: 0.25rem; }
  /* metric cards */
  div[data-testid="stMetric"] {
      border: 1px solid rgba(128,128,128,0.22); border-radius: 10px;
      padding: 0.75rem 1rem 0.6rem; }
  div[data-testid="stMetricLabel"] p { font-size: 0.78rem; color: #6b6a66;
      text-transform: uppercase; letter-spacing: 0.06em; }
  div[data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 700; }
  /* genotype badge */
  .cs-badge { display:inline-block; padding: 0.18rem 0.6rem; border-radius: 999px;
      font-size: 0.8rem; font-weight: 600; color: white; margin-right: 0.4rem; }
  /* sidebar */
  section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
  section[data-testid="stSidebar"] h3 { font-size: 0.8rem; letter-spacing: 0.1em;
      text-transform: uppercase; color: #8a8985; margin-top: 1.2rem; }
  /* tabs */
  button[data-baseweb="tab"] { font-weight: 600; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def base_layout(height: int, **kw) -> dict:
    """Recessive axes, transparent surfaces, consistent fonts."""
    lay = dict(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13, color=INK),
        hoverlabel=dict(font_size=12),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=12),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=GRID, ticks="outside",
                   tickcolor=GRID),
        yaxis=dict(gridcolor=GRID, zeroline=False, showline=False),
    )
    lay.update(kw)
    return lay


# ─────────────────────────────────────────────────────────────────────────────
# Data + simulation (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_all(W: np.ndarray, seed_idx: tuple[int, ...], beta: float, kappa: float,
            steps: int) -> dict[str, np.ndarray]:
    p = Params(beta=beta, kappa=kappa, steps=steps)
    return {g: simulate(W, list(seed_idx), g, p) for g in GENOTYPES}


@st.cache_data(show_spinner=False)
def run_sweep(W: np.ndarray, seed_idx: tuple[int, ...], names: tuple[str, ...],
              kappa: float, steps: int, threshold: float):
    betas = np.round(np.linspace(0.01, 0.20, 20), 3)
    p = Params(kappa=kappa, steps=steps)
    return betas, beta_sweep(W, list(seed_idx), list(names), betas, p, threshold)


@st.cache_data(show_spinner=False)
def run_controls(W: np.ndarray, seed_idx: tuple[int, ...], names: tuple[str, ...],
                 beta: float, kappa: float, steps: int, threshold: float):
    p = Params(beta=beta, kappa=kappa, steps=steps)
    names = list(names)
    arr = arrival_order(simulate(W, list(seed_idx), "ε3/ε3", p), threshold)
    shuf = shuffled_braak_null(arr, braak_stages(names), n=2000)
    rew = rewired_connectome_null(W, list(seed_idx), names, p, threshold, n=100)
    seeds = seed_control(W, names, p, threshold)
    thr = threshold_control(W, list(seed_idx), names, p)
    return shuf, rew, seeds, thr


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("[← Project home](/home)")
    st.markdown("### Connectome")
    source = st.radio(
        "Source", ["HCP group-average", "Synthetic", "Upload CSV"],
        label_visibility="collapsed",
        help="HCP: Human Connectome Project diffusion MRI, 68 Desikan-Killiany "
             "regions (ENIGMA Toolbox). Synthetic: lobe-based toy wiring.",
    )
    if source == "Upload CSV":
        up = st.file_uploader("CSV with region labels in first row and column",
                              type="csv")
        if up is None:
            st.info("Upload a square connectome CSV to continue.")
            st.stop()
        try:
            conn = load_csv_connectome(up)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            st.stop()
    elif source == "Synthetic":
        conn = synthetic_connectome()
    else:
        conn = hcp_connectome()
    names = list(conn.index)
    W = conn.values

    st.markdown("### Genotype")
    try:
        genotype = st.segmented_control("APOE", GENOTYPES, default="ε3/ε3",
                                        label_visibility="collapsed")
    except AttributeError:  # older Streamlit
        genotype = st.selectbox("APOE", GENOTYPES, index=2,
                                label_visibility="collapsed")
    genotype = genotype or "ε3/ε3"

    st.markdown("### Seed")
    default_seed = [n for n in names if n.endswith("entorhinal")] or names[:1]
    seeds = st.multiselect("Seed region(s)", names, default=default_seed,
                           label_visibility="collapsed")
    if not seeds:
        st.warning("Pick at least one seed region.")
        st.stop()

    st.markdown("### Model")
    with st.expander("Parameters", expanded=False):
        beta = st.slider("Spread rate β", 0.005, 0.20, DEFAULTS["beta"], 0.005,
                         help="How fast pathology moves along connections.")
        kappa = st.slider("Clearance κ", 0.0, 0.05, DEFAULTS["kappa"], 0.0025,
                          help="How fast the brain removes pathology.")
        steps = st.slider("Time steps", 50, 1000, DEFAULTS["steps"], 50,
                          help="Steps are dimensionless model time, not years. "
                               "Only the order and relative timing of arrivals "
                               "are meaningful.")
        threshold = st.slider("Arrival threshold", 0.01, 1.0,
                              DEFAULTS["threshold"], 0.01,
                              help="Load a region must reach to count as 'arrived'.")
    st.caption("Defaults: β 0.05 · κ 0.01 · 300 steps · threshold 0.10")

seed_idx = tuple(names.index(s) for s in seeds)
hists = run_all(W, seed_idx, beta, kappa, steps)
hist = hists[genotype]
prof = APOE_PROFILES[genotype]
stages = braak_stages(names)
arr = arrival_order(hist, threshold)
reached = int(np.isfinite(arr).sum())
rho = braak_concordance(arr, stages)
total = hist.sum(axis=1)
ref_total = hists["ε3/ε3"].sum(axis=1)[-1]

# ─────────────────────────────────────────────────────────────────────────────
# Header + KPIs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="cs-eyebrow">Network diffusion · APOE · Braak staging</div>',
            unsafe_allow_html=True)
st.title("Connectome Spread")
st.markdown(
    '<div class="cs-sub">How Alzheimer\'s tau pathology spreads along the brain\'s '
    'wiring, tested against Braak staging, with APOE genotype as a what-if dial.</div>',
    unsafe_allow_html=True)

st.markdown(
    f'<span class="cs-badge" style="background:{GENO_COLOR[genotype]}">{genotype}</span>'
    f'<span style="color:#6b6a66">{prof["label"]} · spread ×{prof["beta"]} · '
    f'clearance ×{prof["kappa"]} · seed: {", ".join(seeds)}</span>',
    unsafe_allow_html=True)
st.write("")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total load", f"{total[-1]:.2f}",
          delta=f"{(total[-1] / ref_total - 1) * 100:+.0f}% vs ε3/ε3"
          if ref_total > 0 else None)
k2.metric("Regions reached", f"{reached} / {len(names)}",
          help=f"Regions whose load crossed {threshold:.2f} within {steps} steps")
k3.metric("Braak concordance", "—" if np.isnan(rho) else f"ρ = {rho:.2f}",
          help="Spearman correlation between simulated arrival order and Braak "
               "tau stages (1 = perfect Braak order, 0 = unrelated)")
finite = arr[np.isfinite(arr)]
k4.metric("Median arrival", f"{int(np.median(finite))} steps" if len(finite) else "—",
          help="Median time step at which reached regions crossed the threshold")

st.write("")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_spread, tab_regions, tab_sens, tab_ctrl, tab_model = st.tabs(
    ["Spread", "Regions", "β sensitivity", "Controls", "Model & data"])

# ── Spread ────────────────────────────────────────────────────────────────────
with tab_spread:
    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.markdown("#### Total pathology over time")
        fig = go.Figure()
        for g in GENOTYPES:
            y = hists[g].sum(axis=1)
            sel = g == genotype
            fig.add_trace(go.Scatter(
                x=np.arange(len(y)), y=y, mode="lines", name=g,
                line=dict(color=GENO_COLOR[g], width=3.2 if sel else 1.6),
                opacity=1.0 if sel else 0.75,
                hovertemplate=f"<b>{g}</b><br>step %{{x}}<br>load %{{y:.2f}}<extra></extra>",
            ))
            fig.add_annotation(x=len(y) - 1, y=y[-1], text=g, showarrow=False,
                               xanchor="left", xshift=6, font=dict(size=11,
                               color=GENO_COLOR[g]))
        fig.update_layout(**base_layout(400, hovermode="x unified",
                          xaxis_title="time step", yaxis_title="total load",
                          showlegend=False))
        fig.update_xaxes(range=[0, len(y) * 1.1])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Each line is the whole-brain pathology load for one genotype; "
                   "the selected genotype is emphasised.")

    with c2:
        st.markdown("#### Arrival cascade")
        order = np.argsort(arr, kind="mergesort")
        order = order[np.isfinite(arr[order])]
        if len(order):
            z = hist[:, order].T
            fig = go.Figure(go.Heatmap(
                z=z, y=[names[i] for i in order], x=np.arange(z.shape[1]),
                colorscale="Blues", showscale=True,
                colorbar=dict(title="load", thickness=10, len=0.8, outlinewidth=0),
                hovertemplate="<b>%{y}</b><br>step %{x}<br>load %{z:.3f}<extra></extra>",
            ))
            fig.update_layout(**base_layout(max(400, 14 * len(order) + 60),
                              xaxis_title="time step"))
            fig.update_yaxes(autorange="reversed", tickfont=dict(size=10),
                             showgrid=False)
            st.plotly_chart(fig, width="stretch",
                            config={"displayModeBar": False})
            st.caption(f"{len(order)} regions that crossed the threshold, sorted by "
                       "arrival. Seed regions are at the top.")
        else:
            st.info("No region crossed the threshold. Lower it or increase steps.")

# ── Regions ───────────────────────────────────────────────────────────────────
with tab_regions:
    c1, c2 = st.columns([1.2, 1])

    df = pd.DataFrame({
        "region": names,
        "lobe": [region_lobe(n) for n in names],
        "final load": hist[-1],
        "arrival step": arr,
        "Braak stage": stages,
    })

    with c1:
        st.markdown("#### Final regional load")
        top = df.sort_values("final load", ascending=False).head(24)
        fig = go.Figure()
        for lobe in LOBE_ORDER:
            d = top[top.lobe == lobe]
            if d.empty:
                continue
            fig.add_trace(go.Bar(
                x=d["final load"], y=d["region"], orientation="h", name=lobe,
                marker=dict(color=LOBE_COLOR[lobe], line=dict(width=0)),
                hovertemplate="<b>%{y}</b><br>load %{x:.3f}<extra>" + lobe + "</extra>",
            ))
        fig.update_layout(**base_layout(max(420, 20 * len(top) + 80), barmode="stack",
                          bargap=0.28, xaxis_title="load",
                          legend=dict(orientation="h", y=1.04, x=0, font=dict(size=11),
                                      bgcolor="rgba(0,0,0,0)")))
        fig.update_yaxes(categoryorder="array",
                         categoryarray=list(top["region"][::-1]),
                         tickfont=dict(size=10), showgrid=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Top 24 regions by load at the final step, coloured by lobe.")

    with c2:
        st.markdown("#### Arrival order vs Braak stage")
        tbl = df[np.isfinite(df["arrival step"])].sort_values("arrival step").copy()
        tbl["arrival step"] = tbl["arrival step"].astype(int)
        tbl["Braak stage"] = tbl["Braak stage"].map(
            lambda s: "—" if np.isnan(s) else ["", "I", "II", "III", "IV", "V", "VI"][int(s)])
        tbl = tbl[["region", "arrival step", "Braak stage", "lobe"]].reset_index(drop=True)
        tbl.index = tbl.index + 1
        st.dataframe(tbl, height=520, width="stretch",
                     column_config={
                         "arrival step": st.column_config.NumberColumn(format="%d"),
                     })
        st.caption("A simulation that follows Braak order should list stage I, "
                   "then III, IV, V, VI from top to bottom.")

# ── Sensitivity ───────────────────────────────────────────────────────────────
with tab_sens:
    st.markdown("#### How robust is the Braak-like ordering to the spread rate?")
    st.markdown(
        "Arrival order depends on **β**. At low β pathology creeps through near "
        "neighbours and the medial-temporal cascade matches Braak staging; at high β "
        "strongly connected distant hubs jump the queue. This sweep measures that "
        "directly, so the result is stated with its range rather than as a single "
        "lucky setting.")
    with st.spinner("Sweeping β…"):
        betas, sweep = run_sweep(W, seed_idx, tuple(names), kappa, steps, threshold)

    fig = go.Figure()
    for g in GENOTYPES:
        sel = g == genotype
        fig.add_trace(go.Scatter(
            x=betas, y=sweep[g], mode="lines+markers", name=g,
            line=dict(color=GENO_COLOR[g], width=3 if sel else 1.6),
            marker=dict(size=7 if sel else 5),
            opacity=1.0 if sel else 0.75,
            hovertemplate=f"<b>{g}</b><br>β %{{x:.3f}}<br>ρ %{{y:.2f}}<extra></extra>",
        ))
    fig.add_vline(x=beta, line=dict(color="rgba(128,128,128,0.55)", dash="dot"),
                  annotation_text=f"current β = {beta:.3f}", annotation_position="top")
    fig.add_hrect(y0=0.5, y1=1.0, fillcolor="rgba(128,128,128,0.07)", line_width=0,
                  annotation_text="ρ ≥ 0.5: Braak-consistent", annotation_position="top left",
                  annotation_font_size=11)
    fig.update_layout(**base_layout(430, hovermode="x unified",
                      xaxis_title="baseline spread rate β",
                      yaxis_title="Braak concordance ρ"))
    fig.update_yaxes(range=[-0.1, 1.0])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    best_b = betas[np.nanargmax(sweep[genotype])]
    window = betas[np.nan_to_num(sweep[genotype]) >= 0.5]
    if len(window):
        st.success(
            f"For **{genotype}** with this seed, concordance peaks at β ≈ {best_b:.3f} "
            f"(ρ = {np.nanmax(sweep[genotype]):.2f}) and stays ≥ 0.5 for "
            f"β ∈ [{window.min():.3f}, {window.max():.3f}].")
    else:
        st.warning(f"For **{genotype}** with this seed, concordance never reaches 0.5 "
                   "— try an entorhinal seed, which is where Braak staging begins.")

# ── Controls ──────────────────────────────────────────────────────────────────
with tab_ctrl:
    st.markdown("#### Is the Braak-like ordering better than chance?")
    st.markdown(
        "A concordance of ρ = 0.64 only means something against a baseline. "
        "Three controls, all run on the ε3/ε3 reference with the current seed and "
        "parameters: **shuffle the Braak labels** (could any labelling of this "
        "arrival sequence score this high?), **rewire the connectome** (same density "
        "and weights, scrambled topology; does the specific wiring matter?), and "
        "**move the seed** (is the entorhinal start special, as Braak staging says?).")
    with st.spinner("Running controls…"):
        shuf, rew, seeds, thr = run_controls(W, seed_idx, tuple(names), beta, kappa,
                                             steps, threshold)

    c1, c2 = st.columns(2)
    for col, null, title, n_lab in ((c1, shuf, "Shuffled Braak labels", "2000 permutations"),
                                    (c2, rew, "Rewired connectome", "100 random graphs")):
        with col:
            st.markdown(f"**{title}** · {n_lab}")
            samples = null.samples[~np.isnan(null.samples)]
            fig = go.Figure(go.Histogram(x=samples, nbinsx=30,
                                         marker=dict(color="#cfe0f7", line=dict(width=0)),
                                         hovertemplate="ρ %{x:.2f}<br>%{y} samples<extra></extra>"))
            fig.add_vline(x=null.observed, line=dict(color="#e34948", width=2.5),
                          annotation_text=f"observed ρ = {null.observed:.2f}",
                          annotation_position="top left", annotation_font_color="#e34948")
            fig.add_vline(x=null.p95, line=dict(color=INK, width=1, dash="dot"),
                          annotation_text=f"null 95th pct = {null.p95:.2f}",
                          annotation_position="bottom right", annotation_font_size=11)
            fig.update_layout(**base_layout(300, xaxis_title="Braak concordance ρ",
                              yaxis_title="count", showlegend=False))
            fig.update_xaxes(range=[-0.7, 0.9])
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            ptxt = "p < 0.001" if null.p_value < 0.001 else f"p = {null.p_value:.3f}"
            st.caption(f"Null mean {null.mean:.2f}, 95th percentile {null.p95:.2f}; "
                       f"observed {null.observed:.2f}, one-sided {ptxt}.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Seed control** · bilateral seed, ε3/ε3")
        sdf = pd.DataFrame(seeds)
        st.dataframe(sdf, hide_index=True, width="stretch",
                     column_config={"ρ": st.column_config.NumberColumn(format="%+.2f")})
        best = max(seeds, key=lambda r: r["ρ"])
        st.caption(f"Highest concordance from a **{best['seed']}** seed "
                   f"(ρ = {best['ρ']:+.2f}). Seeding late-stage regions runs the "
                   "cascade backwards and the sign flips.")
    with c4:
        st.markdown("**Threshold control** · entorhinal seed, ε3/ε3")
        st.dataframe(pd.DataFrame(thr), hide_index=True, width="stretch",
                     column_config={"ρ": st.column_config.NumberColumn(format="%.2f"),
                                    "threshold": st.column_config.NumberColumn(format="%.2f")})
        st.caption("At a high threshold many regions never 'arrive' and tie for last, "
                   "which could inflate ρ. Lowering the threshold until every region "
                   "arrives removes the ties; the ordering should survive.")
    st.caption("Every number here is reproduced by `python reproduce.py` in the repository.")

# ── Model & data ──────────────────────────────────────────────────────────────
with tab_model:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### The model")
        st.latex(r"\frac{d\mathbf{x}}{dt} = -\beta\,\mathbf{L}\mathbf{x} - \kappa\,\mathbf{x} + \mathbf{s}")
        st.markdown("""
- **x** — pathology load in each region
- **L** — normalised graph Laplacian of the connectome: pathology flows from regions
  with more of it to connected regions with less, weighted by connection strength
- **β** — spread rate along wiring · **κ** — clearance rate · **s** — seed production

This is linear network diffusion after Raj, Kuceyeski & Weiner (*Neuron* 2012),
with a clearance term and a constant seed added. Time steps are dimensionless:
only the order and relative timing of arrivals carry meaning. APOE genotype
scales β and κ:
""")
        st.dataframe(pd.DataFrame(
            [{"genotype": g, "effect": v["label"], "spread ×": v["beta"],
              "clearance ×": v["kappa"]} for g, v in APOE_PROFILES.items()]
        ).set_index("genotype"), width="stretch")
        st.caption("Multipliers are illustrative, not fitted. Their direction follows "
                   "the literature: ε4 carriers accumulate tau faster on tau-PET "
                   "(Therriault et al., *JAMA Neurol* 2020) and ApoE isoforms differ "
                   "in brain clearance of amyloid-β, ε4 slowest (Castellano et al., "
                   "*Sci Transl Med* 2011). The magnitudes are a what-if, so the "
                   "genotype curves show the model's assumptions, not a finding.")

    with c2:
        st.markdown("#### Data")
        st.markdown("""
**Connectome.** Group-average structural connectivity from Human Connectome Project
diffusion MRI, 68 cortical regions of the Desikan-Killiany atlas, distributed with
the ENIGMA Toolbox (Larivière et al., *Nature Methods* 2021). Values are
log-transformed streamline counts.

**Braak reference.** Cortical regions grouped into tau Braak stages following the
tau-PET staging scheme (Schöll et al., *Neuron* 2016). Stage II is hippocampal and
not represented in the cortical parcellation.

**Concordance.** Spearman ρ between simulated arrival step and Braak stage. Regions
that never arrive are ranked last.
""")
        st.markdown("#### Limitations")
        st.markdown("""
- Group-average wiring: no individual variability
- Linear diffusion: no regional vulnerability, protein-specific kinetics, or
  vascular contribution
- Arrival order is β-sensitive — see the β sensitivity tab; the window narrows
  for ε4 genotypes
- Braak staging is a tau/Alzheimer's reference; the model itself is agnostic to
  the protein, so results for other proteinopathies need their own staging scheme
""")
        st.markdown("[Project site](https://oluwatobaoyagbemi.github.io/Connectome-Spread) · "
                    "[Source code](https://github.com/OluwatobaOyagbemi/connectome-spread)")
