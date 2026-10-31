import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from connectome import (hcp_connectome, synthetic_connectome, load_csv_connectome,
                        region_lobe)
from model import APOE_PROFILES, Params, simulate, arrival_order

st.set_page_config(page_title="Connectome Spread", layout="wide")
st.title("Connectome Spread")
st.caption(
    "Network-diffusion simulation of neurodegenerative pathology spreading along "
    "brain wiring, with APOE genotype as a tunable risk parameter."
)

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Connectome")
    source = st.radio(
        "Source",
        ["HCP group-average (real)", "Synthetic", "Upload CSV"],
        help="HCP: Human Connectome Project diffusion MRI, 68 Desikan-Killiany "
             "regions, via the ENIGMA Toolbox. Synthetic: lobe-based toy wiring.",
    )
    if source == "Upload CSV":
        uploaded = st.file_uploader("Connectome CSV (labels in first row + column)",
                                    type="csv")
        if uploaded is None:
            st.stop()
        try:
            conn = load_csv_connectome(uploaded)
            st.success(f"Loaded {conn.shape[0]} regions")
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            st.stop()
    elif source == "Synthetic":
        conn = synthetic_connectome()
    else:
        conn = hcp_connectome()

    names = list(conn.index)

    st.header("Simulation")
    genotype = st.selectbox("APOE genotype", list(APOE_PROFILES), index=2)
    default_seed = [n for n in names if "entorhinal" in n] or names[:1]
    seeds = st.multiselect("Seed region(s)", names, default=default_seed)
    steps = st.slider("Time steps", 50, 1000, 300, 50)
    beta = st.slider("Baseline spread rate β", 0.005, 0.2, 0.05, 0.005)
    kappa = st.slider("Baseline clearance κ", 0.0, 0.05, 0.01, 0.0025)
    threshold = st.slider("Arrival threshold", 0.01, 1.0, 0.1, 0.01)

if not seeds:
    st.warning("Pick at least one seed region.")
    st.stop()

W = conn.values
seed_idx = [names.index(s) for s in seeds]
params = Params(beta=beta, kappa=kappa, steps=steps)

hist = simulate(W, seed_idx, genotype, params)
prof = APOE_PROFILES[genotype]
st.markdown(
    f"**{genotype}** — {prof['label']}: spread ×{prof['beta']}, clearance ×{prof['kappa']}"
)

# ---------------- row 1: total load over time, all genotypes ----------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Total pathology over time, by genotype")
    fig = go.Figure()
    for g in APOE_PROFILES:
        h = simulate(W, seed_idx, g, params)
        fig.add_trace(go.Scatter(y=h.sum(axis=1), mode="lines", name=g,
                                 line=dict(width=4 if g == genotype else 1.5)))
    fig.update_layout(xaxis_title="time step", yaxis_title="total load",
                      height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader(f"Final regional load — {genotype}")
    final = pd.DataFrame({"region": names, "load": hist[-1],
                          "lobe": [region_lobe(n) for n in names]})
    final = final.sort_values("load", ascending=False).head(20)
    fig = px.bar(final, x="load", y="region", color="lobe", orientation="h")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=380,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- row 2: spread heatmap + arrival order ----------------
c3, c4 = st.columns([2, 1])

with c3:
    st.subheader("Spread across regions over time")
    order = np.argsort(arrival_order(hist, threshold))
    fig = px.imshow(hist[:, order].T, aspect="auto",
                    labels=dict(x="time step", y="region (sorted by arrival)", color="load"),
                    y=[names[i] for i in order])
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c4:
    st.subheader("Arrival order")
    arr = arrival_order(hist, threshold)
    tbl = pd.DataFrame({"region": names, "arrival step": arr})
    tbl = tbl[np.isfinite(tbl["arrival step"])].sort_values("arrival step")
    tbl["arrival step"] = tbl["arrival step"].astype(int)
    st.dataframe(tbl.reset_index(drop=True), height=600, use_container_width=True)

with st.expander("How the model works"):
    st.markdown(
        """
Each region holds a pathology load *x*. At every step:

`dx/dt = −β · L·x − κ · x + seed`

- **L** is the graph Laplacian of the connectome: pathology flows from regions with
  more of it to connected regions with less, weighted by connection strength.
- **β** sets how fast it spreads along wiring; **κ** how fast the brain clears it.
- **APOE** scales β and κ. ε4 → faster spread, weaker clearance; ε2 → the reverse.

The APOE multipliers are illustrative, not fitted. The default connectome is the
HCP group-average Desikan-Killiany structural matrix from the ENIGMA Toolbox
(Larivière et al. 2021); the synthetic option is a toy for comparison.
"""
    )
