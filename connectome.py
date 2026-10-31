"""Connectome loading.

A connectome here is just a square matrix W where W[i, j] is the strength of the
white-matter connection between region i and region j. Rows/columns are labelled
by region name.

Two sources are supported:
  1. A real connectome uploaded as CSV (region names in first row + first column).
  2. A synthetic connectome, built so the app runs with no data download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 34 cortical regions of the Desikan-Killiany atlas, one hemisphere.
# The synthetic connectome mirrors these across left (L) and right (R).
DK_REGIONS = [
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus",
    "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
    "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual",
    "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral",
    "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
    "postcentral", "posteriorcingulate", "precentral", "precuneus",
    "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
    "superiorparietal", "superiortemporal", "supramarginal", "frontalpole",
    "temporalpole", "transversetemporal", "insula",
]

# Rough lobe membership, used only to make the synthetic wiring plausible
# (regions in the same lobe connect more strongly).
LOBE = {
    "frontal": {"caudalmiddlefrontal", "lateralorbitofrontal", "medialorbitofrontal",
                "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis",
                "precentral", "rostralmiddlefrontal", "superiorfrontal", "frontalpole"},
    "temporal": {"bankssts", "entorhinal", "fusiform", "inferiortemporal",
                 "middletemporal", "parahippocampal", "superiortemporal",
                 "temporalpole", "transversetemporal"},
    "parietal": {"inferiorparietal", "postcentral", "precuneus", "superiorparietal",
                 "supramarginal"},
    "occipital": {"cuneus", "lateraloccipital", "lingual", "pericalcarine"},
    "cingulate": {"caudalanteriorcingulate", "isthmuscingulate",
                  "posteriorcingulate", "rostralanteriorcingulate"},
    "insula": {"insula"},
}


def region_lobe(name: str) -> str:
    base = name.split("_", 1)[1] if "_" in name else name
    for lobe, members in LOBE.items():
        if base in members:
            return lobe
    return "other"


def synthetic_connectome(seed: int = 0) -> pd.DataFrame:
    """Build a 68x68 symmetric, non-negative connectome that looks brain-like.

    Rules (deliberately simple):
      - same lobe, same hemisphere  -> strong connection
      - different lobe, same hemisphere -> weaker connection
      - homologous region across hemispheres -> strong (corpus callosum)
      - everything else across hemispheres -> weak / sparse
    """
    rng = np.random.default_rng(seed)
    names = [f"L_{r}" for r in DK_REGIONS] + [f"R_{r}" for r in DK_REGIONS]
    n = len(names)
    W = np.zeros((n, n))
    half = len(DK_REGIONS)

    for i in range(n):
        for j in range(i + 1, n):
            same_hemi = (i < half) == (j < half)
            homolog = abs(i - j) == half
            same_lobe = region_lobe(names[i]) == region_lobe(names[j])
            if homolog:
                base = 0.8
            elif same_hemi and same_lobe:
                base = 0.6
            elif same_hemi:
                base = 0.15
            else:
                base = 0.03
            # sparsify weak links, add noise
            if base < 0.2 and rng.random() < 0.6:
                continue
            w = max(0.0, base * rng.lognormal(mean=0.0, sigma=0.4))
            W[i, j] = W[j, i] = w

    return pd.DataFrame(W, index=names, columns=names)


def hcp_connectome() -> pd.DataFrame:
    """HCP group-average Desikan-Killiany (68 cortical regions) structural
    connectome, bundled in data/. See data/README.md for provenance."""
    path = Path(__file__).parent / "data" / "hcp_dk68_sc.csv"
    return load_csv_connectome(path)


def load_csv_connectome(file) -> pd.DataFrame:
    """Read a connectome CSV with region labels in the header row and first column."""
    df = pd.read_csv(file, index_col=0)
    df = df.astype(float)
    if list(df.index) != list(df.columns):
        raise ValueError("CSV must be square with matching row and column labels.")
    W = df.values
    W = np.maximum(W, 0)
    W = (W + W.T) / 2  # force symmetry
    np.fill_diagonal(W, 0)
    return pd.DataFrame(W, index=df.index, columns=df.columns)
