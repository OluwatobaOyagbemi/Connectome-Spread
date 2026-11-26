"""Finer parcellations of the same HCP data: the README's resolution table.

Pins the result in both directions: the connectome never clearly beats
distance, and it carries distance-independent information at every resolution.
"""

import numpy as np
import pytest

from connectome import ATLASES, load_atlas
from model import Params
from resolution import atlas_stages, compare_atlas, entorhinal_seed, seed_sensitivity
from validation import hemisphere

FINE = [a for a in ATLASES if a != "dk_68"]


@pytest.mark.parametrize("atlas", FINE)
def test_atlas_files_are_aligned(atlas):
    A = load_atlas(atlas)
    n = len(A["names"])
    assert A["W"].shape == (n, n) and np.allclose(A["W"], A["W"].T)
    hemi = hemisphere(A["names"])
    assert ((A["xyz"][:, 0] < 0) == (hemi == "L")).all()          # centroids match names
    assert all(p[0] == h for p, h in zip(A["parent"], hemi))       # parent in same hemisphere
    stages = atlas_stages(A["parent"])
    assert np.isnan(stages).sum() <= 10                            # few medial-wall parcels
    seed = entorhinal_seed(A)
    assert set(hemi[seed]) == {"L", "R"}
    assert (A["xyz"][seed, 2] < -15).all()                         # seeds are inferior temporal


def test_visual_parcels_are_posterior():
    A = load_atlas("schaefer_200")
    vis = [i for i, n in enumerate(A["names"]) if "_Vis_" in n]
    assert A["xyz"][vis, 1].mean() < -60


@pytest.mark.parametrize("atlas", ATLASES)
def test_resolution_result(atlas):
    r = compare_atlas(atlas, n_null=30, n_boot=300)
    lo, hi = r["ci_connectome_minus_distance"]
    assert lo < 0 < hi                                   # connectome ≈ distance, every atlas
    assert r["partial"]["connectome | distance"] > 0.35  # but carries independent information
    assert r["rho"]["combined"] > max(r["rho"]["connectome"], r["rho"]["distance"])
    assert r["rho"]["connectome"] > r["geometry_null"]["p95"]


def test_resolution_table_values():
    expected = {"dk_68": (0.58, 0.65, 0.74), "schaefer_400": (0.58, 0.49, 0.63),
                "glasser_360": (0.47, 0.50, 0.61)}
    for atlas, (c, d, b) in expected.items():
        r = compare_atlas(atlas, n_null=5, n_boot=50)["rho"]
        assert (r["connectome"], r["distance"], r["combined"]) == pytest.approx((c, d, b), abs=0.005)


def test_seed_definition_does_not_change_the_picture():
    for atlas in ("schaefer_100", "schaefer_400", "glasser_360"):
        for row in seed_sensitivity(atlas, Params()):
            assert row["partial"] > 0.3
            assert abs(row["connectome"] - row["distance"]) < 0.1
