"""The geometry controls: does the connectome beat straight-line distance?

These pin the README's 'Wiring vs distance' numbers, including the ones that
cut against the connectome, so a change cannot quietly flatter the result.
"""

import numpy as np
import pytest

from analysis import seed_scan, seed_stage_trend
from connectome import hcp_connectome, hcp_centroids
from model import Params
from validation import (distance_from_seed, geometry_preserving_rewire,
                        geometry_preserving_null, connectome_vs_distance)

THRESHOLD = 0.10


@pytest.fixture(scope="module")
def hcp():
    conn = hcp_connectome()
    names = list(conn.index)
    seed = [i for i, n in enumerate(names) if n.endswith("entorhinal")]
    return conn.values, names, seed, hcp_centroids(names)


def test_centroids_are_anatomically_plausible(hcp):
    _, names, _, xyz = hcp
    assert xyz.shape == (68, 3)
    left = np.array([n.startswith("L_") for n in names])
    assert (xyz[left, 0] < 0).all() and (xyz[~left, 0] > 0).all()   # hemispheres
    ent = xyz[names.index("L_entorhinal")]
    assert ent[2] < -25 and -20 < ent[1] < 5                          # inferior, mid-temporal
    assert xyz[names.index("L_frontalpole"), 1] > 55                   # anterior
    assert xyz[names.index("L_pericalcarine"), 1] < -70                # posterior


def test_geometry_rewire_preserves_weights_and_distance_profile(hcp):
    W, names, _, xyz = hcp
    R = geometry_preserving_rewire(W, xyz, names, np.random.default_rng(0))
    iu = np.triu_indices(68, 1)
    assert np.allclose(R, R.T)
    assert np.allclose(np.sort(R[iu]), np.sort(W[iu]))                # same weights
    assert not np.allclose(R, W)                                       # but moved
    d = np.linalg.norm(xyz[iu[0]] - xyz[iu[1]], axis=-1)
    wd = lambda M: np.corrcoef(M[iu], d)[0, 1]
    assert wd(R) == pytest.approx(wd(W), abs=0.05)                     # weight-distance kept


def test_distance_alone_matches_the_connectome(hcp):
    W, names, seed, xyz = hcp
    c = connectome_vs_distance(W, xyz, seed, names, Params(), THRESHOLD, n_boot=500)
    assert c["rho"]["connectome"] == pytest.approx(0.64, abs=0.005)
    assert c["rho"]["distance"] == pytest.approx(0.65, abs=0.005)
    lo, hi = c["ci_connectome_minus_distance"]
    assert lo < 0 < hi                                  # cannot tell them apart


def test_connectome_carries_information_distance_lacks(hcp):
    W, names, seed, xyz = hcp
    for t in (0.02, THRESHOLD):
        c = connectome_vs_distance(W, xyz, seed, names, Params(), t, n_boot=200)
        assert c["order_agreement"] < 0.5
        assert c["partial"]["connectome | distance"] > 0.5
        assert c["rho"]["combined"] > max(c["rho"]["connectome"], c["rho"]["distance"]) + 0.05


def test_connectome_beats_geometry_preserving_null(hcp):
    W, names, seed, xyz = hcp
    g = geometry_preserving_null(W, xyz, seed, names, Params(), THRESHOLD, n=60)
    assert 0.3 < g.mean < 0.5           # geometry alone is a strong null...
    assert g.observed > g.p95           # ...that the real wiring still beats


def test_distance_also_picks_entorhinal_as_best_seed(hcp):
    W, names, _, xyz = hcp
    scan = seed_scan(W, names, Params(), THRESHOLD, xyz=xyz)
    assert scan.sort_values("ρ (distance)", ascending=False).seed.iloc[0] == "entorhinal"
    r, _ = seed_stage_trend(scan, "ρ (distance)")
    assert r < -0.5


def test_distance_from_seed_is_zero_at_seed(hcp):
    _, _, seed, xyz = hcp
    assert np.allclose(distance_from_seed(xyz, seed)[seed], 0)
