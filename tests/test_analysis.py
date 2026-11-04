"""Seed scan and lesion scan: the two analyses beyond the published model."""

import numpy as np
import pytest

from connectome import hcp_connectome
from model import Params
from analysis import seed_scan, seed_stage_trend, knockout_scan


@pytest.fixture(scope="module")
def hcp():
    conn = hcp_connectome()
    names = list(conn.index)
    seed = [i for i, n in enumerate(names) if n.endswith("entorhinal")]
    return conn.values, names, seed


def test_entorhinal_is_the_best_seed_of_all_34(hcp):
    W, names, _ = hcp
    scan = seed_scan(W, names, Params(), 0.10)
    assert len(scan) == 34
    assert scan.iloc[0]["seed"] == "entorhinal"
    assert (scan.iloc[1:]["ρ"] < scan.iloc[0]["ρ"]).all()


def test_concordance_falls_with_seed_stage(hcp):
    W, names, _ = hcp
    r, early_minus_late = seed_stage_trend(seed_scan(W, names, Params(), 0.10))
    assert r < -0.5
    assert early_minus_late > 0.3


def test_lesion_scan_scores_survivors_only(hcp):
    """Removing a Braak III region must not be penalised just because the
    removed region can never arrive."""
    W, names, seed = hcp
    ko = knockout_scan(W, names, seed, Params(), 0.10)
    assert len(ko) == 33                       # 34 minus the seed
    assert abs(ko.attrs["baseline"] - 0.64) < 0.01
    assert ko["Δρ"].abs().max() < 0.15         # modest effects, by design


def test_braak_iii_regions_are_bridges(hcp):
    W, names, seed = hcp
    ko = knockout_scan(W, names, seed, Params(), 0.10)
    top4 = set(ko.head(4)["removed"])
    assert {"fusiform", "parahippocampal", "lingual"} <= top4
