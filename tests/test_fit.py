"""The fitter must recover known parameters before it touches real data."""

import numpy as np
import pytest

from connectome import hcp_connectome
from fit import recover, pattern_loss


@pytest.fixture(scope="module")
def hcp():
    conn = hcp_connectome()
    names = list(conn.index)
    seed = [i for i, n in enumerate(names) if n.endswith("entorhinal")]
    return conn.values, names, seed


def test_pattern_loss_is_zero_for_identical_and_scale_invariant():
    x = np.random.default_rng(0).lognormal(0, 1, 68)
    assert pattern_loss(x, x)[0] == pytest.approx(0.0, abs=1e-9)
    assert pattern_loss(3.0 * x, x)[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("beta,kappa", [(0.05, 0.01), (0.09, 0.005), (0.025, 0.0125)])
def test_longitudinal_recovers_both_parameters(hcp, beta, kappa):
    W, _, seed = hcp
    lg = recover(W, seed, beta, kappa)["longitudinal"]
    assert lg.beta == pytest.approx(beta, rel=0.15)
    assert lg.kappa == pytest.approx(kappa, rel=0.15)


@pytest.mark.parametrize("beta,kappa", [(0.05, 0.01), (0.025, 0.0125)])
def test_cross_sectional_recovers_ratio(hcp, beta, kappa):
    W, _, seed = hcp
    cs = recover(W, seed, beta, kappa)["cross_sectional"]
    assert cs.ratio == pytest.approx(beta / kappa, rel=0.25)
    assert cs.r > 0.98


def test_cross_sectional_cannot_separate_beta_from_kappa(hcp):
    """Documents the identifiability limit: for a high β/κ ratio, a single
    snapshot returns the wrong absolute scale even though the pattern fits."""
    W, _, seed = hcp
    cs = recover(W, seed, 0.09, 0.005)["cross_sectional"]
    assert cs.r > 0.98                                   # pattern fits well…
    assert abs(np.log(cs.kappa / 0.005)) > np.log(1.5)   # …but κ is off by >1.5×
