"""Regression tests: the numbers in the README must reproduce.

    pip install pytest && pytest -q
"""

import numpy as np
import pytest

from connectome import hcp_connectome, synthetic_connectome, load_csv_connectome
from model import (APOE_PROFILES, Params, simulate, arrival_order, braak_stages,
                   braak_concordance, laplacian, _rank)
from validation import shuffled_braak_null, rewired_connectome_null, seed_control

THRESHOLD = 0.10


@pytest.fixture(scope="module")
def hcp():
    conn = hcp_connectome()
    names = list(conn.index)
    seed = [i for i, n in enumerate(names) if n.endswith("entorhinal")]
    return conn.values, names, seed


def test_connectome_shape_and_symmetry(hcp):
    W, names, _ = hcp
    assert W.shape == (68, 68)
    assert np.allclose(W, W.T)
    assert (W >= 0).all() and np.allclose(np.diag(W), 0)
    assert sum(n.startswith("L_") for n in names) == 34


def test_laplacian_is_normalised_and_psd(hcp):
    W, _, _ = hcp
    L = laplacian(W)
    ev = np.linalg.eigvalsh(L)
    assert ev.min() > -1e-9
    assert ev.max() <= 2.0 + 1e-9      # normalised Laplacian bound
    # explicit Euler at dt=1 is stable for every slider setting
    assert 0.20 * 1.8 * ev.max() < 2.0


def test_rank_handles_ties():
    assert np.allclose(_rank(np.array([3.0, 1.0, 3.0, 2.0])), [3.5, 1, 3.5, 2])


def test_readme_genotype_table(hcp):
    W, names, seed = hcp
    stages = braak_stages(names)
    expected = {"ε2/ε2": (5.6, 8, 0.55), "ε3/ε3": (7.3, 24, 0.64), "ε4/ε4": (11.5, 63, 0.54)}
    for g, (load, reached, rho) in expected.items():
        h = simulate(W, seed, g, Params())
        a = arrival_order(h, THRESHOLD)
        assert h[-1].sum() == pytest.approx(load, abs=0.05)
        assert int(np.isfinite(a).sum()) == reached
        assert braak_concordance(a, stages) == pytest.approx(rho, abs=0.005)


def test_genotype_ordering_is_monotone(hcp):
    W, _, seed = hcp
    totals = [simulate(W, seed, g, Params())[-1].sum() for g in APOE_PROFILES]
    assert totals == sorted(totals)       # ε2/ε2 < ε2/ε3 < ε3/ε3 < ε3/ε4 < ε4/ε4


def test_entorhinal_seed_beats_both_nulls(hcp):
    W, names, seed = hcp
    stages = braak_stages(names)
    a = arrival_order(simulate(W, seed, "ε3/ε3", Params()), THRESHOLD)
    shuf = shuffled_braak_null(a, stages, n=300)
    assert shuf.p95 < 0.3 and shuf.observed > shuf.p95
    rew = rewired_connectome_null(W, seed, names, Params(), THRESHOLD, n=30)
    assert rew.observed > rew.p95


def test_only_entorhinal_seed_gives_braak_order(hcp):
    W, names, _ = hcp
    rows = {r["seed"]: r["ρ"] for r in seed_control(W, names, Params(), THRESHOLD)}
    assert rows["entorhinal"] > 0.6
    assert all(v < 0.5 for s, v in rows.items() if s != "entorhinal")
    assert rows["precentral"] < 0                  # stage-VI seed runs Braak backwards


def test_ordering_survives_lower_threshold(hcp):
    """At threshold 0.02 every region arrives (no tie for last) and ρ still > 0.6."""
    W, names, seed = hcp
    a = arrival_order(simulate(W, seed, "ε3/ε3", Params()), 0.02)
    assert np.isfinite(a).all()
    assert braak_concordance(a, braak_stages(names)) > 0.6


def test_synthetic_and_csv_loader():
    s = synthetic_connectome()
    assert s.shape == (68, 68) and np.allclose(s.values, s.values.T)
    import io
    buf = io.StringIO(s.to_csv())
    again = load_csv_connectome(buf)
    assert np.allclose(again.values, s.values)
