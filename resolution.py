"""Does a finer parcellation let the connectome beat straight-line distance?

At DK-68 the connectome and distance from the entorhinal seed predict Braak
order equally well. One reasonable objection: 68 regions is coarse, and the
wiring may only show its advantage at higher resolution. This module reruns the
comparison on five finer parcellations of the same HCP data (Schaefer
100/200/300/400 and Glasser 360, from the ENIGMA Toolbox).

Method, per atlas:

- Braak stage: each parcel inherits the stage of the Desikan-Killiany region it
  mostly overlaps on the fsaverage5 surface. Parcels mostly on the medial wall
  get no stage and are left out of the score (they stay in the network).
- Seed: in each hemisphere, the parcels whose majority DK region is entorhinal;
  if none, the single parcel with the most entorhinal vertices. Schaefer atlases
  cover entorhinal cortex poorly, so at Schaefer-100 this is a temporal-pole
  parcel only ~15% entorhinal. Both predictors use the same seed, so the
  comparison stays like-for-like; `seed_sensitivity` checks other definitions.
- Ordering: final-step load, not threshold arrival. A threshold tuned on DK-68
  leaves most fine parcels un-arrived and tied, because the model reaches steady
  state with fine parcels plateauing below it (see `spread_order`).
- CIs: bootstrap over the 34 parent DK regions, so atlases with more parcels do
  not get narrower intervals from parcels that are not independent.
"""

from __future__ import annotations

import numpy as np

from connectome import ATLASES, load_atlas
from model import BRAAK_STAGE, Params, braak_concordance, _rank
from validation import (_partial, connectome_vs_distance, distance_from_seed,
                        geometry_preserving_null, hemisphere, spread_order)


def atlas_stages(parent: list[str]) -> np.ndarray:
    return np.array([BRAAK_STAGE.get(p.split("_", 1)[1], np.nan) for p in parent], float)


def entorhinal_seed(A: dict) -> list[int]:
    """Majority-entorhinal parcels per hemisphere; else the most-entorhinal one."""
    hemi = hemisphere(A["names"])
    seed = []
    for h in ("L", "R"):
        idx = np.flatnonzero(hemi == h)
        maj = [i for i in idx if A["parent"][i].endswith("_entorhinal")]
        seed += maj or [int(idx[np.argmax(A["entorhinal_fraction"][idx])])]
    return seed


def compare_atlas(atlas: str, p: Params = Params(), n_null: int = 200,
                  n_boot: int = 2000) -> dict:
    A = load_atlas(atlas)
    stages = atlas_stages(A["parent"])
    seed = entorhinal_seed(A)
    clusters = [x.split("_", 1)[1] for x in A["parent"]]
    c = connectome_vs_distance(A["W"], A["xyz"], seed, A["names"], p, 0.0, n_boot=n_boot,
                               stages=stages, clusters=clusters, ordering="load")
    g = geometry_preserving_null(A["W"], A["xyz"], seed, A["names"], p, 0.0, n=n_null,
                                 stages=stages, ordering="load")
    return {"atlas": atlas, "n_parcels": len(A["names"]),
            "seed": [A["names"][i] for i in seed],
            "seed_entorhinal_fraction": [round(float(A["entorhinal_fraction"][i]), 2) for i in seed],
            **c, "geometry_null": {"n": n_null, "mean": g.mean, "p95": g.p95, "p": g.p_value}}


def seed_sensitivity(atlas: str, p: Params = Params()) -> list[dict]:
    """The same comparison under alternative entorhinal seed definitions."""
    A = load_atlas(atlas)
    stages = atlas_stages(A["parent"])
    hemi = hemisphere(A["names"])
    ef = A["entorhinal_fraction"]
    defs = {"default": entorhinal_seed(A),
            "top-2 entorhinal parcels per hemisphere":
                [int(i) for h in ("L", "R") for i in np.flatnonzero(hemi == h)[np.argsort(-ef[hemi == h])[:2]]],
            "every parcel ≥10% entorhinal": [int(i) for i in np.flatnonzero(ef >= 0.10)]}
    rows = []
    for label, seed in defs.items():
        if not seed:
            continue
        o = spread_order(A["W"], seed, p, 0.0, ordering="load")
        d = distance_from_seed(A["xyz"], seed)
        rc, rd = braak_concordance(o, stages), braak_concordance(d, stages)
        agree = braak_concordance(o, d)
        rows.append({"seed definition": label, "n seeds": len(seed), "connectome": rc,
                     "distance": rd, "combined": braak_concordance(_rank(o) + _rank(d), stages),
                     "partial": _partial(rc, agree, rd)})
    return rows


def resolution_series(atlases=ATLASES, **kw) -> list[dict]:
    return [compare_atlas(a, **kw) for a in atlases]
