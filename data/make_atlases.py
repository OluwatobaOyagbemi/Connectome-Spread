"""Build the finer-parcellation connectomes used by resolution.py.

    python data/make_atlases.py /path/to/download/dir

For each atlas (Schaefer 100/200/300/400, Glasser 360) this downloads, from the
ENIGMA Toolbox (BSD-3; same HCP pipeline as hcp_dk68_sc.csv):

  matrices/hcp_connectivity/strucMatrix_ctx_<atlas>.csv   structural connectome
  matrices/hcp_connectivity/strucLabels_ctx_<atlas>.csv   parcel names
  parcellations/<atlas>_fsa5.csv                          per-vertex parcel label

and writes two files to data/atlases/:

  <atlas>_sc.csv       the connectome, parcel names in first row and column
  <atlas>_parcels.csv  per parcel: centroid (fsaverage mm), parent Desikan-Killiany
                       region by majority vertex overlap, and the fraction of the
                       parcel's vertices that are DK entorhinal cortex

Parcel i carries label i + 1 on the surface; first half of the parcels is the
left hemisphere. Vertices are taken from the parcel's own hemisphere only
(Glasser's fsa5 file has 17 stray right-hemisphere vertices labelled 180).
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np

from make_centroids import APARC, BASE, FILES, vertices

ATLASES = ["schaefer_100", "schaefer_200", "schaefer_300", "schaefer_400", "glasser_360"]
OUT = Path(__file__).parent / "atlases"


def fetch(download_dir: Path, local: str, remote: str) -> Path:
    path = download_dir / local
    if not path.exists():
        urllib.request.urlretrieve(BASE + remote, path)
    return path


def main(download_dir: Path) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    for local, remote in FILES.items():
        fetch(download_dir, local, remote)
    xyz = np.vstack([vertices(download_dir / "fsa5_lh.gii"), vertices(download_dir / "fsa5_rh.gii")])
    dk = np.loadtxt(download_dir / "aparc_fsa5.csv", dtype=int)
    hemi = np.r_[np.zeros(len(xyz) // 2, int), np.ones(len(xyz) // 2, int)]
    dk_base = np.array([APARC[v if v <= 35 else v - 35] if v > 0 else "unknown" for v in dk])
    OUT.mkdir(exist_ok=True)

    for atlas in ATLASES:
        mat = fetch(download_dir, f"strucMatrix_ctx_{atlas}.csv",
                    f"matrices/hcp_connectivity/strucMatrix_ctx_{atlas}.csv")
        lab = fetch(download_dir, f"strucLabels_ctx_{atlas}.csv",
                    f"matrices/hcp_connectivity/strucLabels_ctx_{atlas}.csv")
        par = fetch(download_dir, f"{atlas}_fsa5.csv", f"parcellations/{atlas}_fsa5.csv")
        names = lab.read_text().strip().split(",")
        W = np.loadtxt(mat, delimiter=",")
        labels = np.loadtxt(par, dtype=int)
        n = len(names)
        assert W.shape == (n, n) and np.allclose(W, W.T), atlas

        rows = ["parcel,x,y,z,parent_dk,entorhinal_fraction"]
        for i, name in enumerate(names):
            h = 0 if i < n // 2 else 1
            m = (labels == i + 1) & (hemi == h)
            assert m.any(), name
            x, y, z = xyz[m].mean(axis=0)
            assert (x < 0) == (h == 0), f"{name} centroid in wrong hemisphere"
            vals, counts = np.unique(dk_base[m], return_counts=True)
            parent = ("L_" if h == 0 else "R_") + vals[counts.argmax()]
            ent = float(np.mean(dk_base[m] == "entorhinal"))
            rows.append(f"{name},{x:.2f},{y:.2f},{z:.2f},{parent},{ent:.3f}")
        (OUT / f"{atlas}_parcels.csv").write_text("\n".join(rows) + "\n")

        header = "," + ",".join(names)
        body = [name + "," + ",".join(f"{v:g}" for v in W[i]) for i, name in enumerate(names)]
        (OUT / f"{atlas}_sc.csv").write_text("\n".join([header] + body) + "\n")
        print(f"{atlas}: {n} parcels")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("enigma_download"))
