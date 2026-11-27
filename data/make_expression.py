"""Build the gene-expression inputs used by vulnerability.py.

    python data/make_expression.py /path/to/download/dir

Downloads two files and writes two:

  ENIGMA Toolbox extras (saratheriver/enigma-extra, ahba/allgenes_stable_r0.2.csv):
      Allen Human Brain Atlas microarray expression (Hawrylycz et al., Nature
      2012) mapped to the Desikan-Killiany atlas with abagen, genes with
      differential stability r > 0.2, values scaled 0-1 per gene.
  ENIGMA Toolbox surfaces/fsa5_sphere_lh.gii: the fsaverage5 sphere.

  ahba_dk34_lh.csv.gz   expression of every gene in the 34 left-hemisphere
                        regions. Left only: two of the six donors contributed
                        right-hemisphere samples, and two right regions are empty.
  dk34_lh_sphere.csv    each left-hemisphere region's centroid on the unit
                        sphere, for spin-permutation nulls.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from make_centroids import APARC, BASE, FILES, vertices

AHBA = "https://media.githubusercontent.com/media/saratheriver/enigma-extra/master/ahba/allgenes_stable_r0.2.csv"
HERE = Path(__file__).parent


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def main(download_dir: Path) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    expr = pd.read_csv(fetch(AHBA, download_dir / "allgenes_stable_r0.2.csv"),
                       on_bad_lines="skip").set_index("label")
    names = (HERE / "hcp_dk68_sc.csv").read_text().splitlines()[0].split(",")[1:]
    lh = [n for n in names if n.startswith("L_")]
    expr = expr.loc[lh]
    expr = expr.loc[:, expr.notna().all()]
    expr.round(4).to_csv(HERE / "ahba_dk34_lh.csv.gz", compression="gzip")

    sphere = vertices(fetch(BASE + "surfaces/fsa5_sphere_lh.gii", download_dir / "fsa5_sphere_lh.gii"))
    labels = np.loadtxt(fetch(BASE + FILES["aparc_fsa5.csv"], download_dir / "aparc_fsa5.csv"),
                        dtype=int)[: len(sphere)]
    rows = ["region,x,y,z"]
    for n in lh:
        c = sphere[labels == APARC.index(n[2:])].mean(axis=0)
        c /= np.linalg.norm(c)
        rows.append(f"{n},{c[0]:.5f},{c[1]:.5f},{c[2]:.5f}")
    (HERE / "dk34_lh_sphere.csv").write_text("\n".join(rows) + "\n")
    print(f"wrote {expr.shape[1]} genes x {len(lh)} regions, and sphere centroids")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("enigma_download"))
