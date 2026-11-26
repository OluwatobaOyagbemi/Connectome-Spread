"""Regenerate dk68_centroids.csv: one (x, y, z) per Desikan-Killiany region.

    python data/make_centroids.py /path/to/download/dir

Downloads three files from the ENIGMA Toolbox (the same source as the
connectome; BSD-3): the fsaverage5 left/right surfaces and the per-vertex
Desikan-Killiany labels. Each region's centroid is the mean position of its
surface vertices, in fsaverage (MNI305) millimetres. Only numpy and the
standard library are needed.
"""

from __future__ import annotations

import base64
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np

BASE = "https://raw.githubusercontent.com/MICA-MNI/ENIGMA/master/enigmatoolbox/datasets/"
FILES = {"fsa5_lh.gii": "surfaces/fsa5_lh.gii", "fsa5_rh.gii": "surfaces/fsa5_rh.gii",
         "aparc_fsa5.csv": "parcellations/aparc_fsa5.csv"}
# FreeSurfer aparc label order (values 1-35 left hemisphere, 36-70 right).
APARC = ["unknown", "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal",
         "corpuscallosum", "cuneus", "entorhinal", "fusiform", "inferiorparietal",
         "inferiortemporal", "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal",
         "lingual", "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral",
         "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine", "postcentral",
         "posteriorcingulate", "precentral", "precuneus", "rostralanteriorcingulate",
         "rostralmiddlefrontal", "superiorfrontal", "superiorparietal", "superiortemporal",
         "supramarginal", "frontalpole", "temporalpole", "transversetemporal", "insula"]


def vertices(path: Path) -> np.ndarray:
    for da in ET.parse(path).getroot().iter("DataArray"):
        if da.get("Intent") == "NIFTI_INTENT_POINTSET":
            raw = base64.b64decode(da.find("Data").text)
            if "gzip" in da.get("Encoding", "").lower():
                raw = zlib.decompress(raw)
            return np.frombuffer(raw, dtype="<f4").reshape(-1, 3).astype(float)
    raise ValueError(f"no vertex array in {path}")


def main(download_dir: Path) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    for local, remote in FILES.items():
        if not (download_dir / local).exists():
            urllib.request.urlretrieve(BASE + remote, download_dir / local)
    labels = np.loadtxt(download_dir / "aparc_fsa5.csv", dtype=int)
    xyz = np.vstack([vertices(download_dir / "fsa5_lh.gii"),
                     vertices(download_dir / "fsa5_rh.gii")])
    assert len(labels) == len(xyz)
    names = (Path(__file__).parent / "hcp_dk68_sc.csv").read_text().splitlines()[0].split(",")[1:]
    out = ["region,x,y,z"]
    for name in names:
        hemi, base = name.split("_", 1)
        mask = labels == APARC.index(base) + (0 if hemi == "L" else 35)
        assert mask.any(), name
        x, y, z = xyz[mask].mean(axis=0)
        out.append(f"{name},{x:.2f},{y:.2f},{z:.2f}")
    dest = Path(__file__).parent / "dk68_centroids.csv"
    dest.write_text("\n".join(out) + "\n")
    print(f"wrote {dest} ({len(names)} regions)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("enigma_download"))
