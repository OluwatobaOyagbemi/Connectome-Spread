# Data

`hcp_dk68_sc.csv` — group-average structural connectome, 68 cortical regions of the
Desikan-Killiany atlas (34 per hemisphere), derived from Human Connectome Project
diffusion MRI. Values are log-transformed streamline counts; higher = stronger
white-matter connection.

Source: ENIGMA Toolbox (Larivière et al., *Nature Methods* 2021),
https://github.com/MICA-MNI/ENIGMA, `enigmatoolbox/datasets/matrices/hcp_connectivity/`
(`strucMatrix_ctx.csv`, `strucLabels_ctx.csv`). Redistributed under the BSD 3-Clause
license; see `LICENSE-ENIGMA`.

`dk68_centroids.csv` — (x, y, z) centroid of each of the 68 regions, in fsaverage
(MNI305) millimetres: the mean position of the region's vertices on the fsaverage5
surface. Used only for the distance controls (`validation.py`). Built by
`make_centroids.py` from three ENIGMA Toolbox files (same source and license as
above): `enigmatoolbox/datasets/surfaces/fsa5_{lh,rh}.gii` and
`enigmatoolbox/datasets/parcellations/aparc_fsa5.csv`.

`atlases/<atlas>_sc.csv`, `atlases/<atlas>_parcels.csv` — HCP structural
connectomes for Schaefer 100/200/300/400 and Glasser 360 (ENIGMA Toolbox,
`strucMatrix_ctx_<atlas>.csv` / `strucLabels_ctx_<atlas>.csv`, same pipeline and
license as above), plus per-parcel centroid, parent Desikan-Killiany region
(majority vertex overlap on fsaverage5) and entorhinal fraction. Built by
`make_atlases.py`; used by `resolution.py`.
