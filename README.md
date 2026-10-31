# Connectome Spread

Interactive simulation of neurodegenerative pathology spreading along the brain's
structural wiring, with APOE genotype as a tunable risk parameter.

**Live demo:** https://connectome-spread.streamlit.app

## Question

Alzheimer's pathology does not appear everywhere at once. It starts in a few
regions (classically the entorhinal cortex) and spreads to connected regions.
If spread follows the wiring, then a person's brain network plus their biggest
genetic risk factor, APOE, should shape *where* and *how fast* pathology arrives.
This app lets you see that.

## Model

Network diffusion on a connectome (after Raj, Kuceyeski & Weiner, *Neuron* 2012):

```
dx/dt = −β · L·x − κ · x + seed
```

| term | meaning |
|---|---|
| `x` | pathology load per region |
| `L` | normalised graph Laplacian of the connectome `W` |
| `β` | spread rate along connections |
| `κ` | clearance rate |
| `seed` | constant production in the chosen seed region(s) |

APOE genotype scales `β` and `κ`: ε4 → faster spread, weaker clearance;
ε2 → the reverse; ε3/ε3 is the reference.

## What the app shows

1. Total pathology over time for all five genotypes, with the selected one highlighted
2. Final regional load ranked, coloured by lobe
3. Heatmap of every region over time, sorted by arrival order
4. Arrival table: the step at which each region crosses a threshold

## Data

Default: a **real group-average structural connectome** from Human Connectome
Project diffusion MRI, 68 cortical regions of the Desikan-Killiany atlas, bundled
in `data/hcp_dk68_sc.csv` (via the ENIGMA Toolbox, BSD-3; see `data/README.md`).

Also available: a synthetic lobe-based toy connectome for comparison, or upload
your own CSV (region labels in the first row and column).

## A first result

Seeding the left entorhinal cortex on the HCP connectome, the first regions
reached are temporal pole, parahippocampal, fusiform and insula — the
medial-temporal territory of early Braak staging. The model was given only the
wiring and a seed; the ordering emerged from the network.

| genotype | total load (300 steps) | regions reached |
|---|---|---|
| ε2/ε2 | 2.8 | 1 |
| ε3/ε3 | 3.7 | 5 |
| ε4/ε4 | 5.7 | 14 |

## Limitations

- APOE multipliers are illustrative, chosen to reflect the direction of known
  effects, not fitted to imaging or biomarker data.
- Group-average connectome: no individual variation in wiring.
- Linear diffusion ignores regional vulnerability, protein-specific kinetics, and
  vascular contributions (the brain–heart axis this project grew out of).

## Next steps

- Add region-level vulnerability weights from gene-expression atlases
- Quantify agreement between simulated arrival order and published Braak stages
- Fit β and κ per genotype to longitudinal tau-PET data

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Author

Oluwatoba Oyagbemi — Biology / CS, CCBC Honors College. Part of a portfolio on
AI applied to cardiovascular and neurodegenerative disease.
