# Connectome Spread

Interactive simulation of neurodegenerative pathology spreading along the brain's
structural wiring, with APOE genotype as a tunable risk parameter.

**Site:** https://oluwatobaoyagbemi.github.io/connectome-spread  
**Simulator:** https://connectome-spread.streamlit.app

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
4. Arrival table with each region's Braak stage, so order can be checked by eye
5. **β sensitivity sweep**: Braak concordance across spread rates, for every genotype
6. KPI row: total load vs ε3/ε3, regions reached, Braak concordance ρ, median arrival

## Data

Default: a **real group-average structural connectome** from Human Connectome
Project diffusion MRI, 68 cortical regions of the Desikan-Killiany atlas, bundled
in `data/hcp_dk68_sc.csv` (via the ENIGMA Toolbox, BSD-3; see `data/README.md`).

Also available: a synthetic lobe-based toy connectome for comparison, or upload
your own CSV (region labels in the first row and column).

## A first result

Seeding both entorhinal cortices on the HCP connectome (β = 0.05, κ = 0.01), the
simulated arrival order correlates with Braak tau staging at **Spearman ρ ≈ 0.64**:
temporal pole, parahippocampal and fusiform regions arrive first, the medial-temporal
territory of early Braak stages. The model was given only the wiring and a seed; the
ordering emerged from the network.

| genotype | total load (300 steps) | regions reached | Braak concordance ρ |
|---|---|---|---|
| ε2/ε2 | 5.6 | 8 | 0.55 |
| ε3/ε3 | 7.3 | 24 | 0.64 |
| ε4/ε4 | 11.5 | 63 | 0.54 |

**This result holds in a parameter window, not everywhere.** The arrival order is
sensitive to β: at low β pathology creeps through near neighbours and matches Braak
order; at high β strongly connected distant hubs (e.g. superior parietal) jump the
queue. The app's *β sensitivity* tab sweeps β from 0.01 to 0.20 and reports the
range over which concordance stays ≥ 0.5 — for ε3/ε3 that is roughly β ∈ [0.01, 0.10],
peaking near 0.02–0.05.

## Limitations

- APOE multipliers are illustrative, chosen to reflect the direction of known
  effects, not fitted to imaging or biomarker data.
- Group-average connectome: no individual variation in wiring.
- Linear diffusion ignores regional vulnerability, protein-specific kinetics, and
  vascular contributions (the brain–heart axis this project grew out of).

## Next steps

- Add region-level vulnerability weights from gene-expression atlases
- Fit β and κ per genotype to longitudinal tau-PET data

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Author

Oluwatoba Oyagbemi — Biology / CS, CCBC Honors College. Part of a portfolio on
AI applied to cardiovascular and neurodegenerative disease.
