# Missing-Data-Induced Phase Transitions\\in Spectral Partial Least Squares

Code for the nine figures and Table 2 of the paper. It covers zero-filled PLS-SVD
under independent entry-wise missingness in both views.

## Installation

The code was run with Python 3.14.

```bash
pip install -r requirements.txt
```

## Model

The complete design is whitened, `X_star.T @ X_star = N I`, and the response is
`Y_star = theta (X_star u0) v0.T + Z` with standard Gaussian noise `Z`. The
observed views are `X = S_x * X_star` and `Y = S_y * Y_star`, where the masks
`S_x`, `S_y` keep each entry independently with probability `rho_x`, `rho_y`.
The estimator of Eq. (3) is the leading singular pair of `X.T @ Y`, computed by
`src.methods.pls_svd`. The theoretical predictions are in `src/theory.py`.

## Figures

Each figure has a script and a notebook. The script `scripts/<name>.py` runs the
simulations and saves `results/<name>.pkl`. The notebook loads that file, draws
the figure, writes it to `figures/` and prints the numbers quoted in the paper.
The results of all simulations are included, so the notebooks run in seconds.

| Figure | Script | Notebook |
|---|---|---|
| 1 | `phase_transition_validation.py` | `fig1_phase_transition_validation.ipynb` |
| 2 | `direction_geometry.py` | `fig2_direction_geometry.ipynb` |
| 3 | `retention_phase_diagram.py` | `fig3_retention_phase_diagram.ipynb` |
| 4 | `noise_distribution_robustness.py` | `fig4_noise_distribution_robustness.ipynb` |
| 5 | `estimator_comparison.py` | `fig5_estimator_comparison.ipynb` |
| 6 | `correlated_noise.py` | `fig6_correlated_noise.ipynb` |
| 7 | `below_isotropic_threshold.py` | `fig7_below_isotropic_threshold.ipynb` |
| 8 | `direction_specific_ceilings.py` | `fig8_direction_specific_ceilings.ipynb` |
| 9 | `aspect_ratio_sensitivity.py` | `fig9_aspect_ratio_sensitivity.ipynb` |

The Figure 8 notebook also displays Table 2.

To rerun a simulation, run its script as a module from the repository root,
for example with 8 worker processes:

```bash
python -m scripts.direction_geometry --workers 8
```

The parameters are in the `PARAMS` dictionary at the top of each script. Every
trial has its own seed, so the results do not depend on the number of workers.

## Biological data

Figure 8 and Table 2 use two public datasets. The simulation script expects them at:

```text
data/xena/HiSeqV2.gz
data/xena/HumanMethylation450.gz
data/pbmc_multiome_10k/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5
```

The first two files are the TCGA BRCA gene expression (RNA-seq) and DNA
methylation (450k) matrices from the UCSC Xena TCGA hub. The third is the 10x
Genomics PBMC multiome dataset (granulocyte-sorted, 10k cells). The notebook
needs only the included results.

## Layout

```text
scripts/        one simulation script per figure
notebooks/      one notebook per figure
src/            theory, estimators, data generation, threshold statistics,
                parallel runs and result files, plotting style
results/        simulation results
figures/        figure PDFs
data/           biological inputs, not included
```

## License

MIT License. See `LICENSE`.
