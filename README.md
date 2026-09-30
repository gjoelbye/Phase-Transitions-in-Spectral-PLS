# Missing-Data-Induced Phase Transitions in Spectral Partial Least Squares

Anders Gjølbye, Emma Kargaard, Ida Kargaard, Lina Skerath, Hiba Nassar and Lars Kai Hansen\
Technical University of Denmark

**Paper:** [arXiv:2601.21294](https://arxiv.org/abs/2601.21294)

This repository has the code for every figure in the paper, along with the saved
simulation results. We study spectral partial least squares (PLS-SVD) when entries
of both views are missing completely at random and filled in with zeros.

## The model

The complete design $X_\star \in \mathbb{R}^{N \times D_x}$ is whitened, and the
complete response is a rank-one signal plus Gaussian noise,

```math
X_\star^\top X_\star = N I_{D_x},
\qquad
Y_\star = \theta \, (X_\star u_0) \, v_0^\top + Z,
\qquad
Z_{ij} \sim \mathcal{N}(0, 1).
```

Each entry of $X_\star$ is kept with probability $\rho_x$ and each entry of
$Y_\star$ with probability $\rho_y$, independently. Missing entries are set to
zero, so we observe $X = S_x \odot X_\star$ and $Y = S_y \odot Y_\star$ for binary
masks $S_x$ and $S_y$. PLS-SVD takes the leading singular vectors
$(\hat u, \hat v)$ of $X^\top Y$ as estimates of $u_0$ and $v_0$, and we measure
recovery by the squared overlaps

```math
R_x^2 = \langle \hat u, u_0 \rangle^2,
\qquad
R_y^2 = \langle \hat v, v_0 \rangle^2.
```

The estimator is `pls_svd` in `src/methods.py`, and the theoretical curves are
computed in `src/theory.py`.

## Installation

We used Python 3.14.

```bash
pip install -r requirements.txt
```

## Reproducing the figures

Every figure has a simulation script in `scripts/` and a notebook in `notebooks/`.
The script saves its results to `results/`. The notebook loads them, draws the
figure into `figures/` and prints the numbers we quote in the paper. Since all
the results are already in the repository, the notebooks run in a few seconds.

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

Table 2 is printed by the Figure 8 notebook.

To rerun a simulation, run its script as a module from the repository root. For
example, with 8 worker processes:

```bash
python -m scripts.direction_geometry --workers 8
```

The parameters are in the `PARAMS` dictionary at the top of each script. Each
trial has its own seed, so the results are the same for any number of workers.
Some of the simulations take several hours.

## Biological data

Figure 8 and Table 2 use two public datasets that are not included here. You
only need them to rerun the simulation, not to run the notebook. The script
looks for the files at

```text
data/xena/HiSeqV2.gz
data/xena/HumanMethylation450.gz
data/pbmc_multiome_10k/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5
```

The first two are the TCGA BRCA gene expression (RNA-seq) and DNA methylation
(450k) matrices from the UCSC Xena TCGA hub. The third is the 10x Genomics PBMC
multiome dataset (granulocyte-sorted, 10k cells).

## Citation

If you use this code, please cite the paper.

```bibtex
@misc{gjolbye2026phase,
  title         = {Missing-Data-Induced Phase Transitions in Spectral Partial Least Squares},
  author        = {Gj{\o}lbye, Anders and Kargaard, Emma and Kargaard, Ida and Skerath, Lina and Nassar, Hiba and Hansen, Lars Kai},
  year          = {2026},
  eprint        = {2601.21294},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2601.21294}
}
```

## License

MIT. See `LICENSE`.
