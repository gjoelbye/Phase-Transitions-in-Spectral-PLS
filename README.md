# Phase Transitions in PLS-SVD with Missing Data

This repository contains code for reproducing the experiments in our ICML 2026 paper on phase transitions in Partial Least Squares Singular Value Decomposition (PLS-SVD) under dual MCAR (Missing Completely At Random) missingness.

## Theory

We study PLS-SVD in a spiked two-view model with dual MCAR missingness. Consider whitened design $X_\star \in \mathbb{R}^{N \times D_x}$ satisfying $X_\star^\top X_\star = N I_{D_x}$, and response

$$Y_\star = \theta (X_\star u_0) v_0^\top + Z, \quad Z_{ij} \overset{\text{iid}}{\sim} \mathcal{N}(0,1)$$

where $u_0, v_0$ are unit signal directions and $\theta \geq 0$ is the signal strength. We observe masked versions $X = S_x \odot X_\star$ and $Y = S_y \odot Y_\star$ with independent MCAR masks having retention probabilities $\rho_x = 1 - m_x$ and $\rho_y = 1 - m_y$.

### Main Result

PLS-SVD computes the leading singular vectors $(\hat{u}, \hat{v})$ of the cross-covariance $\hat{\Sigma}_{XY} = N^{-1} X^\top Y$. Recovery is measured via squared overlaps $R_x^2 = (\hat{u}^\top u_0)^2$ and $R_y^2 = (\hat{v}^\top v_0)^2$.

In the proportional limit with aspect ratios $\alpha_x = N/D_x$ and $\alpha_y = N/D_y$, there exists a sharp phase transition at

$$\theta_{\mathrm{crit}} = \frac{1}{(\alpha_x \alpha_y)^{1/4} \sqrt{\rho}}, \quad \rho = \rho_x \rho_y$$

- **Subcritical** ($\theta < \theta_{\mathrm{crit}}$): No recovery is possible — $R_x^2 = R_y^2 = 0$
- **Supercritical** ($\theta > \theta_{\mathrm{crit}}$): Recovery succeeds with overlaps

$$R_x^2 = \frac{\alpha_x \alpha_y \rho^2 \theta^4 - 1}{\alpha_y \rho \theta^2 (\alpha_x \rho \theta^2 + 1)}, \quad R_y^2 = \frac{\alpha_x \alpha_y \rho^2 \theta^4 - 1}{\alpha_x \rho \theta^2 (\alpha_y \rho \theta^2 + 1)}$$

Dual missingness acts as signal attenuation: the effective spike strength becomes $\theta_{\mathrm{eff}} = \sqrt{\rho} \, \theta$, increasing the required signal by $1/\sqrt{\rho}$ compared to full observation.

### Whitening Requirement

The theory requires $X_\star^\top X_\star = N I$ after preprocessing. Since MCAR masking destroys whitening, we must **rewhiten after masking**:

1. Apply MCAR masks: $X_{\mathrm{obs}} = S_x \odot X_\star$
2. Rewhiten: $X_w = X_{\mathrm{obs}} (X_{\mathrm{obs}}^\top X_{\mathrm{obs}} / N)^{-1/2}$
3. Compute cross-covariance and SVD on $(X_w, Y_{\mathrm{obs}})$

For real data, use `whiten_to_identity()` after dimensionality reduction.

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -r requirements.txt
```

## Quick Start

```python
from src import ModelParams, run_multiple_trials, theoretical_overlaps

# Define model parameters
params = ModelParams(
    N=500,      # Sample size
    Dx=200,     # Dimension of X
    Dy=150,     # Dimension of Y
    theta=2.0,  # Signal strength
    mx=0.3,     # 30% missing in X
    my=0.3      # 30% missing in Y
)

# Check critical threshold
print(f"theta_crit = {params.theta_crit:.3f}")
print(f"Is supercritical: {params.is_supercritical}")

# Run experiments
results = run_multiple_trials(params, n_trials=20)

# Compare theory vs empirical
print(f"Rx^2: Theory = {results['Rx2_theory']:.3f}, "
      f"Empirical = {results['Rx2_pls_mean']:.3f}")
```

## Reproducing Paper Figures

| Figure | Notebook | Description |
|--------|----------|-------------|
| Figure 1 | `experiments/fig1_phase_transition_validation.ipynb` | Phase transition curves (theory vs empirical) |
| Figure 2 | `experiments/fig2_missingness_effects.ipynb` | X-only vs joint missingness comparison |
| Figure 3 | `experiments/fig3_biological_validation.ipynb` | Semi-synthetic validation on TCGA and PBMC |
| Figure 4 | `experiments/fig4_split_half_diagnostics.ipynb` | Split-half stability diagnostics |
| Figure A1 | `experiments/figA1_robustness_analysis.ipynb` | Robustness to non-Gaussian noise |

Run notebooks with:
```bash
cd experiments
jupyter notebook
```

## Repository Structure

```
PhaseTransition/
|-- src/                          # Python package
|   |-- __init__.py               # Package exports
|   |-- core.py                   # ModelParams, theoretical_overlaps
|   |-- methods.py                # pls_svd, compute_overlaps, baselines
|   |-- data.py                   # Data generation utilities
|   |-- runners.py                # Experiment runners and diagnostics
|-- experiments/                  # Jupyter notebooks
|   |-- fig1_*.ipynb              # Phase transition validation
|   |-- fig2_*.ipynb              # Missingness effects
|   |-- fig3_*.ipynb              # Biological validation
|   |-- fig4_*.ipynb              # Split-half diagnostics
|   |-- figA1_*.ipynb             # Robustness analysis
|-- tests/
|   |-- test_setup.py             # Setup verification
|   |-- test_computations.py      # Computation tests
|   |-- test_predictions.py       # Theory prediction tests
|   |-- test_theory_computations.py
|-- figures/                      # Generated figures (PDF)
|-- results/                      # Experiment results cache
|-- data/                         # Downloaded datasets
|-- Latex/                        # Paper source (not included)
|-- pyproject.toml
|-- requirements.txt
|-- LICENSE
```

## Data

The notebooks download data automatically when needed:

- **TCGA BRCA**: Gene expression and copy number data from UCSC Xena
- **PBMC Multiome 10k**: Single-cell RNA and ATAC from 10x Genomics

Data is cached in `data/` and excluded from git.

## Citation

```bibtex
@inproceedings{anonymous2026icml,
  title={Missing-Data-Induced Phase Transitions in Spectral PLS for Multimodal Learning},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## Acknowledgments

Acknowledgments withheld for anonymous review.

## License

MIT License - see LICENSE file.
