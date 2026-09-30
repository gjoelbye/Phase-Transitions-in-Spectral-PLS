"""Synthetic designs, masked samples and biological data loaders."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import hadamard

from .methods import inv_sqrtm_psd, pls_svd


@dataclass
class ModelParams:
    """Dimensions, signal strength, missingness rates and planted directions."""
    N: int
    Dx: int
    Dy: int
    theta: float
    mx: float
    my: float
    u0: np.ndarray
    v0: np.ndarray

    @property
    def alpha_x(self):
        return self.N / self.Dx

    @property
    def alpha_y(self):
        return self.N / self.Dy

    @property
    def rho_x(self):
        return 1 - self.mx

    @property
    def rho_y(self):
        return 1 - self.my


@lru_cache(maxsize=4)
def _hadamard_base(N: int) -> np.ndarray:
    """Walsh-Hadamard matrix of order N, cached across trials."""
    return hadamard(N).astype(float)


def generate_flat_orthogonal(N: int, Dx: int, rng: np.random.Generator) -> np.ndarray:
    """Randomized Walsh-Hadamard design with entries +-1 and ``X.T @ X = N I``.

    Rows and columns of the order-N Hadamard matrix are permuted, Dx columns
    are kept and rows and columns get random signs. N must be a power of two.
    """
    H = _hadamard_base(N)
    H = H[rng.permutation(N), :][:, rng.permutation(N)[:Dx]]
    H *= rng.choice([-1.0, 1.0], size=(N, 1))
    H *= rng.choice([-1.0, 1.0], size=(1, Dx))
    return H


def planted_directions(Dx: int, Dy: int, rng: np.random.Generator):
    """Random unit vectors u0 in R^Dx and v0 in R^Dy."""
    u0 = rng.normal(size=Dx)
    v0 = rng.normal(size=Dy)
    return u0 / np.linalg.norm(u0), v0 / np.linalg.norm(v0)


def whiten_to_identity(X: np.ndarray) -> np.ndarray:
    """Mean-center and whiten ``X`` so that ``X.T @ X / N`` is identity."""
    X_centered = X - X.mean(axis=0, keepdims=True)
    N = X_centered.shape[0]
    cov = (X_centered.T @ X_centered) / N
    w, V = np.linalg.eigh(cov)
    w = np.maximum(w, 1e-12)
    return (X_centered @ V) / np.sqrt(w)


def generate_data(params: ModelParams, seed: int):
    """One masked sample (X, Y, Sx, Sy) from the spiked two-view model.

    The complete design is a Haar-random orthogonal matrix scaled to
    ``X_star.T @ X_star = N I``, and ``Y_star = theta X_star u0 v0.T + Z``.
    """
    np.random.seed(seed)
    X_star = np.random.randn(params.N, params.Dx)
    Q, _ = np.linalg.qr(X_star)
    X_star = Q * np.sqrt(params.N)
    signal = params.theta * np.outer(X_star @ params.u0, params.v0)
    noise = np.random.randn(params.N, params.Dy)
    Y_star = signal + noise
    Sx = np.random.binomial(1, params.rho_x, size=(params.N, params.Dx))
    Sy = np.random.binomial(1, params.rho_y, size=(params.N, params.Dy))
    X = Sx * X_star
    Y = Sy * Y_star
    return X, Y, Sx, Sy


def masked_trials(params: ModelParams, n_trials: int):
    """Squared left overlap and top singular value of PLS-SVD over seeded trials.

    Trial t draws its sample with ``generate_data(params, seed=t)``.
    """
    rx2 = np.empty(n_trials)
    sigma1 = np.empty(n_trials)
    for trial in range(n_trials):
        X, Y, _, _ = generate_data(params, seed=trial)
        u_hat, _, s1 = pls_svd(X, Y)
        rx2[trial] = float(u_hat @ params.u0) ** 2
        sigma1[trial] = s1
    return rx2, sigma1


def generate_data_non_gaussian(params: ModelParams, noise_type: str, seed: int):
    """As ``generate_data``, with unit-variance noise from the named law."""
    np.random.seed(seed)

    X_star = np.random.randn(params.N, params.Dx)
    Q, _ = np.linalg.qr(X_star)
    X_star = Q * np.sqrt(params.N)

    if noise_type == 'gaussian':
        noise = np.random.randn(params.N, params.Dy)
    elif noise_type == 't5':
        raw = np.random.standard_t(df=5, size=(params.N, params.Dy))
        noise = raw / np.sqrt(5 / 3)
    elif noise_type == 't4.5':
        raw = np.random.standard_t(df=4.5, size=(params.N, params.Dy))
        noise = raw / np.sqrt(4.5 / 2.5)
    elif noise_type == 't3':
        raw = np.random.standard_t(df=3, size=(params.N, params.Dy))
        noise = raw / np.sqrt(3)
    elif noise_type == 'laplace':
        noise = np.random.laplace(loc=0, scale=1/np.sqrt(2), size=(params.N, params.Dy))
    elif noise_type == 'heteroskedastic':
        # Shared column variances give unit marginal variance and excess kurtosis 0.25.
        column_variances = np.random.uniform(0.5, 1.5, size=params.Dy)
        noise = np.random.randn(params.N, params.Dy) * np.sqrt(column_variances)[None, :]

    signal = params.theta * np.outer(X_star @ params.u0, params.v0)
    Y_star = signal + noise
    Sx = np.random.binomial(1, params.rho_x, size=(params.N, params.Dx))
    Sy = np.random.binomial(1, params.rho_y, size=(params.N, params.Dy))
    X = Sx * X_star
    Y = Sy * Y_star
    return X, Y, Sx, Sy


def generate_heterogeneous_whitened(N: int, Dx: int, row_scales, seed: int) -> np.ndarray:
    """Whitened design with heterogeneous row norms.

    ``row_scales`` sets the row scales before whitening. Whitening attenuates
    their contrast, so the realized spread is measured from the returned matrix.
    """
    row_scales = np.asarray(row_scales, dtype=float)
    row_scales = row_scales / np.sqrt(np.mean(row_scales ** 2))

    rng = np.random.default_rng(seed)
    G = row_scales[:, None] * rng.standard_normal((N, Dx))
    W = inv_sqrtm_psd(G.T @ G / N, eps=1e-12)
    return G @ W


def leverage_coupled_direction(X_star: np.ndarray, weight: float, seed: int) -> np.ndarray:
    """Unit direction with controlled score-leverage coupling.

    A weight in [-1, 1] interpolates from a random direction toward the leading
    (weight > 0) or trailing (weight < 0) eigenvector of the leverage-weighted
    Gram matrix.
    """
    N = X_star.shape[0]
    r = np.sum(X_star ** 2, axis=1)
    M = X_star.T @ (r[:, None] * X_star) / N
    _, evecs = np.linalg.eigh(M)
    target = evecs[:, -1] if weight >= 0 else evecs[:, 0]

    rng = np.random.default_rng(seed)
    g = rng.standard_normal(X_star.shape[1])
    g /= np.linalg.norm(g)
    u = (1.0 - abs(weight)) * g + abs(weight) * target
    return u / np.linalg.norm(u)


# Biological designs use seeded PCA to fix component order and signs.

def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def load_tcga_whitened(Dx: int = 200, Dy: int = 200, pca_seed: int = 42):
    """TCGA-BRCA expression against methylation, both whitened to identity.

    Reads the two Xena matrices from ``data/xena``, transposes them to
    samples x features, keeps the samples present in both, drops feature
    columns with missing values, standardises, reduces to ``Dx``/``Dy``
    principal components and whitens.
    """
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    root = _data_dir() / "xena"
    expr = pd.read_csv(root / "HiSeqV2.gz", sep="\t", index_col=0,
                       compression="gzip").T
    meth = pd.read_csv(root / "HumanMethylation450.gz", sep="\t", index_col=0,
                       compression="gzip").T
    common = expr.index.intersection(meth.index)
    X = expr.loc[common].dropna(axis=1).values.astype(np.float64)
    Y = meth.loc[common].dropna(axis=1).values.astype(np.float64)

    X = StandardScaler().fit_transform(X)
    Y = StandardScaler().fit_transform(Y)
    X = PCA(n_components=Dx, random_state=pca_seed).fit_transform(X)
    Y = PCA(n_components=Dy, random_state=pca_seed).fit_transform(Y)
    return whiten_to_identity(X), whiten_to_identity(Y)


def load_pbmc_whitened(Dx: int = 200, Dy: int = 200, pca_seed: int = 42):
    """PBMC multiome RNA against ATAC, both whitened to identity.

    Reads the 10x multiome matrix from ``data/pbmc_multiome_10k``. RNA is
    filtered, normalized, log-transformed and reduced to 2000 highly variable
    genes; ATAC is filtered and its first 5000 peaks are kept. 5000 cells
    present in both are sampled, and each view is standardized, reduced to
    ``Dx``/``Dy`` principal components and whitened.
    """
    import muon as mu
    import scanpy as sc
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    path = (_data_dir() / "pbmc_multiome_10k"
            / "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5")
    mdata = mu.read_10x_h5(str(path))
    rna, atac = mdata.mod["rna"], mdata.mod["atac"]

    sc.pp.filter_cells(rna, min_genes=200)
    sc.pp.filter_genes(rna, min_cells=3)
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, n_top_genes=2000)
    rna = rna[:, rna.var["highly_variable"]]

    sc.pp.filter_cells(atac, min_genes=200)
    sc.pp.filter_genes(atac, min_cells=3)

    common = rna.obs_names.intersection(atac.obs_names)
    selected = np.random.default_rng(42).choice(common, size=5000, replace=False)
    X = rna[selected].X.toarray()
    Y = atac[selected].X.toarray()[:, :5000]

    X = StandardScaler().fit_transform(X)
    Y = StandardScaler().fit_transform(Y)
    X = PCA(n_components=Dx, random_state=pca_seed).fit_transform(X)
    Y = PCA(n_components=Dy, random_state=pca_seed).fit_transform(Y)
    return whiten_to_identity(X), whiten_to_identity(Y)
