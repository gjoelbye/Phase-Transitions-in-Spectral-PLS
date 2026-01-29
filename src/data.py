"""
Data generation utilities.

This module contains:
- whiten_to_identity: eigen-whiten real data
- apply_mcar: apply MCAR missingness to existing data
- generate_data: generate synthetic data with MCAR missingness
- generate_data_non_gaussian: data with non-Gaussian noise
- generate_semi_synthetic: semi-synthetic data using real X
"""

import numpy as np
from typing import Tuple, Optional

from .core import ModelParams


def whiten_to_identity(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Mean-center and eigen-whiten so that (1/N) X^T X ~ I.

    This is needed when working with real data (e.g., after PCA/LSI reduction)
    to satisfy the theoretical assumption X_star^T X_star = N I.

    Args:
        X: Data matrix (N x D)
        eps: Small constant for numerical stability

    Returns:
        X_whitened: Whitened matrix (N x D) where (1/N) X_whitened^T X_whitened ~ I
    """
    # Center
    X_centered = X - X.mean(axis=0, keepdims=True)
    N = X_centered.shape[0]

    # Compute empirical covariance
    cov = (X_centered.T @ X_centered) / N

    # Eigendecomposition
    w, V = np.linalg.eigh(cov)
    w = np.maximum(w, eps)  # Avoid division by tiny eigenvalues

    # Whiten: X_whitened = X_centered @ V @ diag(1/sqrt(w))
    X_whitened = (X_centered @ V) / np.sqrt(w)

    return X_whitened


def apply_mcar(
    X: np.ndarray,
    Y: np.ndarray,
    mx: float,
    my: Optional[float] = None,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply MCAR (Missing Completely At Random) missingness to existing data.

    This is useful for real data experiments where we want to simulate missingness
    on already preprocessed/whitened data.

    Args:
        X: Data matrix X (N x Dx)
        Y: Data matrix Y (N x Dy)
        mx: Missingness rate for X (fraction of entries to mask)
        my: Missingness rate for Y (if None, uses mx for symmetric missingness)
        seed: Random seed for reproducibility

    Returns:
        X_obs: X with missing entries set to zero (N x Dx)
        Y_obs: Y with missing entries set to zero (N x Dy)
        Sx: Mask for X (1 = observed, 0 = missing)
        Sy: Mask for Y (1 = observed, 0 = missing)
    """
    if my is None:
        my = mx

    rng = np.random.default_rng(seed)

    # Generate MCAR masks
    Sx = rng.binomial(1, 1 - mx, size=X.shape)
    Sy = rng.binomial(1, 1 - my, size=Y.shape)

    # Apply masks (missing-as-zero)
    X_obs = X * Sx
    Y_obs = Y * Sy

    return X_obs, Y_obs, Sx, Sy


def generate_data(
    params: ModelParams,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data under the spiked two-view model with MCAR missingness.

    Model:
        X_star^T X_star = N I_Dx  (whitened design)
        Y_star = theta (X_star u0) v0^T + Z,  Z_ij ~ N(0,1)
        X = S_x * X_star,  Y = S_y * Y_star  (missing-as-zero)

    Args:
        params: Model parameters
        seed: Random seed for reproducibility

    Returns:
        X: Observed design matrix (N x Dx) with zeros for missing entries
        Y: Observed response matrix (N x Dy) with zeros for missing entries
        Sx: Mask for X (1 = observed, 0 = missing)
        Sy: Mask for Y (1 = observed, 0 = missing)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate whitened design X_star such that X_star^T X_star = N I_Dx
    # Method: Generate random matrix, QR decompose, scale by sqrt(N)
    X_star = np.random.randn(params.N, params.Dx)
    Q, R = np.linalg.qr(X_star)
    X_star = Q * np.sqrt(params.N)

    # Generate response Y_star = theta (X_star u0) v0^T + Z
    signal = params.theta * np.outer(X_star @ params.u0, params.v0)
    noise = np.random.randn(params.N, params.Dy)
    Y_star = signal + noise

    # Generate MCAR masks
    Sx = np.random.binomial(1, params.rho_x, size=(params.N, params.Dx))
    Sy = np.random.binomial(1, params.rho_y, size=(params.N, params.Dy))

    # Apply masks (missing-as-zero)
    X = Sx * X_star
    Y = Sy * Y_star

    return X, Y, Sx, Sy


def generate_data_non_gaussian(
    params: ModelParams,
    noise_type: str = 'gaussian',
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data under the spiked two-view model with non-Gaussian noise.

    Supports various noise distributions for robustness experiments.

    Args:
        params: Model parameters
        noise_type: One of 'gaussian', 't5', 't4.5', 't3', 'laplace', 'heteroskedastic'
        seed: Random seed for reproducibility

    Returns:
        X: Observed design matrix (N x Dx) with zeros for missing entries
        Y: Observed response matrix (N x Dy) with zeros for missing entries
        Sx: Mask for X (1 = observed, 0 = missing)
        Sy: Mask for Y (1 = observed, 0 = missing)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate whitened design X_star such that X_star^T X_star = N I_Dx
    X_star = np.random.randn(params.N, params.Dx)
    Q, R = np.linalg.qr(X_star)
    X_star = Q * np.sqrt(params.N)

    # Generate noise based on type
    if noise_type == 'gaussian':
        noise = np.random.randn(params.N, params.Dy)
    elif noise_type == 't5':
        # Student-t with df=5, scaled to unit variance
        raw = np.random.standard_t(df=5, size=(params.N, params.Dy))
        noise = raw / np.sqrt(5 / 3)
    elif noise_type == 't4.5':
        raw = np.random.standard_t(df=4.5, size=(params.N, params.Dy))
        noise = raw / np.sqrt(4.5 / 2.5)
    elif noise_type == 't3':
        # Student-t with df=3, scaled to unit variance
        raw = np.random.standard_t(df=3, size=(params.N, params.Dy))
        noise = raw / np.sqrt(3)
    elif noise_type == 'laplace':
        # Laplace with unit variance: scale = 1/sqrt(2)
        noise = np.random.laplace(loc=0, scale=1/np.sqrt(2), size=(params.N, params.Dy))
    elif noise_type == 'heteroskedastic':
        # Gaussian with random variance per entry
        sigmas = np.random.uniform(0.5, 1.5, size=(params.N, params.Dy))
        noise = np.random.randn(params.N, params.Dy) * sigmas
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    # Generate response Y_star = theta (X_star u0) v0^T + noise
    signal = params.theta * np.outer(X_star @ params.u0, params.v0)
    Y_star = signal + noise

    # Generate MCAR masks
    Sx = np.random.binomial(1, params.rho_x, size=(params.N, params.Dx))
    Sy = np.random.binomial(1, params.rho_y, size=(params.N, params.Dy))

    # Apply masks (missing-as-zero)
    X = Sx * X_star
    Y = Sy * Y_star

    return X, Y, Sx, Sy


def generate_semi_synthetic(
    X_real: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    theta: float,
    mx: float,
    my: float,
    Dy: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate semi-synthetic data using real X and biological signal directions.

    Uses real whitened X data but generates synthetic Y with controlled
    signal strength and Gaussian noise.

    Args:
        X_real: Whitened real data matrix (N x Dx), should satisfy X^T X ~ N I
        u0: Signal direction in X space (Dx,), unit norm
        v0: Signal direction in Y space (Dy,), unit norm
        theta: Signal strength
        mx: Missingness rate in X
        my: Missingness rate in Y
        Dy: Dimension of Y (can differ from len(v0) for padding)
        seed: Random seed

    Returns:
        X: Observed X with MCAR missingness (N x Dx)
        Y: Observed Y with MCAR missingness (N x Dy)
        Sx: Mask for X
        Sy: Mask for Y
    """
    if seed is not None:
        np.random.seed(seed)

    N, Dx = X_real.shape

    # Generate Y_star = theta (X_real u0) v0^T + Z
    latent = X_real @ u0  # (N,)
    signal = theta * np.outer(latent, v0)  # (N, Dy)
    noise = np.random.randn(N, Dy)
    Y_star = signal + noise

    # Generate MCAR masks
    Sx = np.random.binomial(1, 1 - mx, size=(N, Dx))
    Sy = np.random.binomial(1, 1 - my, size=(N, Dy))

    # Apply masks
    X = Sx * X_real
    Y = Sy * Y_star

    return X, Y, Sx, Sy


def generate_data_mar(
    params: ModelParams,
    mar_type: str = 'signal_dependent',
    mar_strength: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data with MAR (Missing At Random) missingness.

    Unlike MCAR, MAR means missingness probability depends on observed values.
    This tests robustness of MCAR-derived theory to realistic violations.

    Args:
        params: Model parameters (mx, my are used as base missingness rates)
        mar_type: Type of MAR mechanism:
            - 'signal_dependent': Miss probability depends on |X @ u0| (signal strength)
            - 'magnitude_dependent': Miss probability depends on |X_ij| (entry magnitude)
            - 'thresholded': High probability of missing entries above threshold
            - 'correlated': Y missingness depends on X values
        mar_strength: How strongly missingness depends on values (0 = MCAR, 1 = strong MAR)
        seed: Random seed for reproducibility

    Returns:
        X: Observed design matrix with MAR missingness
        Y: Observed response matrix with MAR missingness
        Sx: Mask for X (1 = observed, 0 = missing)
        Sy: Mask for Y (1 = observed, 0 = missing)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate whitened design X_star
    X_star = np.random.randn(params.N, params.Dx)
    Q, R = np.linalg.qr(X_star)
    X_star = Q * np.sqrt(params.N)

    # Generate response Y_star = theta (X_star u0) v0^T + Z
    signal = params.theta * np.outer(X_star @ params.u0, params.v0)
    noise = np.random.randn(params.N, params.Dy)
    Y_star = signal + noise

    # Compute latent signal for MAR mechanisms
    latent_x = X_star @ params.u0  # (N,) - latent score per sample
    latent_y = Y_star @ params.v0  # (N,) - latent score per sample

    # Base missingness rates
    mx_base = params.mx
    my_base = params.my

    if mar_type == 'signal_dependent':
        # Missingness probability increases with |latent signal|
        # P(miss) = mx_base + mar_strength * (|latent| - mean) / (2 * std)
        # Clipped to [0.01, 0.99]
        latent_x_norm = (np.abs(latent_x) - np.mean(np.abs(latent_x))) / (np.std(np.abs(latent_x)) + 1e-8)
        latent_y_norm = (np.abs(latent_y) - np.mean(np.abs(latent_y))) / (np.std(np.abs(latent_y)) + 1e-8)

        # Per-row missingness probabilities
        px_row = mx_base + mar_strength * 0.3 * latent_x_norm  # (N,)
        py_row = my_base + mar_strength * 0.3 * latent_y_norm  # (N,)

        px_row = np.clip(px_row, 0.01, 0.99)
        py_row = np.clip(py_row, 0.01, 0.99)

        # Generate masks with row-varying probabilities
        Sx = np.zeros((params.N, params.Dx), dtype=int)
        Sy = np.zeros((params.N, params.Dy), dtype=int)
        for i in range(params.N):
            Sx[i] = np.random.binomial(1, 1 - px_row[i], size=params.Dx)
            Sy[i] = np.random.binomial(1, 1 - py_row[i], size=params.Dy)

    elif mar_type == 'magnitude_dependent':
        # Missingness probability depends on entry magnitude |X_ij|
        # Using sigmoid: P(miss) = mx_base + mar_strength * sigmoid(scale * (|X| - median))
        scale = 2.0  # Controls steepness

        # Normalize magnitudes
        X_mag = np.abs(X_star)
        Y_mag = np.abs(Y_star)
        X_median = np.median(X_mag)
        Y_median = np.median(Y_mag)

        # Sigmoid transformation
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        px_entry = mx_base + mar_strength * 0.4 * (sigmoid(scale * (X_mag - X_median)) - 0.5)
        py_entry = my_base + mar_strength * 0.4 * (sigmoid(scale * (Y_mag - Y_median)) - 0.5)

        px_entry = np.clip(px_entry, 0.01, 0.99)
        py_entry = np.clip(py_entry, 0.01, 0.99)

        # Generate masks with entry-varying probabilities
        Sx = (np.random.rand(params.N, params.Dx) > px_entry).astype(int)
        Sy = (np.random.rand(params.N, params.Dy) > py_entry).astype(int)

    elif mar_type == 'thresholded':
        # High probability of missing entries with large absolute values
        # P(miss | |X_ij| > tau) = mx_base + mar_strength * 0.6
        # P(miss | |X_ij| <= tau) = mx_base
        tau_x = np.percentile(np.abs(X_star), 75)  # Top 25% magnitude
        tau_y = np.percentile(np.abs(Y_star), 75)

        px_entry = np.where(np.abs(X_star) > tau_x,
                           mx_base + mar_strength * 0.5,
                           mx_base)
        py_entry = np.where(np.abs(Y_star) > tau_y,
                           my_base + mar_strength * 0.5,
                           my_base)

        px_entry = np.clip(px_entry, 0.01, 0.99)
        py_entry = np.clip(py_entry, 0.01, 0.99)

        Sx = (np.random.rand(params.N, params.Dx) > px_entry).astype(int)
        Sy = (np.random.rand(params.N, params.Dy) > py_entry).astype(int)

    elif mar_type == 'correlated':
        # Y missingness depends on X values (cross-view MAR)
        # P(Y_ij miss) depends on X_i @ u0 (latent X signal for that row)
        latent_x_norm = (latent_x - np.mean(latent_x)) / (np.std(latent_x) + 1e-8)

        # X uses standard MCAR
        Sx = np.random.binomial(1, 1 - mx_base, size=(params.N, params.Dx))

        # Y missingness depends on X latent
        py_row = my_base + mar_strength * 0.3 * np.abs(latent_x_norm)
        py_row = np.clip(py_row, 0.01, 0.99)

        Sy = np.zeros((params.N, params.Dy), dtype=int)
        for i in range(params.N):
            Sy[i] = np.random.binomial(1, 1 - py_row[i], size=params.Dy)

    else:
        raise ValueError(f"Unknown mar_type: {mar_type}. "
                        f"Use 'signal_dependent', 'magnitude_dependent', 'thresholded', or 'correlated'.")

    # Apply masks (missing-as-zero)
    X = Sx * X_star
    Y = Sy * Y_star

    return X, Y, Sx, Sy
