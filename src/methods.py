"""
PLS-SVD estimation methods and baselines.

This module contains:
- pls_svd: PLS-SVD with optional pre-whitening
- compute_overlaps: squared overlap computation
- mean_imputation_pls: baseline with mean imputation
"""

import numpy as np
from typing import Tuple

from .core import inv_sqrtm_psd


def pls_svd(
    X: np.ndarray,
    Y: np.ndarray,
    prewhiten: bool = True,
    eps: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute PLS-SVD with optional pre-whitening of observed X.

    This follows the correct procedure:
    1. (Optional) Rewhiten X_obs to restore X_w^T X_w ~ N I after masking
    2. Compute cross-covariance C = (1/N) X_w^T Y_obs
    3. Take top singular vectors of C

    Args:
        X: Design matrix (N x Dx), possibly with missing entries as zeros
        Y: Response matrix (N x Dy), possibly with missing entries as zeros
        prewhiten: If True, rewhiten X using its empirical covariance (recommended)
        eps: Regularization for inverse square root

    Returns:
        u_hat: Estimated direction in X (Dx,)
        v_hat: Estimated direction in Y (Dy,)
        sigma1: Top singular value
    """
    N = X.shape[0]

    # Step 1: Rewhiten X to restore orthogonality lost by masking
    if prewhiten:
        # Compute empirical covariance of observed X
        S_xx = (X.T @ X) / N

        # Compute inverse square root
        A = inv_sqrtm_psd(S_xx, eps=eps)

        # Prewhiten: X_w such that (1/N) X_w^T X_w ~ I
        X_w = X @ A
    else:
        X_w = X

    # Step 2: Compute cross-covariance
    C = (X_w.T @ Y) / N

    # Step 3: SVD
    U, S, Vt = np.linalg.svd(C, full_matrices=False)

    u_hat = U[:, 0]
    v_hat = Vt[0, :]
    sigma1 = S[0]

    return u_hat, v_hat, sigma1


def compute_overlaps(
    u_hat: np.ndarray,
    v_hat: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray
) -> Tuple[float, float]:
    """
    Compute squared overlaps with ground truth.

    Args:
        u_hat: Estimated direction in X
        v_hat: Estimated direction in Y
        u0: True direction in X
        v0: True direction in Y

    Returns:
        Rx2: Squared overlap (u_hat^T u0)^2
        Ry2: Squared overlap (v_hat^T v0)^2
    """
    Rx2 = (u_hat @ u0)**2
    Ry2 = (v_hat @ v0)**2
    return Rx2, Ry2


def mean_imputation_pls(
    X: np.ndarray,
    Y: np.ndarray,
    Sx: np.ndarray,
    Sy: np.ndarray,
    prewhiten: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean imputation: replace missing entries with column means.

    Args:
        X, Y: Data matrices with missing entries as zeros
        Sx, Sy: Missingness masks
        prewhiten: If True, rewhiten X_imp before computing cross-cov

    Returns:
        u_hat: Estimated direction in X
        v_hat: Estimated direction in Y
    """
    # Impute X
    X_imp = X.copy()
    for j in range(X.shape[1]):
        mask = Sx[:, j].astype(bool)
        if mask.sum() > 0:
            col_mean = X[mask, j].mean()
            X_imp[~mask, j] = col_mean

    # Impute Y
    Y_imp = Y.copy()
    for j in range(Y.shape[1]):
        mask = Sy[:, j].astype(bool)
        if mask.sum() > 0:
            col_mean = Y[mask, j].mean()
            Y_imp[~mask, j] = col_mean

    # Optionally prewhiten
    N = X.shape[0]
    if prewhiten:
        S_xx = (X_imp.T @ X_imp) / N
        A = inv_sqrtm_psd(S_xx)
        X_imp = X_imp @ A

    # Cross-covariance SVD
    C = X_imp.T @ Y_imp / N

    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    return U[:, 0], Vt[0, :]


def em_pls(
    X: np.ndarray,
    Y: np.ndarray,
    Sx: np.ndarray,
    Sy: np.ndarray,
    n_iter: int = 50,
    tol: float = 1e-6,
    prewhiten: bool = True,
    eps: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    EM algorithm for probabilistic PLS with missing data.

    Iteratively:
    1. E-step: Impute missing values using current rank-1 estimate
    2. M-step: Update PLS directions from imputed data

    This is a simplified EM approach that imputes missing entries
    based on the current rank-1 signal estimate, then re-estimates
    directions. More sophisticated approaches would integrate over
    the posterior of missing values.

    Args:
        X: Design matrix with missing entries as zeros (N x Dx)
        Y: Response matrix with missing entries as zeros (N x Dy)
        Sx: Mask for X (1 = observed, 0 = missing)
        Sy: Mask for Y (1 = observed, 0 = missing)
        n_iter: Maximum number of EM iterations
        tol: Convergence tolerance for direction change
        prewhiten: If True, rewhiten X at each M-step
        eps: Regularization for inverse square root

    Returns:
        u_hat: Estimated direction in X (Dx,)
        v_hat: Estimated direction in Y (Dy,)
        sigma1: Final top singular value
    """
    N, Dx = X.shape
    _, Dy = Y.shape

    # Initialize with mean imputation
    X_imp = X.copy().astype(float)
    Y_imp = Y.copy().astype(float)

    # Mean impute X
    for j in range(Dx):
        mask = Sx[:, j].astype(bool)
        if mask.sum() > 0:
            col_mean = X[mask, j].mean()
            X_imp[~mask, j] = col_mean

    # Mean impute Y
    for j in range(Dy):
        mask = Sy[:, j].astype(bool)
        if mask.sum() > 0:
            col_mean = Y[mask, j].mean()
            Y_imp[~mask, j] = col_mean

    # Initial PLS estimate
    u_hat, v_hat, sigma1 = pls_svd(X_imp, Y_imp, prewhiten=prewhiten, eps=eps)

    for iteration in range(n_iter):
        u_prev = u_hat.copy()
        v_prev = v_hat.copy()

        # E-step: Impute missing values using current estimate
        # Model: Y ≈ sigma1 * (X @ u) @ v^T
        # For missing X_ij: estimate X_ij from Y using the model
        # For missing Y_ij: estimate Y_ij from X using the model

        # Estimate latent scores
        # z_i ≈ X_i @ u (latent score for sample i)
        # For observed entries, use them directly
        # For imputed, use previous iteration's estimate

        # Compute latent scores from current imputed X
        z_x = X_imp @ u_hat  # (N,)

        # E-step for Y: Y_ij = sigma1 * z_i * v_j + noise
        # Missing Y_ij: impute as sigma1 * z_i * v_j
        Y_signal = sigma1 * np.outer(z_x, v_hat)
        Y_imp_new = Y.copy().astype(float)
        Y_imp_new[~Sy.astype(bool)] = Y_signal[~Sy.astype(bool)]

        # E-step for X: More complex, need to estimate z from Y then X from z
        # Simplified: for missing X_ij, use column mean (keep previous imputation)
        # This is a simplification; full EM would integrate over z
        # For now, we only update Y imputation in E-step

        Y_imp = Y_imp_new

        # M-step: Update PLS directions from imputed data
        u_hat, v_hat, sigma1 = pls_svd(X_imp, Y_imp, prewhiten=prewhiten, eps=eps)

        # Check convergence
        u_change = 1 - np.abs(u_hat @ u_prev)
        v_change = 1 - np.abs(v_hat @ v_prev)

        if u_change < tol and v_change < tol:
            break

    return u_hat, v_hat, sigma1


def iterative_svd_pls(
    X: np.ndarray,
    Y: np.ndarray,
    Sx: np.ndarray,
    Sy: np.ndarray,
    rank: int = 5,
    n_iter: int = 20,
    prewhiten: bool = True,
    eps: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Low-rank SVD imputation followed by PLS-SVD.

    Uses iterative soft-thresholded SVD to impute missing entries,
    then runs PLS-SVD on the completed matrices.

    Args:
        X: Design matrix with missing entries as zeros (N x Dx)
        Y: Response matrix with missing entries as zeros (N x Dy)
        Sx: Mask for X (1 = observed, 0 = missing)
        Sy: Mask for Y (1 = observed, 0 = missing)
        rank: Rank for low-rank approximation during imputation
        n_iter: Number of SVD iterations for imputation
        prewhiten: If True, rewhiten X before computing cross-cov
        eps: Regularization for inverse square root

    Returns:
        u_hat: Estimated direction in X (Dx,)
        v_hat: Estimated direction in Y (Dy,)
        sigma1: Top singular value
    """
    def impute_low_rank(M: np.ndarray, S: np.ndarray, rank: int, n_iter: int) -> np.ndarray:
        """Impute missing entries using iterative low-rank SVD."""
        M_imp = M.copy().astype(float)

        # Initialize missing entries with column means
        for j in range(M.shape[1]):
            mask = S[:, j].astype(bool)
            if mask.sum() > 0:
                col_mean = M[mask, j].mean()
                M_imp[~mask, j] = col_mean
            else:
                M_imp[~mask, j] = 0.0

        for _ in range(n_iter):
            # Low-rank approximation
            U, s, Vt = np.linalg.svd(M_imp, full_matrices=False)
            # Keep top 'rank' components
            k = min(rank, len(s))
            M_low_rank = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

            # Update only missing entries
            M_imp = np.where(S.astype(bool), M, M_low_rank)

        return M_imp

    # Impute X and Y using low-rank SVD
    X_imp = impute_low_rank(X, Sx, rank, n_iter)
    Y_imp = impute_low_rank(Y, Sy, rank, n_iter)

    # Run PLS-SVD on imputed data
    u_hat, v_hat, sigma1 = pls_svd(X_imp, Y_imp, prewhiten=prewhiten, eps=eps)

    return u_hat, v_hat, sigma1


def oracle_pls(
    X_star: np.ndarray,
    Y_star: np.ndarray,
    prewhiten: bool = False,
    eps: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Oracle PLS: uses complete (unmasked) data.

    This serves as an upper bound on performance - the best we could
    do if there were no missing data.

    Args:
        X_star: Complete design matrix (N x Dx)
        Y_star: Complete response matrix (N x Dy)
        prewhiten: If True, rewhiten X (usually False for oracle since X_star is already whitened)
        eps: Regularization for inverse square root

    Returns:
        u_hat: Estimated direction in X (Dx,)
        v_hat: Estimated direction in Y (Dy,)
        sigma1: Top singular value
    """
    return pls_svd(X_star, Y_star, prewhiten=prewhiten, eps=eps)
