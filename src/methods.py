"""
PLS-SVD estimation methods and baselines.

This module contains:
- pls_svd: PLS-SVD with optional pre-whitening
- compute_overlaps: squared overlap computation
- complete_case_analysis: baseline using only complete rows
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


def complete_case_analysis(
    X: np.ndarray,
    Y: np.ndarray,
    Sx: np.ndarray,
    Sy: np.ndarray,
    prewhiten: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete-case analysis: use only rows with no missing entries.

    Args:
        X, Y: Data matrices with missing entries as zeros
        Sx, Sy: Missingness masks
        prewhiten: If True, rewhiten X_complete before computing cross-cov

    Returns:
        u_hat: Estimated direction in X
        v_hat: Estimated direction in Y
    """
    # Find complete cases (no missing in either X or Y)
    complete_rows = (Sx.all(axis=1)) & (Sy.all(axis=1))

    if complete_rows.sum() < 2:
        # Not enough complete cases
        Dx, Dy = X.shape[1], Y.shape[1]
        return np.zeros(Dx), np.zeros(Dy)

    X_complete = X[complete_rows]
    Y_complete = Y[complete_rows]
    N_complete = X_complete.shape[0]

    # Optionally prewhiten X_complete
    if prewhiten:
        S_xx = (X_complete.T @ X_complete) / N_complete
        A = inv_sqrtm_psd(S_xx)
        X_complete = X_complete @ A

    # Cross-covariance SVD
    C = X_complete.T @ Y_complete / N_complete

    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    return U[:, 0], Vt[0, :]


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
