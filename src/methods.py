"""PLS-SVD and the estimator baselines used in the paper."""

from typing import Tuple

import numpy as np


def inv_sqrtm_psd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Inverse square root of a symmetric PSD matrix, eigenvalues floored at eps."""
    A_sym = (A + A.T) / 2
    w, V = np.linalg.eigh(A_sym)
    w = np.maximum(w, eps)
    return (V / np.sqrt(w)) @ V.T


def pls_svd(
    X: np.ndarray,
    Y: np.ndarray,
    prewhiten: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return the leading singular pair of ``X.T @ Y / N``.

    ``prewhiten=False`` is the estimator of Eq. (3). The optional re-whitening is
    a comparison estimator and returns its left direction in whitened coordinates.
    """
    N = X.shape[0]
    if prewhiten:
        S_xx = (X.T @ X) / N
        A = inv_sqrtm_psd(S_xx)
        X_w = X @ A
    else:
        X_w = X
    C = (X_w.T @ Y) / N
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    return U[:, 0], Vt[0, :], S[0]


def compute_overlaps(
    u_hat: np.ndarray,
    v_hat: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray
) -> Tuple[float, float]:
    """Return squared overlaps with the two true directions."""
    Rx2 = (u_hat @ u0)**2
    Ry2 = (v_hat @ v0)**2
    return Rx2, Ry2


def _mean_impute(M: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Fill each column's missing entries with its observed mean, then center."""
    M_imp = M.astype(float)
    for j in range(M.shape[1]):
        observed = S[:, j].astype(bool)
        M_imp[~observed, j] = M[observed, j].mean()
    return M_imp - M_imp.mean(axis=0, keepdims=True)


def mean_imputation_pls(
    X: np.ndarray,
    Y: np.ndarray,
    Sx: np.ndarray,
    Sy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Column-mean imputation followed by PLS-SVD."""
    X_imp = _mean_impute(X, Sx)
    Y_imp = _mean_impute(Y, Sy)

    N = X.shape[0]
    C = X_imp.T @ Y_imp / N

    U, _, Vt = np.linalg.svd(C, full_matrices=False)
    return U[:, 0], Vt[0, :]


def em_pls(
    X: np.ndarray,
    Y: np.ndarray,
    Sx: np.ndarray,
    Sy: np.ndarray,
    n_iter: int = 50,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Iterate rank-one response imputation and PLS-SVD updates.

    Starts from mean imputation. Each step refills the missing response
    entries from the current rank-one fit and recomputes PLS-SVD.
    """
    X_imp = _mean_impute(X, Sx)
    Y_imp = _mean_impute(Y, Sy)
    u_hat, v_hat, sigma1 = pls_svd(X_imp, Y_imp)

    for _ in range(n_iter):
        u_prev = u_hat.copy()
        v_prev = v_hat.copy()
        z_x = X_imp @ u_hat
        Y_signal = sigma1 * np.outer(z_x, v_hat)
        Y_imp_new = Y.astype(float)
        Y_imp_new[~Sy.astype(bool)] = Y_signal[~Sy.astype(bool)]
        Y_imp = Y_imp_new - Y_imp_new.mean(axis=0, keepdims=True)
        u_hat, v_hat, sigma1 = pls_svd(X_imp, Y_imp)
        u_change = 1 - np.abs(u_hat @ u_prev)
        v_change = 1 - np.abs(v_hat @ v_prev)

        if u_change < tol and v_change < tol:
            break

    return u_hat, v_hat, sigma1


def impute_low_rank(
    M: np.ndarray,
    S: np.ndarray,
    rank: int,
    tol: float = 1e-5,
    max_iter: int = 500,
) -> np.ndarray:
    """Fill missing entries by rank-truncated SVD until convergence."""
    M = np.asarray(M, dtype=float)
    obs = S.astype(bool)
    M_imp = M.copy()
    for j in range(M.shape[1]):
        col = obs[:, j]
        M_imp[~col, j] = M[col, j].mean()

    for _ in range(max_iter):
        U, sv, Vt = np.linalg.svd(M_imp, full_matrices=False)
        low = U[:, :rank] @ np.diag(sv[:rank]) @ Vt[:rank, :]
        M_new = np.where(obs, M, low)
        change = np.linalg.norm(M_new - M_imp) / (np.linalg.norm(M_imp) + 1e-12)
        M_imp = M_new
        if change < tol:
            break
    return M_imp


def soft_impute(
    M: np.ndarray,
    S: np.ndarray,
    lam: float,
    tol: float = 1e-5,
    max_iter: int = 500,
    init: np.ndarray = None,
) -> np.ndarray:
    """Fill missing entries by iterative singular-value thresholding.

    ``init`` provides a warm start for a decreasing sequence of thresholds.
    """
    M = np.asarray(M, dtype=float)
    obs = S.astype(bool)
    Z = np.zeros_like(M) if init is None else np.array(init, dtype=float)
    for _ in range(max_iter):
        U, sv, Vt = np.linalg.svd(np.where(obs, M, Z), full_matrices=False)
        sv_shrunk = np.maximum(sv - lam, 0.0)
        k = int((sv_shrunk > 0).sum())
        Z_new = (U[:, :k] * sv_shrunk[:k]) @ Vt[:k, :] if k else np.zeros_like(M)
        change = np.linalg.norm(Z_new - Z) / (np.linalg.norm(Z) + 1e-12)
        Z = Z_new
        if change < tol:
            break
    return np.where(obs, M, Z)


def _holdout_split(S: np.ndarray, holdout_frac: float, seed: int):
    """Hide a fraction of the observed entries; return (train mask, held-out index)."""
    obs = S.astype(bool)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(obs.ravel())
    n_hold = int(round(holdout_frac * idx.size))
    hold = rng.choice(idx, size=n_hold, replace=False)
    train = obs.ravel().copy()
    train[hold] = False
    return train.reshape(S.shape), hold


def select_rank_holdout(
    M: np.ndarray,
    S: np.ndarray,
    ranks=(1, 2, 5, 10, 20),
    holdout_frac: float = 0.1,
    seed: int = 0,
    tol: float = 1e-5,
    max_iter: int = 500,
):
    """Choose the truncation rank by prediction on held-out observed entries.

    A fraction of the observed entries is hidden, the rest drive the imputation,
    and the rank minimising squared error on the hidden entries is returned.
    """
    M = np.asarray(M, dtype=float)
    train_mask, hold = _holdout_split(S, holdout_frac, seed)
    truth = M.ravel()[hold]
    errors = {}
    for r in ranks:
        filled = impute_low_rank(M * train_mask, train_mask, int(r),
                                 tol=tol, max_iter=max_iter)
        errors[int(r)] = float(np.mean((filled.ravel()[hold] - truth) ** 2))
    return min(errors, key=errors.get)


def select_lambda_holdout(
    M: np.ndarray,
    S: np.ndarray,
    holdout_frac: float = 0.1,
    seed: int = 0,
    tol: float = 1e-5,
    max_iter: int = 500,
) -> float:
    """Select a soft-impute threshold on held-out observed entries.

    The grid includes ``lam_zero``, the operator norm of the zero-filled
    training matrix, and warm-starts each lower threshold from the previous fit.
    """
    M = np.asarray(M, dtype=float)
    train_mask, hold = _holdout_split(S, holdout_frac, seed)
    truth = M.ravel()[hold]
    M_train = M * train_mask
    lam_zero = float(np.linalg.svd(M_train, compute_uv=False)[0])
    grid = ([lam_zero]
            + [lam_zero * f for f in (0.99, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80)]
            + list(lam_zero * np.geomspace(0.70, 0.02, 14)))
    errors, Z = {}, None
    for lam in grid:
        filled = soft_impute(M_train, train_mask, float(lam), tol=tol,
                             max_iter=max_iter, init=Z)
        Z = filled
        errors[float(lam)] = float(np.mean((filled.ravel()[hold] - truth) ** 2))
    return min(errors, key=errors.get)


def soft_impute_pls(
    X: np.ndarray,
    Y: np.ndarray,
    Sx: np.ndarray,
    Sy: np.ndarray,
    lam_x: float,
    lam_y: float,
    tol: float = 1e-5,
    max_iter: int = 500,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Nuclear-norm imputation of both views followed by PLS-SVD."""
    X_imp = soft_impute(X, Sx, lam_x, tol=tol, max_iter=max_iter)
    Y_imp = soft_impute(Y, Sy, lam_y, tol=tol, max_iter=max_iter)
    return pls_svd(X_imp, Y_imp)


def iterative_svd_pls(
    X: np.ndarray,
    Y: np.ndarray,
    Sx: np.ndarray,
    Sy: np.ndarray,
    rank_x: int,
    rank_y: int,
    tol: float = 1e-5,
    max_iter: int = 500,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Rank-truncated SVD imputation of both views, then PLS-SVD."""
    X_imp = impute_low_rank(X, Sx, rank_x, tol=tol, max_iter=max_iter)
    Y_imp = impute_low_rank(Y, Sy, rank_y, tol=tol, max_iter=max_iter)
    return pls_svd(X_imp, Y_imp)
