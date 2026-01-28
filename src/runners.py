"""
Experiment runners and parallel workers.

This module contains:
- run_single_trial, run_multiple_trials: comprehensive trial runners
- run_single_trial_pls_only, run_multiple_trials_pls_only: optimized for figures
- Worker functions for multiprocessing
- Diagnostic functions (split-half stability, bootstrap variance)
"""

import numpy as np
from typing import Optional

from .core import ModelParams, theoretical_overlaps, inv_sqrtm_psd
from .methods import pls_svd, compute_overlaps, complete_case_analysis, mean_imputation_pls
from .data import generate_data, generate_data_non_gaussian


def run_single_trial(params: ModelParams, seed: Optional[int] = None) -> dict:
    """
    Run a single trial: generate data, estimate, compute overlaps.

    Args:
        params: Model parameters
        seed: Random seed

    Returns:
        Dictionary with results including empirical and theoretical overlaps
    """
    # Generate data
    X, Y, Sx, Sy = generate_data(params, seed=seed)

    # PLS-SVD with prewhitening (our method)
    u_hat, v_hat, sigma1 = pls_svd(X, Y, prewhiten=True)
    Rx2_pls, Ry2_pls = compute_overlaps(u_hat, v_hat, params.u0, params.v0)

    # Naive PLS (no prewhitening)
    u_hat_naive, v_hat_naive, sigma1_naive = pls_svd(X, Y, prewhiten=False)
    Rx2_naive, Ry2_naive = compute_overlaps(u_hat_naive, v_hat_naive, params.u0, params.v0)

    # Complete-case analysis
    u_hat_cc, v_hat_cc = complete_case_analysis(X, Y, Sx, Sy, prewhiten=True)
    Rx2_cc, Ry2_cc = compute_overlaps(u_hat_cc, v_hat_cc, params.u0, params.v0)

    # Mean imputation
    u_hat_mi, v_hat_mi = mean_imputation_pls(X, Y, Sx, Sy, prewhiten=True)
    Rx2_mi, Ry2_mi = compute_overlaps(u_hat_mi, v_hat_mi, params.u0, params.v0)

    # Theoretical predictions
    Rx2_theory, Ry2_theory = theoretical_overlaps(params)

    return {
        'Rx2_pls': Rx2_pls,
        'Ry2_pls': Ry2_pls,
        'Rx2_naive': Rx2_naive,
        'Ry2_naive': Ry2_naive,
        'Rx2_cc': Rx2_cc,
        'Ry2_cc': Ry2_cc,
        'Rx2_mi': Rx2_mi,
        'Ry2_mi': Ry2_mi,
        'Rx2_theory': Rx2_theory,
        'Ry2_theory': Ry2_theory,
        'sigma1': sigma1,
        'theta_crit': params.theta_crit,
        'is_supercritical': params.is_supercritical,
    }


def run_multiple_trials(
    params: ModelParams,
    n_trials: int = 20,
    verbose: bool = True
) -> dict:
    """
    Run multiple trials and aggregate results.

    Args:
        params: Model parameters
        n_trials: Number of independent trials
        verbose: Print progress

    Returns:
        Dictionary with mean and std of overlaps across trials
    """
    results = []
    for trial in range(n_trials):
        if verbose and (trial + 1) % 5 == 0:
            print(f"  Trial {trial + 1}/{n_trials}")
        result = run_single_trial(params, seed=trial)
        results.append(result)

    # Aggregate
    aggregated = {}
    for key in results[0].keys():
        values = [r[key] for r in results]
        if isinstance(values[0], (int, float, np.floating)):
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)

    # Add theoretical values (constant across trials)
    aggregated['Rx2_theory'] = results[0]['Rx2_theory']
    aggregated['Ry2_theory'] = results[0]['Ry2_theory']
    aggregated['theta_crit'] = results[0]['theta_crit']
    aggregated['is_supercritical'] = results[0]['is_supercritical']

    return aggregated


def run_single_trial_pls_only(params: ModelParams, seed: Optional[int] = None) -> dict:
    """
    Run a single trial with only PLS-SVD (no baseline methods).

    This is optimized for figure generation where we only need PLS-SVD results.
    Skips naive PLS, complete-case analysis, and mean imputation to save compute time.

    Args:
        params: Model parameters
        seed: Random seed

    Returns:
        Dictionary with PLS-SVD results and theoretical predictions
    """
    # Generate data
    X, Y, Sx, Sy = generate_data(params, seed=seed)

    # PLS-SVD with prewhitening (our method)
    u_hat, v_hat, sigma1 = pls_svd(X, Y, prewhiten=True)
    Rx2_pls, Ry2_pls = compute_overlaps(u_hat, v_hat, params.u0, params.v0)

    # Theoretical predictions
    Rx2_theory, Ry2_theory = theoretical_overlaps(params)

    return {
        'Rx2_pls': Rx2_pls,
        'Ry2_pls': Ry2_pls,
        'Rx2_theory': Rx2_theory,
        'Ry2_theory': Ry2_theory,
        'theta_crit': params.theta_crit,
    }


def run_multiple_trials_pls_only(params: ModelParams, n_trials: int = 20) -> dict:
    """
    Run multiple trials with only PLS-SVD (optimized for figure generation).

    Args:
        params: Model parameters
        n_trials: Number of independent trials

    Returns:
        Dictionary with mean and std of overlaps across trials
    """
    results = []
    for trial in range(n_trials):
        result = run_single_trial_pls_only(params, seed=trial)
        results.append(result)

    # Aggregate
    aggregated = {}
    for key in results[0].keys():
        values = [r[key] for r in results]
        if isinstance(values[0], (int, float, np.floating)):
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)

    # Add theoretical values (constant across trials)
    aggregated['Rx2_theory'] = results[0]['Rx2_theory']
    aggregated['Ry2_theory'] = results[0]['Ry2_theory']
    aggregated['theta_crit'] = results[0]['theta_crit']

    return aggregated


# Parallel processing worker functions (must be picklable)

def _run_experiment_worker(args):
    """Worker function for parallel experiment execution."""
    params_dict, n_trials = args
    params = ModelParams(**params_dict)
    return run_multiple_trials_pls_only(params, n_trials=n_trials)


def _run_grid_worker(args):
    """Worker function for grid experiments."""
    params_dict, n_trials, i, j = args
    params = ModelParams(**params_dict)
    result = run_multiple_trials_pls_only(params, n_trials=n_trials)
    return (i, j, result['Rx2_pls_mean'], result['Rx2_theory'])


def _run_n_worker(args):
    """Worker function for multiple N experiments."""
    params_dict, n_trials, N, theta, theta_crit = args
    params = ModelParams(**params_dict)
    result = run_multiple_trials_pls_only(params, n_trials=n_trials)
    result['N'] = N
    result['theta'] = theta
    result['theta_norm'] = theta / theta_crit
    return result


def run_single_trial_non_gaussian(
    params: ModelParams,
    noise_type: str,
    seed: Optional[int] = None
) -> dict:
    """
    Run a single trial with non-Gaussian noise.

    Args:
        params: Model parameters
        noise_type: Type of noise distribution
        seed: Random seed

    Returns:
        Dictionary with PLS-SVD results and theoretical predictions
    """
    X, Y, Sx, Sy = generate_data_non_gaussian(params, noise_type=noise_type, seed=seed)

    u_hat, v_hat, sigma1 = pls_svd(X, Y, prewhiten=True)
    Rx2_pls, Ry2_pls = compute_overlaps(u_hat, v_hat, params.u0, params.v0)

    # Theoretical predictions (Gaussian theory)
    Rx2_theory, Ry2_theory = theoretical_overlaps(params)

    return {
        'Rx2_pls': Rx2_pls,
        'Ry2_pls': Ry2_pls,
        'Rx2_theory': Rx2_theory,
        'Ry2_theory': Ry2_theory,
        'sigma1': sigma1,
        'theta_crit': params.theta_crit,
    }


def run_multiple_trials_non_gaussian(
    params: ModelParams,
    noise_type: str,
    n_trials: int = 20
) -> dict:
    """
    Run multiple trials with non-Gaussian noise.

    Args:
        params: Model parameters
        noise_type: Type of noise distribution
        n_trials: Number of trials

    Returns:
        Dictionary with mean and std of results
    """
    results = []
    for trial in range(n_trials):
        result = run_single_trial_non_gaussian(params, noise_type, seed=trial)
        results.append(result)

    aggregated = {}
    for key in results[0].keys():
        values = [r[key] for r in results]
        if isinstance(values[0], (int, float, np.floating)):
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)

    aggregated['Rx2_theory'] = results[0]['Rx2_theory']
    aggregated['Ry2_theory'] = results[0]['Ry2_theory']
    aggregated['theta_crit'] = results[0]['theta_crit']

    return aggregated


def _run_non_gaussian_worker(args):
    """Worker function for non-Gaussian noise experiments."""
    params_dict, noise_type, n_trials, theta, theta_crit = args
    params = ModelParams(**params_dict)
    result = run_multiple_trials_non_gaussian(params, noise_type, n_trials=n_trials)
    result['theta'] = theta
    result['theta_norm'] = theta / theta_crit
    result['noise_type'] = noise_type
    return result


# Diagnostic functions

def compute_sigma_ratio(
    X: np.ndarray,
    Y: np.ndarray,
    prewhiten: bool = True,
    eps: float = 1e-10
) -> tuple[float, float, float]:
    """
    Compute top singular values of cross-covariance matrix.

    Args:
        X: Design matrix (N x Dx)
        Y: Response matrix (N x Dy)
        prewhiten: If True, prewhiten X before computing cross-covariance
        eps: Regularization for numerical stability

    Returns:
        sigma1: Top singular value
        sigma2: Second singular value
        ratio: sigma1 / sigma2 (spike ratio)
    """
    N = X.shape[0]

    if prewhiten:
        S_xx = (X.T @ X) / N
        A = inv_sqrtm_psd(S_xx, eps=eps)
        X_w = X @ A
    else:
        X_w = X

    C = (X_w.T @ Y) / N
    U, S, Vt = np.linalg.svd(C, full_matrices=False)

    sigma1 = S[0]
    sigma2 = S[1] if len(S) > 1 else 0.0
    ratio = sigma1 / sigma2 if sigma2 > eps else np.inf

    return sigma1, sigma2, ratio


def split_half_stability(
    X: np.ndarray,
    Y: np.ndarray,
    Sx: Optional[np.ndarray] = None,
    Sy: Optional[np.ndarray] = None,
    n_splits: int = 50,
    prewhiten: bool = True,
    seed: int = 42,
    eps: float = 1e-10
) -> tuple[float, float, float, float]:
    """
    Compute split-half stability of PLS directions.

    Splits data into two halves, runs PLS on each, and measures
    agreement between estimated directions in the original coordinate space.

    Note: When prewhitening is used, each half has its own whitening transformation.
    The PLS vectors must be transformed back to the original coordinate space
    before comparison, otherwise they live in incompatible spaces.

    Args:
        X: Design matrix (N x Dx)
        Y: Response matrix (N x Dy)
        Sx: Mask for X (optional, for masked data)
        Sy: Mask for Y (optional, for masked data)
        n_splits: Number of random splits to average over
        prewhiten: If True, prewhiten X in each half
        seed: Random seed
        eps: Regularization for inverse square root

    Returns:
        stability_x_mean: Mean squared overlap of u directions across splits
        stability_x_std: Std of squared overlap of u directions
        stability_y_mean: Mean squared overlap of v directions across splits
        stability_y_std: Std of squared overlap of v directions
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]

    stabilities_x = []
    stabilities_y = []

    for _ in range(n_splits):
        # Random split
        perm = rng.permutation(N)
        half = N // 2
        idx1, idx2 = perm[:half], perm[half:2*half]

        # Split data
        X1, X2 = X[idx1], X[idx2]
        Y1, Y2 = Y[idx1], Y[idx2]

        if prewhiten:
            # Compute whitening matrices for each half
            S_xx_1 = (X1.T @ X1) / half
            A1 = inv_sqrtm_psd(S_xx_1, eps=eps)
            X1_w = X1 @ A1

            S_xx_2 = (X2.T @ X2) / half
            A2 = inv_sqrtm_psd(S_xx_2, eps=eps)
            X2_w = X2 @ A2

            # Compute cross-covariance and SVD
            C1 = (X1_w.T @ Y1) / half
            U1, _, Vt1 = np.linalg.svd(C1, full_matrices=False)
            u1_whitened, v1 = U1[:, 0], Vt1[0, :]

            C2 = (X2_w.T @ Y2) / half
            U2, _, Vt2 = np.linalg.svd(C2, full_matrices=False)
            u2_whitened, v2 = U2[:, 0], Vt2[0, :]

            # Transform u vectors back to original coordinate space
            u1 = A1 @ u1_whitened
            u1 = u1 / np.linalg.norm(u1)

            u2 = A2 @ u2_whitened
            u2 = u2 / np.linalg.norm(u2)
        else:
            # No prewhitening: vectors are already in same space
            u1, v1, _ = pls_svd(X1, Y1, prewhiten=False)
            u2, v2, _ = pls_svd(X2, Y2, prewhiten=False)

        # Compute squared overlaps (now in same coordinate space)
        stabilities_x.append((u1 @ u2) ** 2)
        stabilities_y.append((v1 @ v2) ** 2)

    return (
        np.mean(stabilities_x),
        np.std(stabilities_x),
        np.mean(stabilities_y),
        np.std(stabilities_y)
    )


def bootstrap_direction_variance(
    X: np.ndarray,
    Y: np.ndarray,
    n_bootstrap: int = 100,
    prewhiten: bool = True,
    seed: int = 42
) -> tuple[float, float]:
    """
    Compute bootstrap variance of PLS directions.

    Args:
        X: Design matrix (N x Dx)
        Y: Response matrix (N x Dy)
        n_bootstrap: Number of bootstrap samples
        prewhiten: If True, prewhiten X
        seed: Random seed

    Returns:
        var_u: Variance of u_hat across bootstrap samples (average over components)
        var_v: Variance of v_hat across bootstrap samples (average over components)
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]

    u_samples = []
    v_samples = []

    for _ in range(n_bootstrap):
        # Bootstrap resample
        idx = rng.choice(N, size=N, replace=True)
        X_boot = X[idx]
        Y_boot = Y[idx]

        # Run PLS
        u_hat, v_hat, _ = pls_svd(X_boot, Y_boot, prewhiten=prewhiten)

        # Align sign (arbitrary sign in SVD)
        if len(u_samples) > 0:
            if u_hat @ u_samples[0] < 0:
                u_hat = -u_hat
                v_hat = -v_hat

        u_samples.append(u_hat)
        v_samples.append(v_hat)

    u_samples = np.array(u_samples)
    v_samples = np.array(v_samples)

    # Compute variance (average over components)
    var_u = np.mean(np.var(u_samples, axis=0))
    var_v = np.mean(np.var(v_samples, axis=0))

    return var_u, var_v


def _run_diagnostics_worker(args):
    """Worker function for computing observables."""
    params_dict, n_trials, theta, theta_crit = args
    params = ModelParams(**params_dict)

    # Collect results for multiple trials
    sigma1_list = []
    sigma2_list = []
    ratio_list = []
    rx2_list = []
    ry2_list = []

    for trial in range(n_trials):
        X, Y, Sx, Sy = generate_data(params, seed=trial)

        # PLS-SVD
        u_hat, v_hat, _ = pls_svd(X, Y, prewhiten=True)
        rx2, ry2 = compute_overlaps(u_hat, v_hat, params.u0, params.v0)
        rx2_list.append(rx2)
        ry2_list.append(ry2)

        # Singular values
        s1, s2, ratio = compute_sigma_ratio(X, Y, prewhiten=True)
        sigma1_list.append(s1)
        sigma2_list.append(s2)
        ratio_list.append(ratio)

    # Compute split-half stability (uses multiple internal splits)
    X, Y, Sx, Sy = generate_data(params, seed=0)
    stab_x_mean, stab_x_std, stab_y_mean, stab_y_std = split_half_stability(
        X, Y, n_splits=50, prewhiten=True
    )

    return {
        'theta': theta,
        'theta_norm': theta / theta_crit,
        'N': params.N,
        'Rx2_mean': np.mean(rx2_list),
        'Rx2_std': np.std(rx2_list),
        'Ry2_mean': np.mean(ry2_list),
        'Ry2_std': np.std(ry2_list),
        'Rx2_theory': theoretical_overlaps(params)[0],
        'Ry2_theory': theoretical_overlaps(params)[1],
        'sigma1_mean': np.mean(sigma1_list),
        'sigma1_std': np.std(sigma1_list),
        'sigma2_mean': np.mean(sigma2_list),
        'ratio_mean': np.mean(ratio_list),
        'ratio_std': np.std(ratio_list),
        'stability_x_mean': stab_x_mean,
        'stability_x_std': stab_x_std,
        'stability_y_mean': stab_y_mean,
        'stability_y_std': stab_y_std,
    }
