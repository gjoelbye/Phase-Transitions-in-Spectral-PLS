"""Figure 6: PLS-SVD under correlated response noise, before and after whitening.

The noise has covariance Sigma. Each trial runs PLS-SVD on the raw response
("naive") and on the response whitened by Sigma^{-1/2}. Whitening changes the
effective signal to theta * ||Sigma^{-1/2} v0||, which sets the whitened prediction.

Saves results/correlated_noise.pkl.
Run from the repository root: python -m scripts.correlated_noise --workers 8
"""
import numpy as np
from scipy.linalg import toeplitz

from src import theory
from src.methods import inv_sqrtm_psd, pls_svd
from src.utils import parallel_map, save_results, workers_from_argv

PARAMS = dict(
    N=1000, Dx=200, Dy=150, rho_x=0.7, rho_y=0.7,
    # Toeplitz covariance r^|i-j| on a grid of signal strengths and correlations r
    n_trials=400, n_theta=30, theta_min_factor=0.5, theta_max_factor=4.0,
    n_correlations=20, min_correlation=0.0, max_correlation=0.95,
    seed=42,
    # Diagnostic curves: selected Toeplitz correlations and rank-5 factor covariances
    diagnostic_trials=400, diagnostic_n_theta=30,
    diagnostic_theta_min_factor=0.5, diagnostic_theta_max_factor=4.0,
    diagnostic_correlations=(0.0, 0.3, 0.6, 0.9),
    factor_variances=(0.0, 1.0, 3.0, 10.0), factor_rank=5, factor_seed=123,
    diagnostic_seed=0,
)


def sqrt_psd(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * np.sqrt(np.maximum(values, 1e-10))) @ vectors.T


def trial(job):
    """Naive and whitened R_x^2, and the whitened effective signal, for one draw."""
    N, Dx, Dy, theta, rho_x, rho_y, sqrt_cov, inv_sqrt_cov, seed = job
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(N, Dx)), mode="reduced")
    X_star = Q * np.sqrt(N)
    u0 = rng.normal(size=Dx)
    v0 = rng.normal(size=Dy)
    u0 /= np.linalg.norm(u0)
    v0 /= np.linalg.norm(v0)
    noise = rng.normal(size=(N, Dy))
    Y_correlated = theta * np.outer(X_star @ u0, v0) + noise @ sqrt_cov
    Y_whitened = Y_correlated @ inv_sqrt_cov
    adjusted = theta * np.linalg.norm(inv_sqrt_cov @ v0)
    Sx = rng.binomial(1, rho_x, size=(N, Dx))
    Sy = rng.binomial(1, rho_y, size=(N, Dy))
    X = Sx * X_star
    u_naive, _, _ = pls_svd(X, Sy * Y_correlated)
    u_white, _, _ = pls_svd(X, Sy * Y_whitened)
    return float(u_naive @ u0) ** 2, float(u_white @ u0) ** 2, float(adjusted)


def sweep(p, covariance, theta_values, n_trials, seeds, workers, desc):
    """Trial means over a signal grid for one noise covariance."""
    alpha_x, alpha_y = p["N"] / p["Dx"], p["N"] / p["Dy"]
    sqrt_cov, inv_sqrt_cov = sqrt_psd(covariance), inv_sqrtm_psd(covariance)
    jobs = [(p["N"], p["Dx"], p["Dy"], float(theta), p["rho_x"], p["rho_y"],
             sqrt_cov, inv_sqrt_cov, seed)
            for theta, seed in zip(np.repeat(theta_values, n_trials), seeds)]
    values = np.asarray(parallel_map(trial, jobs, workers, desc=desc))
    values = values.reshape(len(theta_values), n_trials, 3)
    predicted_white = np.asarray([
        [theory.overlaps_at(alpha_x, alpha_y, p["rho_x"], p["rho_y"], value)[0] for value in row]
        for row in values[:, :, 2]
    ])
    return {
        "Rx2_naive_mean": values[:, :, 0].mean(axis=1),
        "Rx2_whitened_mean": values[:, :, 1].mean(axis=1),
        "theory_whitened_mean": predicted_white.mean(axis=1),
        "cond": float(np.linalg.cond(covariance)),
    }


def run(p, workers=1):
    alpha_x, alpha_y = p["N"] / p["Dx"], p["N"] / p["Dy"]
    theta_c = float(theory.theta_c(alpha_x, alpha_y, p["rho_x"], p["rho_y"]))

    # Toeplitz field: every correlation r on the signal grid, fresh seeds per cell
    theta_values = np.linspace(p["theta_min_factor"] * theta_c,
                               p["theta_max_factor"] * theta_c, p["n_theta"])
    r_values = np.linspace(p["min_correlation"], p["max_correlation"], p["n_correlations"])
    sweeps = {}
    for r_index, r in enumerate(r_values):
        seeds = [p["seed"] + r_index * 10_000_000 + j * 10_000 + t
                 for j in range(len(theta_values)) for t in range(p["n_trials"])]
        sweeps[float(r)] = sweep(p, toeplitz(r ** np.arange(p["Dy"])), theta_values,
                                 p["n_trials"], seeds, workers, desc=f"r={r:.2f}")
    toeplitz_field = {
        "sweeps": sweeps, "r_values": r_values, "theta_values": theta_values,
        "theta_c": theta_c,
        "theory_naive": np.asarray(theory.overlap_curves(
            alpha_x, alpha_y, p["rho_x"], p["rho_y"], theta_values)["Rx2"]),
    }

    # Diagnostics: all covariances share the same seeds
    theta_d = np.linspace(p["diagnostic_theta_min_factor"] * theta_c,
                          p["diagnostic_theta_max_factor"] * theta_c, p["diagnostic_n_theta"])
    seeds = [p["diagnostic_seed"] + t * 10_000 + j
             for j in range(p["diagnostic_n_theta"]) for t in range(p["diagnostic_trials"])]
    toeplitz_curves = {
        r: sweep(p, toeplitz(r ** np.arange(p["Dy"])), theta_d, p["diagnostic_trials"],
                 seeds, workers, desc=f"diagnostic r={r}")
        for r in p["diagnostic_correlations"]
    }
    factors = np.random.default_rng(p["factor_seed"]).normal(size=(p["Dy"], p["factor_rank"]))
    factor_curves = {
        variance: sweep(p, np.eye(p["Dy"]) + variance * (factors @ factors.T), theta_d,
                        p["diagnostic_trials"], seeds, workers, desc=f"factor {variance}")
        for variance in p["factor_variances"]
    }
    return {
        "params": p,
        "toeplitz_field": toeplitz_field,
        "covariance_diagnostics": {
            "toeplitz": toeplitz_curves, "factor": factor_curves,
            "theta_values": theta_d, "theta_c": theta_c,
            "theory_naive": np.asarray(theory.overlap_curves(
                alpha_x, alpha_y, p["rho_x"], p["rho_y"], theta_d)["Rx2"]),
        },
    }


if __name__ == "__main__":
    save_results("correlated_noise", run(PARAMS, workers_from_argv()))
