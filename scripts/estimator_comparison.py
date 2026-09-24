"""Figure 5: zero-filled PLS-SVD against imputation baselines.

Seven estimators are compared on the same data: zero-filled PLS-SVD, re-whitened
PLS-SVD, mean imputation, EM, rank-truncated SVD imputation, soft-impute and
the complete-data oracle. Imputation hyperparameters are chosen once per setting
on a separate held-out draw.

Saves results/estimator_comparison.pkl.
Run from the repository root: python -m scripts.estimator_comparison --workers 8
"""
import numpy as np

from src import theory
from src.data import ModelParams, generate_data, planted_directions
from src.methods import (em_pls, inv_sqrtm_psd, iterative_svd_pls, mean_imputation_pls,
                         pls_svd, select_lambda_holdout, select_rank_holdout, soft_impute_pls)
from src.utils import parallel_map, save_results, workers_from_argv

PARAMS = dict(
    n_trials=30,
    # (a) moderate missingness, signal sweep
    moderate_N=1000, moderate_Dx=200, moderate_Dy=150, moderate_missingness=0.3,
    moderate_n_theta=20, moderate_theta_min_factor=0.5, moderate_theta_max_factor=2.0,
    # (b) heavy missingness, signal sweep
    low_N=800, low_Dx=400, low_Dy=200, low_missingness=0.7,
    low_n_theta=24, low_theta_min_factor=0.5, low_theta_max_factor=3.5,
    # (c) missingness sweep at theta = 1.5 theta_c, dimensions as in (a)
    retention_missingness_values=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    retention_theta_factor=1.5,
    seed=20260811,
)

# Convergence tolerances and iteration caps for the imputation baselines
IMPUTE_BUDGET = {"tol": 1e-4, "max_iter": 50}
SVT_BUDGET = {"tol": 1e-4, "max_iter": 200}


def select_tuning(params, seed):
    """Imputation ranks and soft-impute thresholds chosen on held-out entries."""
    X, Y, Sx, Sy = generate_data(params, seed=seed)
    return {"rank_x": select_rank_holdout(X, Sx, seed=seed, **IMPUTE_BUDGET),
            "rank_y": select_rank_holdout(Y, Sy, seed=seed + 2, **IMPUTE_BUDGET),
            "lam_x": select_lambda_holdout(X, Sx, seed=seed, **SVT_BUDGET),
            "lam_y": select_lambda_holdout(Y, Sy, seed=seed + 1, **SVT_BUDGET)}


def one_trial(params, seed, tuning):
    """Squared left overlap of all seven estimators on one masked sample."""
    # The same draw as generate_data, keeping the complete data for the oracle
    np.random.seed(seed)
    X_star = np.random.randn(params.N, params.Dx)
    Q, _ = np.linalg.qr(X_star)
    X_star = Q * np.sqrt(params.N)
    Y_star = params.theta * np.outer(X_star @ params.u0, params.v0) \
        + np.random.randn(params.N, params.Dy)
    Sx = np.random.binomial(1, params.rho_x, size=(params.N, params.Dx))
    Sy = np.random.binomial(1, params.rho_y, size=(params.N, params.Dy))
    X = Sx * X_star
    Y = Sy * Y_star

    u_pls, _, _ = pls_svd(X, Y)
    # Re-whitened PLS-SVD, with its direction mapped back to X coordinates
    u_white, _, _ = pls_svd(X, Y, prewhiten=True)
    u_pw = inv_sqrtm_psd((X.T @ X) / X.shape[0]) @ u_white
    u_pw /= np.linalg.norm(u_pw)
    u_mi, _ = mean_imputation_pls(X, Y, Sx, Sy)
    u_em, _, _ = em_pls(X, Y, Sx, Sy)
    u_svd, _, _ = iterative_svd_pls(X, Y, Sx, Sy, tuning["rank_x"], tuning["rank_y"],
                                    **IMPUTE_BUDGET)
    u_si, _, _ = soft_impute_pls(X, Y, Sx, Sy, lam_x=tuning["lam_x"], lam_y=tuning["lam_y"],
                                 **SVT_BUDGET)
    u_or, _, _ = pls_svd(X_star, Y_star)   # complete-data reference
    estimates = {"pls": u_pls, "pls_prewhite": u_pw, "mean_imp": u_mi, "em_pls": u_em,
                 "iter_svd": u_svd, "soft_impute": u_si, "oracle": u_or}
    return {f"Rx2_{name}": float(u @ params.u0) ** 2 for name, u in estimates.items()}


def compare(job):
    """Mean and std of every estimator's R_x^2 over trials at one setting."""
    params, n_trials, label = job
    params = ModelParams(**params)
    # Seed n_trials follows the trial seeds, so tuning uses a draw of its own
    tuning = select_tuning(params, seed=n_trials)
    trials = [one_trial(params, seed=trial, tuning=tuning) for trial in range(n_trials)]
    result = {}
    for key in trials[0]:
        values = [trial[key] for trial in trials]
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values))
    result.update(label)
    return result


def signal_sweep(p, prefix, direction_seed, workers):
    """Panels (a) and (b): every estimator over a signal grid around theta_c."""
    N, Dx, Dy, m = (p[f"{prefix}_{key}"] for key in ("N", "Dx", "Dy", "missingness"))
    alpha_x, alpha_y, rho = N / Dx, N / Dy, 1 - m
    theta_c = float(theory.theta_c(alpha_x, alpha_y, rho, rho))
    theta_values = np.linspace(p[f"{prefix}_theta_min_factor"] * theta_c,
                               p[f"{prefix}_theta_max_factor"] * theta_c, p[f"{prefix}_n_theta"])
    u0, v0 = planted_directions(Dx, Dy, np.random.default_rng(direction_seed))
    jobs = [({"N": N, "Dx": Dx, "Dy": Dy, "theta": float(theta), "mx": m, "my": m,
              "u0": u0, "v0": v0}, p["n_trials"], {"theta": float(theta)})
            for theta in theta_values]
    rows = parallel_map(compare, jobs, workers, desc=prefix)
    return {
        "results": rows, "theta_values": theta_values, "theta_c": theta_c,
        "kappa": float(theory.kappa(alpha_x, rho)),
        "theory_Rx2": np.asarray(theory.overlap_curves(alpha_x, alpha_y, rho, rho,
                                                       theta_values)["Rx2"]),
        "N": N, "Dx": Dx, "Dy": Dy, "m": m, "n_trials": p["n_trials"],
    }


def retention_sweep(p, workers):
    """Panel (c): every estimator at theta = 1.5 theta_c(m) as missingness m grows."""
    N, Dx, Dy = p["moderate_N"], p["moderate_Dx"], p["moderate_Dy"]
    alpha_x, alpha_y = N / Dx, N / Dy
    m_values = p["retention_missingness_values"]
    factor = p["retention_theta_factor"]
    u0, v0 = planted_directions(Dx, Dy, np.random.default_rng(p["seed"]))
    theta_c_m = [theory.theta_c(alpha_x, alpha_y, 1 - m, 1 - m) for m in m_values]
    jobs = [({"N": N, "Dx": Dx, "Dy": Dy, "theta": float(factor * tc), "mx": m, "my": m,
              "u0": u0, "v0": v0}, p["n_trials"], {"m": m, "theta": float(factor * tc)})
            for m, tc in zip(m_values, theta_c_m)]
    rows = parallel_map(compare, jobs, workers, desc="retention")
    return {
        "results": rows, "m_values": m_values, "theta_c_m": theta_c_m,
        "theory_Rx2": np.asarray([theory.overlaps_at(alpha_x, alpha_y, 1 - m, 1 - m, factor * tc)[0]
                                  for m, tc in zip(m_values, theta_c_m)]),
        "kappa_m": np.asarray([theory.kappa(alpha_x, 1 - m) for m in m_values]),
        "n_trials": p["n_trials"],
    }


def run(p, workers=1):
    return {
        "params": p,
        "moderate_retention": signal_sweep(p, "moderate", p["seed"], workers),
        "low_retention": signal_sweep(p, "low", p["seed"] + 1, workers),
        "retention_sweep": retention_sweep(p, workers),
    }


if __name__ == "__main__":
    save_results("estimator_comparison", run(PARAMS, workers_from_argv()))
