"""Figure 4: overlap curves under non-Gaussian and heteroskedastic noise.

Saves results/noise_distribution_robustness.pkl.
Run from the repository root: python -m scripts.noise_distribution_robustness --workers 8
"""
import numpy as np

from src import theory
from src.data import ModelParams, generate_data_non_gaussian, planted_directions
from src.methods import compute_overlaps, pls_svd
from src.utils import parallel_map, save_results, workers_from_argv

PARAMS = dict(
    N=1000, Dx=200, Dy=150, rho_x=0.7, rho_y=0.7,
    n_trials=100, n_theta=25, theta_min_factor=0.6, theta_max_factor=2.0,
    supercritical_factor=1.1,   # theta > 1.1 theta_c enters the MAE
    noise_types=("gaussian", "t5", "t4.5", "t3", "laplace", "heteroskedastic"),
    seed=20260811,
)


def noise_trials(job):
    """Mean and spread of the squared overlaps at one signal strength and noise law."""
    params, noise_type, n_trials, theta = job
    params = ModelParams(**params)
    rx2 = np.empty(n_trials)
    ry2 = np.empty(n_trials)
    for trial in range(n_trials):
        X, Y, _, _ = generate_data_non_gaussian(params, noise_type=noise_type, seed=trial)
        u_hat, v_hat, _ = pls_svd(X, Y)
        rx2[trial], ry2[trial] = compute_overlaps(u_hat, v_hat, params.u0, params.v0)
    return {
        "Rx2_pls_mean": float(rx2.mean()), "Rx2_pls_std": float(rx2.std()),
        "Ry2_pls_mean": float(ry2.mean()), "theta": theta, "noise_type": noise_type,
    }


def run(p, workers=1):
    alpha_x, alpha_y = p["N"] / p["Dx"], p["N"] / p["Dy"]
    theta_c = float(theory.theta_c(alpha_x, alpha_y, p["rho_x"], p["rho_y"]))
    theta_values = np.linspace(p["theta_min_factor"] * theta_c,
                               p["theta_max_factor"] * theta_c, p["n_theta"])
    u0, v0 = planted_directions(p["Dx"], p["Dy"], np.random.default_rng(p["seed"]))
    noise = {}
    for noise_type in p["noise_types"]:
        jobs = [({"N": p["N"], "Dx": p["Dx"], "Dy": p["Dy"], "theta": float(theta),
                  "mx": 1 - p["rho_x"], "my": 1 - p["rho_y"], "u0": u0, "v0": v0},
                 noise_type, p["n_trials"], float(theta))
                for theta in theta_values]
        noise[noise_type] = parallel_map(noise_trials, jobs, workers, desc=noise_type)

    curves = theory.overlap_curves(alpha_x, alpha_y, p["rho_x"], p["rho_y"], theta_values)
    theory_Rx2, theory_Ry2 = np.asarray(curves["Rx2"]), np.asarray(curves["Ry2"])
    # Mean absolute error against the prediction, supercritical signals only
    supercritical = theta_values / theta_c > p["supercritical_factor"]
    mae = {name: float(np.mean(np.abs(
               np.asarray([row["Rx2_pls_mean"] for row in rows])[supercritical]
               - theory_Rx2[supercritical])))
           for name, rows in noise.items()}
    return {
        "params": p, "noise": noise, "theta_values": theta_values, "theta_c": theta_c,
        "theory_Rx2": theory_Rx2, "theory_Ry2": theory_Ry2,
        "mae_supercritical": mae,
    }


if __name__ == "__main__":
    save_results("noise_distribution_robustness", run(PARAMS, workers_from_argv()))
