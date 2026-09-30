"""Figure 3: the retention phase diagram and finite-size convergence.

Saves results/retention_phase_diagram.pkl.
Run from the repository root: python -m scripts.retention_phase_diagram --workers 8
"""
import numpy as np

from src import theory
from src.data import ModelParams, masked_trials, planted_directions
from src.utils import parallel_map, save_results, workers_from_argv

PARAMS = dict(
    # (a) R_x^2 on a grid of signal strength theta and joint retention rho_x * rho_y,
    # with equal retention in both views
    N=2000, Dx=300, Dy=240,
    n_theta=90, theta_min=0.2, theta_max=2.0,
    n_rho=90, joint_retention_min=0.05, joint_retention_max=0.95,
    grid_trials=30,              # trials per cell within the band around theta_c
    grid_band_min_factor=0.6, grid_band_max_factor=1.8,
    grid_background_trials=6,    # trials per cell elsewhere
    supercritical_factor=1.1,    # cells with theta > 1.1 theta_c enter the MAE
    # Finer signal grid across the boundary on every fourth retention row
    boundary_stride=4, boundary_fraction=0.35, band_n_theta=20, band_trials=60,
    # (b) Finite-size curves
    finite_sizes=(100, 250, 500, 1000, 2000, 5000),
    finite_trials=(800, 600, 500, 300, 150, 25),
    finite_alpha_x=2.5, finite_alpha_y=2.5, finite_missing_x=0.2, finite_missing_y=0.2,
    seed=20260811,
)


def mean_rx2(job):
    """Mean measured R_x^2 over n_trials seeded trials at one parameter set."""
    params, n_trials = job
    return float(masked_trials(ModelParams(**params), n_trials)[0].mean())


def cell(job):
    """Mean measured and predicted R_x^2 at one parameter set."""
    params = ModelParams(**job[0])
    predicted, _ = theory.overlaps_at(params.alpha_x, params.alpha_y,
                                      params.rho_x, params.rho_y, params.theta)
    return mean_rx2(job), predicted


def phase_grid(p, workers):
    alpha_x, alpha_y = p["N"] / p["Dx"], p["N"] / p["Dy"]
    theta_grid = np.linspace(p["theta_min"], p["theta_max"], p["n_theta"])
    rho_grid = np.linspace(p["joint_retention_min"], p["joint_retention_max"], p["n_rho"])
    u0, v0 = planted_directions(p["Dx"], p["Dy"], np.random.default_rng(p["seed"]))
    jobs = []
    for rho in rho_grid:
        retention = np.sqrt(rho)
        threshold = theory.theta_c(alpha_x, alpha_y, retention, retention)
        for theta in theta_grid:
            near = p["grid_band_min_factor"] * threshold <= theta <= p["grid_band_max_factor"] * threshold
            jobs.append(({"N": p["N"], "Dx": p["Dx"], "Dy": p["Dy"], "theta": float(theta),
                          "mx": 1 - retention, "my": 1 - retention, "u0": u0, "v0": v0},
                         p["grid_trials"] if near else p["grid_background_trials"]))
    values = np.array(parallel_map(cell, jobs, workers, desc="phase grid"))
    empirical = values[:, 0].reshape(p["n_rho"], p["n_theta"])
    predicted = values[:, 1].reshape(p["n_rho"], p["n_theta"])
    thresholds = np.asarray([theory.theta_c(alpha_x, alpha_y, np.sqrt(rho), np.sqrt(rho))
                             for rho in rho_grid])
    supercritical = theta_grid[None, :] > p["supercritical_factor"] * thresholds[:, None]
    return {
        "Rx2_empirical": empirical, "theta_grid": theta_grid, "rho_grid": rho_grid,
        "theta_c_rows": thresholds,
        "correlation": float(np.corrcoef(empirical.ravel(), predicted.ravel())[0, 1]),
        "supercritical_mae": float(np.abs(empirical[supercritical] - predicted[supercritical]).mean()),
    }


def boundary(p, workers):
    alpha_x, alpha_y = p["N"] / p["Dx"], p["N"] / p["Dy"]
    rho = np.linspace(p["joint_retention_min"], p["joint_retention_max"],
                      p["n_rho"])[::p["boundary_stride"]]
    u0, v0 = planted_directions(p["Dx"], p["Dy"], np.random.default_rng(p["seed"]))
    jobs = []
    for joint in rho:
        retention = np.sqrt(joint)
        threshold = theory.theta_c(alpha_x, alpha_y, retention, retention)
        for theta in np.linspace((1 - p["boundary_fraction"]) * threshold,
                                 (1 + p["boundary_fraction"]) * threshold, p["band_n_theta"]):
            jobs.append(({"N": p["N"], "Dx": p["Dx"], "Dy": p["Dy"], "theta": float(theta),
                          "mx": 1 - retention, "my": 1 - retention, "u0": u0, "v0": v0},
                         p["band_trials"]))
    values = np.array(parallel_map(cell, jobs, workers, desc="boundary"))
    return {"Rx2_empirical": values[:, 0].reshape(len(rho), p["band_n_theta"]),
            "Rx2_theoretical": values[:, 1].reshape(len(rho), p["band_n_theta"])}


def finite_size(p, workers):
    alpha_x, alpha_y = p["finite_alpha_x"], p["finite_alpha_y"]
    mx, my = p["finite_missing_x"], p["finite_missing_y"]
    theta_c = float(theory.theta_c(alpha_x, alpha_y, 1 - mx, 1 - my))
    # theta / theta_c, denser across the transition
    factors = np.unique(np.round(np.concatenate([
        np.linspace(0.50, 0.85, 8), np.linspace(0.85, 1.25, 16), np.linspace(1.25, 2.00, 10),
    ]), 6))
    results = {}
    for N, n_trials in zip(p["finite_sizes"], p["finite_trials"]):
        Dx, Dy = int(N / alpha_x), int(N / alpha_y)
        u0, v0 = planted_directions(Dx, Dy, np.random.default_rng(p["seed"] + N))
        jobs = [({"N": N, "Dx": Dx, "Dy": Dy, "theta": float(factor * theta_c),
                  "mx": mx, "my": my, "u0": u0, "v0": v0}, n_trials)
                for factor in factors]
        measured = parallel_map(mean_rx2, jobs, workers, desc=f"N={N}")
        results[N] = [{"theta_norm": float(factor * theta_c) / theta_c, "Rx2_pls_mean": rx2}
                      for factor, rx2 in zip(factors, measured)]
    return {
        "results": results, "N_configs": list(p["finite_sizes"]),
        "theta_crit": theta_c, "alpha_x": alpha_x, "alpha_y": alpha_y, "mx": mx, "my": my,
    }


def run(p, workers=1):
    return {
        "params": p,
        "phase_diagram": phase_grid(p, workers),
        "phase_diagram_boundary": boundary(p, workers),
        "finite_size": finite_size(p, workers),
    }


if __name__ == "__main__":
    save_results("retention_phase_diagram", run(PARAMS, workers_from_argv()))
