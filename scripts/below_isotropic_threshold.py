"""Figure 7: finite-size recovery thresholds below the isotropic threshold.

For each design retention rho_x and sample size N, simulates R_x^2 and the top
singular value on a signal grid around the masked and isotropic thresholds,
plus a null sweep at theta = 0 for the detection statistic.

Saves results/below_isotropic_threshold.pkl.
Run from the repository root: python -m scripts.below_isotropic_threshold --workers 8
"""
import numpy as np

from src.analysis import threshold_grid
from src.data import ModelParams, masked_trials, planted_directions
from src.utils import parallel_map, save_results, workers_from_argv

PARAMS = dict(
    alpha_x=2.0, alpha_y=2.0, rho_y=1.0,
    rho_x_values=(0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    sample_sizes=(500, 1000, 2000),
    n_trials=100,   # signal trials per grid point
    n_null=500,     # null trials at theta = 0
    n_theta=44, dense_points_per_threshold=8,
    seed=20260811,
)


def trials(job):
    """R_x^2 and sigma_1 over n_trials seeded trials at one parameter set."""
    params, n_trials = job
    return masked_trials(ModelParams(**params), n_trials)


def run(p, workers=1):
    sweeps = {}
    n_trials, n_null = p["n_trials"], p["n_null"]
    for N in p["sample_sizes"]:
        Dx, Dy = int(round(N / p["alpha_x"])), int(round(N / p["alpha_y"]))
        u0, v0 = planted_directions(Dx, Dy, np.random.default_rng(p["seed"] + N))
        for rho_x in p["rho_x_values"]:
            grid = threshold_grid(p["alpha_x"], p["alpha_y"], rho_x, p["rho_y"],
                                  p["n_theta"], p["dense_points_per_threshold"])

            def params(theta):
                return {"N": N, "Dx": Dx, "Dy": Dy, "theta": theta,
                        "mx": 1 - rho_x, "my": 1 - p["rho_y"], "u0": u0, "v0": v0}

            # The signal grid, then the null sweep at theta = 0.
            jobs = [(params(float(theta)), n_trials) for theta in grid]
            jobs.append((params(0.0), n_null))
            values = parallel_map(trials, jobs, workers, desc=f"N={N} rho_x={rho_x}")
            *signal, (_, null_sigma1) = values
            sweeps[f"rx{int(round(rho_x * 100)):03d}_N{N}"] = {
                "theta": grid,
                "Rx2": np.stack([rx2 for rx2, _ in signal]),
                "sigma1": np.stack([sigma1 for _, sigma1 in signal]),
                "null_sigma1": null_sigma1,
                "n_trials": n_trials, "n_null": n_null,
                "rho_x": rho_x, "rho_y": p["rho_y"],
                "alpha_x": p["alpha_x"], "alpha_y": p["alpha_y"], "N": N,
            }
    return {"params": p, "threshold_sweeps": sweeps}


if __name__ == "__main__":
    save_results("below_isotropic_threshold", run(PARAMS, workers_from_argv()))
