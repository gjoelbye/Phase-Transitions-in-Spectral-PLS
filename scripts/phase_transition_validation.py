"""Figure 1: the phase transition, its rho_y scaling and threshold vs retention.

Saves results/phase_transition_validation.pkl.
Run from the repository root: python -m scripts.phase_transition_validation --workers 8
"""
import numpy as np

from src import theory
from src.analysis import threshold_grid
from src.data import ModelParams, generate_flat_orthogonal, masked_trials, planted_directions
from src.methods import compute_overlaps, pls_svd
from src.utils import parallel_map, save_results, workers_from_argv

PARAMS = dict(
    # (a) overlap curves on a signal grid around theta_c
    N=2048, Dx=1024, Dy=1024, rho_x=0.3, rho_y=0.6, n_trials=100, n_theta=25,
    # (b) R_x^2 against theta * sqrt(rho_y) for several response retentions
    collapse_N=2000, collapse_Dx=400, collapse_Dy=100, collapse_rho_x=0.7,
    collapse_rho_y_values=(1.0, 0.8, 0.6, 0.4, 0.2),
    collapse_n_theta=24, collapse_trials=60,
    # (c) threshold sweeps over the design retention
    threshold_sample_size=2000, threshold_alpha_x=2.0, threshold_alpha_y=2.0,
    threshold_rho_y=1.0,
    threshold_rho_x_values=(0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                            0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    threshold_n_theta=44, threshold_dense_points_per_threshold=8,
    threshold_trials=100,
    seed=20260811,
)


def transition_trial(job):
    """One design, mask and noise draw, swept over the whole signal grid."""
    design, trial, thetas, N, Dx, Dy, rho_x, rho_y, u0, v0 = job
    rng = np.random.default_rng(trial)
    if design == "hadamard":
        X_star = generate_flat_orthogonal(N, Dx, rng)
    else:
        Q, _ = np.linalg.qr(rng.normal(size=(N, Dx)))
        X_star = Q * np.sqrt(N)
    Sx = rng.binomial(1, rho_x, size=(N, Dx))
    Sy = rng.binomial(1, rho_y, size=(N, Dy))
    noise = rng.normal(size=(N, Dy))
    X = Sx * X_star
    score = X_star @ u0
    rx, ry = [], []
    for theta in thetas:
        Y = Sy * (theta * np.outer(score, v0) + noise)
        u_hat, v_hat, _ = pls_svd(X, Y)
        left, right = compute_overlaps(u_hat, v_hat, u0, v0)
        rx.append(left)
        ry.append(right)
    return {"design": design, "trial": trial, "Rx2": np.asarray(rx), "Ry2": np.asarray(ry)}


def left_overlaps(job):
    """Squared left overlaps over n_trials seeded trials at one parameter set."""
    params, n_trials = job
    return masked_trials(ModelParams(**params), n_trials)[0]


def transition(p, workers):
    """Panel (a): Hadamard and Haar designs on n_theta signal strengths plus theta_c^iso."""
    N, Dx, Dy, rho_x, rho_y = p["N"], p["Dx"], p["Dy"], p["rho_x"], p["rho_y"]
    alpha_x, alpha_y = N / Dx, N / Dy
    theta_c = float(theory.theta_c(alpha_x, alpha_y, rho_x, rho_y))
    theta_iso = float(theory.theta_crit_iso(alpha_x, alpha_y, rho_x * rho_y))
    theta = np.append(np.linspace(0.5 * theta_c, 2.5 * theta_c, p["n_theta"]), theta_iso)
    u0, v0 = planted_directions(Dx, Dy, np.random.default_rng(0))
    jobs = [(design, trial, theta, N, Dx, Dy, rho_x, rho_y, u0, v0)
            for design in ("hadamard", "qr") for trial in range(p["n_trials"])]
    rows = parallel_map(transition_trial, jobs, workers, desc="transition")
    return {
        "theta": theta, "theta_c": theta_c, "rows": rows,
        "parameters": {"alpha_x": alpha_x, "alpha_y": alpha_y, "rho_x": rho_x, "rho_y": rho_y},
    }


def collapse(p, workers):
    """Panel (b): the same scaled signal grid theta * sqrt(rho_y) for each rho_y."""
    N, Dx, Dy = p["collapse_N"], p["collapse_Dx"], p["collapse_Dy"]
    rho_x, rho_y_values = p["collapse_rho_x"], p["collapse_rho_y_values"]
    alpha_x, alpha_y = N / Dx, N / Dy
    scaled_theta_c = float(theory.theta_c(alpha_x, alpha_y, rho_x, 1.0))
    scaled_grid = np.linspace(0.5, 2.5, p["collapse_n_theta"]) * scaled_theta_c
    u0, v0 = planted_directions(Dx, Dy, np.random.default_rng(p["seed"]))
    curves = {}
    for rho_y in rho_y_values:
        jobs = [({"N": N, "Dx": Dx, "Dy": Dy, "theta": float(scaled / np.sqrt(rho_y)),
                  "mx": 1 - rho_x, "my": 1 - rho_y, "u0": u0, "v0": v0},
                 p["collapse_trials"])
                for scaled in scaled_grid]
        rx2 = parallel_map(left_overlaps, jobs, workers, desc=f"collapse rho_y={rho_y}")
        curves[rho_y] = {"Rx2": np.stack(rx2)}
    return {
        "curves": curves, "scaled_grid": scaled_grid, "scaled_theta_c": scaled_theta_c,
        "rho_y_values": rho_y_values, "alpha_x": alpha_x, "alpha_y": alpha_y, "rho_x": rho_x,
    }


def threshold_sweeps(p, workers):
    """Panel (c): R_x^2 on a signal grid around both thresholds, for each rho_x."""
    N = p["threshold_sample_size"]
    alpha_x, alpha_y, rho_y = p["threshold_alpha_x"], p["threshold_alpha_y"], p["threshold_rho_y"]
    Dx, Dy = int(round(N / alpha_x)), int(round(N / alpha_y))
    u0, v0 = planted_directions(Dx, Dy, np.random.default_rng(p["seed"] + N))
    output = {}
    for rho_x in p["threshold_rho_x_values"]:
        grid = threshold_grid(alpha_x, alpha_y, rho_x, rho_y, p["threshold_n_theta"],
                              p["threshold_dense_points_per_threshold"])
        jobs = [({"N": N, "Dx": Dx, "Dy": Dy, "theta": float(theta),
                  "mx": 1 - rho_x, "my": 1 - rho_y, "u0": u0, "v0": v0},
                 p["threshold_trials"])
                for theta in grid]
        rx2 = parallel_map(left_overlaps, jobs, workers, desc=f"threshold rho_x={rho_x}")
        output[f"rx{int(round(rho_x * 100)):03d}_N{N}"] = {
            "theta": grid, "Rx2": np.stack(rx2), "n_trials": p["threshold_trials"],
            "rho_x": rho_x, "rho_y": rho_y, "alpha_x": alpha_x, "alpha_y": alpha_y, "N": N,
        }
    return output


def run(p, workers=1):
    return {
        "params": p,
        "transition": transition(p, workers),
        "collapse": collapse(p, workers),
        "threshold_sweeps": threshold_sweeps(p, workers),
    }


if __name__ == "__main__":
    save_results("phase_transition_validation", run(PARAMS, workers_from_argv()))
