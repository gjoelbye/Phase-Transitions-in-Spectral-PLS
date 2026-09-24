"""Figure 2: angle between the estimated and planted left directions as rho_x falls.

Saves results/direction_geometry.pkl.
Run from the repository root: python -m scripts.direction_geometry --workers 8
"""
import numpy as np

from src.data import generate_flat_orthogonal, planted_directions
from src.methods import pls_svd
from src.utils import parallel_map, save_results, workers_from_argv

PARAMS = dict(
    N=1024, Dx=512, Dy=256, rho_y=1.0,
    theta_levels=(1.0, 2.0, 8.0),
    rho_x_values=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.12, 0.07),
    n_masks=5,     # design masks per (theta, rho_x)
    n_noise=12,    # noise and response-mask draws per design mask
    seed=20260814,
)


def level(job):
    """R_x^2 over n_masks x n_noise draws at one (rho_x, theta)."""
    rho_x, theta, seed, X_star, u0, v0, rho_y, n_masks, n_noise = job
    N, Dx = X_star.shape
    Dy = v0.size
    rng = np.random.default_rng(seed)
    rx = np.empty((n_masks, n_noise))
    score = X_star @ u0
    for mask_index in range(n_masks):
        X = X_star * rng.binomial(1, rho_x, size=(N, Dx))
        for noise_index in range(n_noise):
            Y_star = theta * np.outer(score, v0) + rng.normal(size=(N, Dy))
            Sy = rng.binomial(1, rho_y, size=(N, Dy))
            u_hat, _, _ = pls_svd(X, Sy * Y_star)
            rx[mask_index, noise_index] = float(u_hat @ u0) ** 2
    return {"rho_x": float(rho_x), "theta": float(theta),
            "Rx2": rx.ravel(), "Rx2_by_mask": rx.mean(axis=1)}


def run(p, workers=1):
    # One randomized Walsh-Hadamard design and one planted pair shared by every level
    rng = np.random.default_rng(p["seed"])
    X_star = generate_flat_orthogonal(p["N"], p["Dx"], rng)
    u0, v0 = planted_directions(p["Dx"], p["Dy"], rng)
    jobs = [(rho_x, theta, p["seed"] + 200 + 50 * i + j, X_star, u0, v0,
             p["rho_y"], p["n_masks"], p["n_noise"])
            for i, theta in enumerate(p["theta_levels"])
            for j, rho_x in enumerate(p["rho_x_values"])]
    return {"params": p, "sweep": parallel_map(level, jobs, workers, desc="geometry")}


if __name__ == "__main__":
    save_results("direction_geometry", run(PARAMS, workers_from_argv()))
