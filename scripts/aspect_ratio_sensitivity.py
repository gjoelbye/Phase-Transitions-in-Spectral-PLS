"""Figure 9: overlap curves across five aspect-ratio regimes.

Saves results/aspect_ratio_sensitivity.pkl.
Run from the repository root: python -m scripts.aspect_ratio_sensitivity --workers 8
"""
import numpy as np

from src import theory
from src.methods import pls_svd
from src.utils import parallel_map, save_results, workers_from_argv

PARAMS = dict(
    geometries=(          # (name, N, Dx, Dy)
        ("Thin-X", 1000, 20, 150),
        ("Moderate", 1000, 200, 150),
        ("Wide-X", 1000, 500, 150),
        ("Thin-Y", 1000, 200, 20),
        ("Fat-Y", 1000, 200, 500),
    ),
    rho_x=0.7, rho_y=0.7,
    n_trials=100, n_theta=25, theta_min_factor=0.5, theta_max_factor=2.5,
    seed=42,
)


def trial(job):
    """R_x^2 for one draw of design, directions, noise and masks."""
    N, Dx, Dy, theta, rho_x, rho_y, seed = job
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((N, Dx)), mode="reduced")
    X_star = Q * np.sqrt(N)
    u0 = rng.normal(size=Dx)
    v0 = rng.normal(size=Dy)
    u0 /= np.linalg.norm(u0)
    v0 /= np.linalg.norm(v0)
    Y_star = theta * np.outer(X_star @ u0, v0) + rng.normal(size=(N, Dy))
    X = rng.binomial(1, rho_x, size=(N, Dx)) * X_star
    Y = rng.binomial(1, rho_y, size=(N, Dy)) * Y_star
    u_hat, _, _ = pls_svd(X, Y)
    return float(u_hat @ u0) ** 2


def run(p, workers=1):
    rho_x, rho_y = p["rho_x"], p["rho_y"]
    rows = []
    for g, (name, N, Dx, Dy) in enumerate(p["geometries"]):
        alpha_x, alpha_y = N / Dx, N / Dy
        theta_c = float(theory.theta_c(alpha_x, alpha_y, rho_x, rho_y))
        theta_values = np.linspace(p["theta_min_factor"] * theta_c,
                                   p["theta_max_factor"] * theta_c, p["n_theta"])
        jobs = [(N, Dx, Dy, float(theta), rho_x, rho_y,
                 p["seed"] + g * 1_000_000 + j * 10_000 + t)
                for j, theta in enumerate(theta_values) for t in range(p["n_trials"])]
        values = np.asarray(parallel_map(trial, jobs, workers, desc=name))
        values = values.reshape(p["n_theta"], p["n_trials"])
        rows.append({
            "name": name, "alpha_x": alpha_x, "alpha_y": alpha_y,
            "theta_values": theta_values, "theta_crit": theta_c,
            "theta_c_iso": float(theory.theta_crit_iso(alpha_x, alpha_y, rho_x * rho_y)),
            "Rx2_mean": values.mean(axis=1), "Rx2_std": values.std(axis=1),
            "theory": np.asarray(theory.overlap_curves(alpha_x, alpha_y, rho_x, rho_y,
                                                       theta_values)["Rx2"]),
            "theory_iso": np.asarray([theory.isotropic_reference(alpha_x, alpha_y,
                                                                 rho_x * rho_y, theta)[0]
                                      for theta in theta_values]),
        })
    return {"params": p, "aspect_ratios": {"configs": rows}}


if __name__ == "__main__":
    save_results("aspect_ratio_sensitivity", run(PARAMS, workers_from_argv()))
