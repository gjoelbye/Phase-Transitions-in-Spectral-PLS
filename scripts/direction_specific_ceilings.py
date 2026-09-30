"""Figure 8 and Table 2: direction-specific recovery ceilings.

- Figure 8 (a): R_x^2 against the ceiling kappa_tau at high signal, for the
  leading PLS directions and random directions of the TCGA-BRCA and PBMC
  designs, and for whitened synthetic designs with heterogeneous row leverage.
- Figure 8 (b): finite-signal R_x^2 curves as the leverage contrast grows.
- Table 2: kappa_tau of the leading directions at three retentions.
- Printed numbers only: finite-signal curves of the named leverage profiles,
  matched random-design controls, and detection sweeps of the top singular value.

Needs the raw data under data/ (see README). Saves results/direction_specific_ceilings.pkl.
Run from the repository root: python -m scripts.direction_specific_ceilings --workers 8
"""
import numpy as np

from src import theory
from src.data import (generate_heterogeneous_whitened, leverage_coupled_direction,
                      load_pbmc_whitened, load_tcga_whitened, planted_directions)
from src.methods import compute_overlaps, pls_svd
from src.utils import parallel_map, save_results, workers_from_argv

PARAMS = dict(
    # Biological designs, reduced to 200 principal components per view
    biological_Dx=200, biological_Dy=200,
    retention=0.7,          # rho_x = rho_y on the biological designs
    k_biological=8,         # leading PLS directions used as planted directions
    n_random=30,            # random planted directions
    n_trials=100,
    theta_factor=30.0,      # "high signal" means theta = 30 theta_c
    direction_seed=1234, pbmc_trial_seed_offset=10000,
    survey_retentions=(0.9, 0.7, 0.5),   # columns of Table 2
    # Heterogeneous-leverage synthetic designs
    leverage_N=2000, leverage_Dx=500, leverage_Dy=500,
    leverage_rho_x=0.5, leverage_rho_y=1.0,
    leverage_trials=40, leverage_n_theta=20,
    leverage_fraction=0.3,  # a fraction of rows has its scale raised by sqrt(ratio)
    leverage_ratios=(1.0, 1.4, 2.0, 2.7, 3.6, 5.0, 7.0, 10.0, 14.0, 18.0),
    leverage_weight=1.0,    # direction fully coupled to leverage in the finite-signal sweeps
    high_signal_profiles=(  # (name, fraction of raised rows, ratio)
        ("uniform", 0.0, 1.0),
        ("mild", 0.3, 2.0),
        ("moderate", 0.3, 4.0),
        ("strong", 0.3, 9.0),
    ),
    direction_weights=(-1.0, -0.5, 0.0, 0.5, 1.0),
    # Controls and detection
    direction_control_seed=20260813,
    detection_missingness_values=(0.3, 0.5),
    detection_null_trials=500, detection_trials=200, detection_n_theta=40,
    detection_theta_min_factor=0.5, detection_theta_max_factor=1.5,
    detection_seed=20260811, detection_control_seed=20260812,
    seed=20260811,
)


# Biological designs

def oracle_overlap(X, score, Sy, v0, u0):
    """R_x^2 of the left singular vector of the noise-free masked cross-covariance."""
    vectors, _, _ = np.linalg.svd(X.T @ ((score[:, None] * Sy) * v0[None, :]),
                                  full_matrices=False)
    return float(vectors[:, 0] @ u0) ** 2


def biological_direction(job):
    """Measured and noise-free R_x^2 for one planted pair on a biological design."""
    kind, index, u0, v0, X_w, theta, retention, n_trials, seed = job
    N, Dx = X_w.shape
    Dy = v0.size
    score = X_w @ u0
    measured, oracle = np.empty(n_trials), np.empty(n_trials)
    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        Y_star = theta * np.outer(score, v0) + rng.normal(size=(N, Dy))
        Sx = rng.binomial(1, retention, size=(N, Dx))
        Sy = rng.binomial(1, retention, size=(N, Dy))
        X = Sx * X_w
        u_hat, v_hat, _ = pls_svd(X, Sy * Y_star)
        measured[trial] = compute_overlaps(u_hat, v_hat, u0, v0)[0]
        oracle[trial] = oracle_overlap(X, score, Sy, v0, u0)
    tau = float(theory.tau_N(score, np.sum(X_w ** 2, axis=1), N))
    return {"kind": kind, "index": index, "tau": tau,
            "kappa_tau": float(theory.kappa_tau(retention, tau)),
            "Rx2": measured, "oracle": oracle}


def biological_dataset(name, X_w, Y_w, p, workers):
    """Leading PLS directions of the data and random directions as planted pairs."""
    N, Dx = X_w.shape
    Dy = Y_w.shape[1]
    alpha_x, alpha_y = N / Dx, N / Dy
    theta_c = float(theory.theta_c(alpha_x, alpha_y, p["retention"], p["retention"]))
    theta = p["theta_factor"] * theta_c
    U, _, Vt = np.linalg.svd(X_w.T @ Y_w / N, full_matrices=False)
    directions = [("bio", i, U[:, i], Vt[i]) for i in range(p["k_biological"])]
    rng = np.random.default_rng(p["direction_seed"])
    for i in range(p["n_random"]):
        directions.append(("random", i, *planted_directions(Dx, Dy, rng)))
    seed = p["seed"] + (0 if name == "tcga" else p["pbmc_trial_seed_offset"])
    jobs = [(*direction, X_w, theta, p["retention"], p["n_trials"], seed)
            for direction in directions]
    return {
        "rows": parallel_map(biological_direction, jobs, workers, desc=name),
        "theta": theta, "theta_c": theta_c,
        "kappa_uniform": float(theory.kappa(alpha_x, p["retention"])),
        "alpha_x": alpha_x, "alpha_y": alpha_y, "N": N, "Dx": Dx, "Dy": Dy,
        "n_trials": p["n_trials"],
    }


def incoherence(X_w, Y_w, u0, v0):
    """Delocalization of u0, v0, the design and the score, each on a log scale."""
    N, Dx = X_w.shape
    Dy = Y_w.shape[1]
    score = X_w @ u0
    return (float(np.abs(u0).max() * np.sqrt(Dx) / np.log(Dx)),
            float(np.abs(v0).max() * np.sqrt(Dy) / np.log(Dy)),
            float(np.abs(X_w).max() / np.log(N)),
            float(np.abs(score).max() / np.log(N)))


def survey(datasets, raw_data, p):
    """Table 2: kappa_tau of the leading directions at several retentions."""
    output = {}
    for name in ("tcga", "pbmc"):
        X_w, Y_w = raw_data[name]
        N, Dx = X_w.shape
        U, _, Vt = np.linalg.svd(X_w.T @ Y_w / N, full_matrices=False)
        taus = {row["index"]: row["tau"] for row in datasets[name]["rows"] if row["kind"] == "bio"}
        rows = [{"index": i, "tau": float(taus[i]),
                 "kappa": {f"{rho:.1f}": float(theory.kappa_tau(rho, float(taus[i])))
                           for rho in p["survey_retentions"]},
                 "diagnostics": incoherence(X_w, Y_w, U[:, i], Vt[i])}
                for i in range(p["k_biological"])]
        output[name] = {"rows": rows, "N": N, "Dx": Dx, "Dy": Y_w.shape[1]}
    return output


# Heterogeneous-leverage synthetic designs

def leverage_trial(job):
    """R_x^2 and realized leverage coupling for one heterogeneous-design trial."""
    cfg, scales, weight, theta, trial = job
    N, Dx, Dy = cfg["N"], cfg["Dx"], cfg["Dy"]
    seed = cfg["seed"] + trial
    X_star = generate_heterogeneous_whitened(N, Dx, np.asarray(scales), seed=seed)
    u0 = leverage_coupled_direction(X_star, weight, seed=seed + 1)
    rng = np.random.default_rng(seed + 2)
    v0 = rng.standard_normal(Dy)
    v0 /= np.linalg.norm(v0)
    r = np.sum(X_star ** 2, axis=1)
    b = X_star @ u0
    Y_star = theta * np.outer(b, v0) + rng.standard_normal((N, Dy))
    Sx = rng.binomial(1, cfg["rho_x"], (N, Dx))
    Sy = rng.binomial(1, cfg["rho_y"], (N, Dy))
    u_hat, _, _ = pls_svd(Sx * X_star, Sy * Y_star)
    return {"profile": cfg["profile"], "weight": float(weight), "theta": float(theta),
            "trial": trial, "tau": float(theory.tau_N(b, r, N)), "cv": float(r.std() / r.mean()),
            "Rx2": float(u_hat @ u0) ** 2}


def leverage(p, workers):
    N, Dx, Dy = p["leverage_N"], p["leverage_Dx"], p["leverage_Dy"]
    rho_x, rho_y = p["leverage_rho_x"], p["leverage_rho_y"]
    alpha_x, alpha_y = N / Dx, N / Dy
    theta_c = float(theory.theta_c(alpha_x, alpha_y, rho_x, rho_y))
    theta_values = np.linspace(0.5, 2.5, p["leverage_n_theta"]) * theta_c
    theta_high = p["theta_factor"] * theta_c
    trials = range(p["leverage_trials"])

    def cfg(profile):
        return {"N": N, "Dx": Dx, "Dy": Dy, "rho_x": rho_x, "rho_y": rho_y,
                "seed": p["seed"], "profile": profile}

    def scales(fraction, ratio):
        """Row scales: the first fraction * N rows get sqrt(ratio), the rest 1."""
        s = np.ones(N)
        s[:int(round(fraction * N))] = np.sqrt(ratio)
        return s

    # Finite-signal curves as the leverage ratio grows (fully coupled direction)
    jobs = [(cfg(f"r{ratio:g}"), scales(p["leverage_fraction"], ratio), p["leverage_weight"],
             float(theta), t)
            for ratio in p["leverage_ratios"] for theta in theta_values for t in trials]
    # High signal: every profile and direction weight
    high_jobs = [(cfg(name), scales(fraction, ratio), weight, float(theta_high), t)
                 for name, fraction, ratio in p["high_signal_profiles"]
                 for weight in p["direction_weights"] for t in trials]
    # Finite-signal curves for the named profiles
    profile_jobs = [(cfg(name), scales(fraction, ratio), p["leverage_weight"], float(theta), t)
                    for name, fraction, ratio in p["high_signal_profiles"]
                    for theta in theta_values for t in trials]
    rows = parallel_map(leverage_trial, jobs, workers, desc="leverage ratios")
    high_rows = parallel_map(leverage_trial, high_jobs, workers, desc="high signal")
    profile_rows = parallel_map(leverage_trial, profile_jobs, workers, desc="profiles")

    profiles = []
    for name, _, _ in p["high_signal_profiles"]:
        for weight in p["direction_weights"]:
            selected = [r for r in high_rows if r["profile"] == name and r["weight"] == weight]
            tau = float(np.mean([r["tau"] for r in selected]))
            profiles.append({"profile": name, "weight": weight, "tau": tau,
                             "cv": float(np.mean([r["cv"] for r in selected])),
                             "kappa_tau": float(theory.kappa_tau(rho_x, tau)),
                             "Rx2": np.asarray([r["Rx2"] for r in selected])})
    shared = {"rho_x": rho_x, "rho_y": rho_y, "alpha_x": alpha_x, "alpha_y": alpha_y,
              "N": N, "Dx": Dx, "Dy": Dy, "n_trials": p["leverage_trials"]}
    return {
        "high_signal_profiles": {
            "rows": profiles, "trial_rows": high_rows, "theta": theta_high,
            "theta_c": theta_c, "kappa_uniform": float(theory.kappa(alpha_x, rho_x)), **shared,
        },
        "finite_signal_leverage": {
            "rows": rows, "theta_values": theta_values, "theta_c": theta_c,
            "ratios": p["leverage_ratios"], "fraction": p["leverage_fraction"],
            "weight": p["leverage_weight"], **shared,
        },
        "finite_signal_profiles": {
            "rows": profile_rows, "theta_values": theta_values, "theta_c": theta_c,
            "profiles": p["high_signal_profiles"], "weight": p["leverage_weight"], **shared,
        },
    }


# Matched random-design controls

def control_trial(job):
    X_w, u0, v0, theta, retention, seed = job
    rng = np.random.default_rng(seed)
    N, Dx = X_w.shape
    Y_star = theta * np.outer(X_w @ u0, v0) + rng.normal(size=(N, v0.size))
    Sx = rng.binomial(1, retention, size=(N, Dx))
    Sy = rng.binomial(1, retention, size=(N, v0.size))
    u_hat, _, _ = pls_svd(Sx * X_w, Sy * Y_star)
    return float(u_hat @ u0) ** 2


def direction_controls(raw_data, p, workers):
    """A Haar design of each biological sample size, at high signal."""
    output = {}
    for name, (X_data, _) in raw_data.items():
        N = X_data.shape[0]
        Dx, Dy = p["biological_Dx"], p["biological_Dy"]
        rng = np.random.default_rng(p["direction_control_seed"])
        Q, _ = np.linalg.qr(rng.normal(size=(N, Dx)), mode="reduced")
        X_w = Q * np.sqrt(N)
        u0, v0 = planted_directions(Dx, Dy, rng)
        alpha_x, alpha_y = N / Dx, N / Dy
        theta = p["theta_factor"] * float(theory.theta_c(alpha_x, alpha_y,
                                                         p["retention"], p["retention"]))
        jobs = [(X_w, u0, v0, theta, p["retention"], p["direction_control_seed"] + 1000 + t)
                for t in range(p["n_trials"])]
        values = np.asarray(parallel_map(control_trial, jobs, workers, desc=f"control {name}"))
        tau = theory.tau_N(X_w @ u0, np.sum(X_w ** 2, axis=1), N)
        kappa_uniform = float(theory.kappa(alpha_x, p["retention"]))
        output[name] = {
            "Rx2_mean": float(values.mean()),
            "Rx2_se": float(values.std() / np.sqrt(values.size)),
            "kappa_tau_N": float(theory.kappa_tau(p["retention"], tau)),
            "kappa_uniform": kappa_uniform,
            # Gap between kappa and the finite-signal prediction at this theta
            "shortfall": float(kappa_uniform - theory.overlaps_at(
                alpha_x, alpha_y, p["retention"], p["retention"], theta)[0]),
            "theta": theta, "N": N, "Dx": Dx, "Dy": Dy,
        }
    return output


# Detection of the top singular value

def detection_trial(job):
    X_w, u0, v0, retention, theta, seed = job
    rng = np.random.default_rng(seed)
    N, Dx = X_w.shape
    Y_star = theta * np.outer(X_w @ u0, v0) + rng.normal(size=(N, v0.size))
    Sx = rng.binomial(1, retention, size=(N, Dx))
    Sy = rng.binomial(1, retention, size=(N, v0.size))
    _, _, sigma1 = pls_svd(Sx * X_w, Sy * Y_star)
    return float(sigma1)


def detection_sweep(name, X_w, u0, v0, missingness, p, workers):
    """sigma_1 under the null (theta = 0) and on a signal grid around theta_c."""
    N, Dx = X_w.shape
    alpha_x, alpha_y = N / Dx, N / v0.size
    retention = 1 - missingness
    theta_c = float(theory.theta_c(alpha_x, alpha_y, retention, retention))
    theta = np.linspace(p["detection_theta_min_factor"] * theta_c,
                        p["detection_theta_max_factor"] * theta_c, p["detection_n_theta"])
    null_jobs = [(X_w, u0, v0, retention, 0.0, p["detection_seed"] + t)
                 for t in range(p["detection_null_trials"])]
    signal_jobs = [(X_w, u0, v0, retention, float(value), p["detection_seed"] + t)
                   for value in theta for t in range(p["detection_trials"])]
    null_sigma1 = np.asarray(parallel_map(detection_trial, null_jobs, workers, desc=f"{name} null"))
    sigma1 = np.asarray(parallel_map(detection_trial, signal_jobs, workers, desc=name))
    return {"theta": theta, "sigma1": sigma1.reshape(p["detection_n_theta"], p["detection_trials"]),
            "null_sigma1": null_sigma1, "theta_c": theta_c}


def detection(raw_data, p, workers):
    """Leading PLS direction of each dataset, with a matched Haar-design control."""
    output = {}
    for dataset, (X_w, Y_w) in raw_data.items():
        N, Dx = X_w.shape
        U, _, Vt = np.linalg.svd(X_w.T @ Y_w / N, full_matrices=False)
        rng = np.random.default_rng(p["detection_control_seed"])
        Q, _ = np.linalg.qr(rng.normal(size=(N, Dx)), mode="reduced")
        control_X = Q * np.sqrt(N)
        control_u, control_v = planted_directions(Dx, Vt.shape[1], rng)
        for missingness in p["detection_missingness_values"]:
            name = f"{dataset}_m{int(round(missingness * 100)):02d}"
            block = detection_sweep(name, X_w, U[:, 0], Vt[0], missingness, p, workers)
            block["control"] = detection_sweep(f"{name}_control", control_X, control_u,
                                               control_v, missingness, p, workers)
            output[name] = block
    return output


def run(p, workers=1):
    raw_data = {
        "tcga": load_tcga_whitened(p["biological_Dx"], p["biological_Dy"]),
        "pbmc": load_pbmc_whitened(p["biological_Dx"], p["biological_Dy"]),
    }
    datasets = {name: biological_dataset(name, *raw_data[name], p, workers)
                for name in ("tcga", "pbmc")}
    return {
        "params": p,
        **datasets,
        **leverage(p, workers),
        "survey": survey(datasets, raw_data, p),
        "direction_controls": direction_controls(raw_data, p, workers),
        "detection": detection(raw_data, p, workers),
    }


if __name__ == "__main__":
    save_results("direction_specific_ceilings", run(PARAMS, workers_from_argv()))
