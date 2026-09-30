"""Threshold statistics shared by the experiment scripts and the notebooks."""

import numpy as np

from . import theory


def threshold_grid(alpha_x, alpha_y, rho_x, rho_y, n_theta, n_dense):
    """Signal grid for a threshold sweep.

    Evenly spaced from half the masked threshold theta_c to twice the isotropic
    threshold, plus ``n_dense`` extra points within 20% of each threshold.
    """
    masked = theory.theta_c(alpha_x, alpha_y, rho_x, rho_y)
    isotropic = theory.theta_crit_iso(alpha_x, alpha_y, rho_x * rho_y)
    base = np.linspace(0.5 * masked, 2.0 * isotropic, n_theta - 2 * n_dense)
    dense = [np.linspace(0.8 * t, 1.2 * t, n_dense) for t in (masked, isotropic)]
    return np.unique(np.concatenate([base, *dense]))


def level_crossing(theta, curve, level):
    """Interpolated theta where ``curve`` last rises above ``level``, or None."""
    above = curve > level
    if not above[-1]:
        return None
    below = np.where(~above)[0]
    if len(below) == 0:
        return None
    i = below[-1]
    t0, t1 = theta[i], theta[i + 1]
    f0, f1 = curve[i], curve[i + 1]
    return float(t0 + (level - f0) / (f1 - f0) * (t1 - t0))


def mean_crossing(theta, curve, levels):
    """Average crossing over the recovery levels that ``curve`` crosses.

    Returns the average (None if no level is crossed) and the crossed levels.
    """
    theta = np.asarray(theta, dtype=float)
    curve = np.asarray(curve, dtype=float)
    crossings = {}
    for level in levels:
        crossing = level_crossing(theta, curve, level)
        if crossing is not None:
            crossings[float(level)] = crossing
    mean = float(np.mean(list(crossings.values()))) if crossings else None
    return mean, sorted(crossings)


def threshold_from_levels(theta, rx2_trials, levels, n_boot=2000, boot_seed=0):
    """Measured crossing of the trial-mean R_x^2 curve and a 95% bootstrap interval.

    ``rx2_trials`` has one row per signal strength and one column per trial.
    Whole trial curves are resampled, since a trial shares its design, mask and
    noise seed across signal strengths.
    """
    rx2_trials = np.asarray(rx2_trials, dtype=float)
    n_trials = rx2_trials.shape[1]
    theta_hat, levels_used = mean_crossing(theta, rx2_trials.mean(axis=1), levels)

    rng = np.random.default_rng(boot_seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_trials, size=n_trials)
        crossing, _ = mean_crossing(theta, rx2_trials[:, idx].mean(axis=1), levels_used)
        if crossing is not None:
            draws.append(crossing)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"theta_hat": theta_hat, "ci95": (float(lo), float(hi))}


def detection_threshold(theta, sigma1_trials, null_sigma1, quantile=0.95):
    """Signal strength where half the trials exceed the null quantile of sigma_1."""
    boundary = np.quantile(np.asarray(null_sigma1, dtype=float), quantile)
    probability = (np.asarray(sigma1_trials) > boundary).mean(axis=1)
    return level_crossing(np.asarray(theta, dtype=float), probability, 0.5)
