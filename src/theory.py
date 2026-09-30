"""Mask-aware PLS-SVD predictions from Section 3 and Appendix E."""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import NamedTuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import fsolve, minimize_scalar

# Branch-trace continuation bound and grid resolution.
_ZETA_HI = 40.0
_N_GRID = 6000


def kappa(alpha_x: float, rho_x: float) -> float:
    """Equal-leverage recovery ceiling, Eq. (9)."""
    return alpha_x * rho_x / (alpha_x * rho_x + 1.0 - rho_x)


def n_a(alpha_x: float, rho_x: float) -> float:
    """Squared-norm limit of the corrupted direction."""
    return rho_x + (1.0 - rho_x) / alpha_x


def tau_N(b, row_sq_norms, N: int) -> float:
    """Finite-sample score-leverage coupling, Eq. (8)."""
    b = np.asarray(b, dtype=float)
    r = np.asarray(row_sq_norms, dtype=float)
    return float(np.sum(b**2 * r) / N**2)


def kappa_tau(rho_x: float, tau: float) -> float:
    """Direction-specific ceiling of Theorem 3.3.

    Under equal leverage tau = 1/alpha_x, and this reduces to ``kappa``.
    """
    return rho_x / (rho_x + (1.0 - rho_x) * tau)


def theta_crit_iso(alpha_x: float, alpha_y: float, rho: float) -> float:
    """Isotropic comparator threshold at joint retention ``rho``."""
    return 1.0 / ((alpha_x * alpha_y) ** 0.25 * np.sqrt(rho))


def isotropic_reference(alpha_x: float, alpha_y: float, rho: float,
                        theta: float):
    """Isotropic closed-form overlaps, or zeros below their threshold."""
    discriminant = alpha_x * alpha_y * rho**2 * theta**4
    if discriminant <= 1:
        return 0.0, 0.0
    rx2 = (discriminant - 1) / (alpha_y * rho * theta**2
                                * (alpha_x * rho * theta**2 + 1))
    ry2 = (discriminant - 1) / (alpha_x * rho * theta**2
                                * (alpha_y * rho * theta**2 + 1))
    return float(rx2), float(ry2)


def _fsolve_quiet(func, x0, **kwargs):
    """Suppress expected slow-progress warnings from branch continuation."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fsolve(func, x0, **kwargs)


def _gh(z: float, e1: float, e2: float, rho_x: float):
    return (1.0 / ((z - e2) - rho_x / (z - e1)),
            1.0 / ((z - e1) - rho_x / (z - e2)))


def inner(z: float, rho_x: float, alpha_x: float, warm=(0.0, 0.0)):
    """Solve the mask-averaged fixed point at ``z = sqrt(w)``."""
    z = float(z)
    sigma2, c = 1.0 - rho_x, 1.0 / alpha_x

    def residual(e):
        e1, e2 = float(e[0]), float(e[1])
        g, h = _gh(z, e1, e2, rho_x)
        return [e1 - sigma2 * c * g,
                e2 - sigma2 * (c * h + (1.0 - c) / (z - e1))]

    sol = _fsolve_quiet(residual, np.asarray(warm, dtype=float), xtol=1e-14)
    e1, e2 = float(sol[0]), float(sol[1])
    g, h = _gh(z, e1, e2, rho_x)
    return g, h, e1, e2


def _jacobian_det(z: float, e1: float, e2: float, rho_x: float,
                  alpha_x: float, eps: float = 1e-7) -> float:
    """Jacobian determinant of the (e1, e2) map.

    The physical branch of Lemma E.4 ends where this degenerates.
    """
    sigma2, c = 1.0 - rho_x, 1.0 / alpha_x

    def residual(e1_, e2_):
        g, h = _gh(z, e1_, e2_, rho_x)
        return np.array([e1_ - sigma2 * c * g,
                         e2_ - sigma2 * (c * h + (1.0 - c) / (z - e1_))])

    base = residual(e1, e2)
    col0 = (residual(e1 + eps, e2) - base) / eps
    col1 = (residual(e1, e2 + eps) - base) / eps
    return float(np.linalg.det(np.column_stack([col0, col1])))


def solve_supercritical(alpha_x, alpha_y, rho_x, rho_y, theta):
    """Solve the supercritical outlier system of Proposition E.7.

    Candidate roots must have ``Z > 0``, ``delta`` in ``(-1, 0)``, and a
    spectral parameter at or above the traced bulk edge.
    """
    alpha_x, alpha_y, rho_x = float(alpha_x), float(alpha_y), float(rho_x)
    na = n_a(alpha_x, rho_x)
    s2 = theta ** 2 * rho_y * na

    def residual(x):
        Z, delta = float(x[0]), float(x[1])
        w = alpha_y * (1.0 + delta) * Z ** 2
        z = np.sqrt(w)
        g, h, _, _ = inner(z, rho_x, alpha_x)
        return [delta + (1.0 + delta) * (alpha_y / alpha_x) * (z * g - 1.0),
                s2 * alpha_y * (z * h - 1.0) / na - 1.0]

    tr = _branch_trace(alpha_x, alpha_y, rho_x)

    def attempt(x0):
        sol = _fsolve_quiet(residual, np.asarray(x0, dtype=float),
                            xtol=1e-13)
        Z, delta = abs(float(sol[0])), float(sol[1])
        if not (np.isfinite(Z) and np.isfinite(delta) and -1.0 < delta < 0.0):
            return False, Z, delta
        r = residual([Z, delta])
        zeta = np.sqrt(alpha_y * (1.0 + delta) * Z ** 2)
        ok = (max(abs(r[0]), abs(r[1])) < 1e-9
              and zeta >= tr.zeta_edge - 1e-7)
        return ok, Z, delta

    starts = [(max(2.0, 1.5 * theta), -0.3)]
    theta1_eq = theta * np.sqrt(rho_y)          # rho_y only rescales theta
    if tr.theta1[0] <= theta1_eq <= tr.theta1[-1]:
        starts.append((float(np.interp(theta1_eq, tr.theta1, tr.z_hat)),
                       float(np.interp(theta1_eq, tr.theta1, tr.delta))))
    starts.append((np.sqrt((s2 + 1.0 / alpha_x) * (s2 + 1.0 / alpha_y) / s2),
                   -min(0.3, 1.0 / (alpha_y * s2 + 1.0))))
    for x0 in starts:
        ok, Z, delta = attempt(x0)
        if ok:
            break
    else:
        raise RuntimeError("no physical outlier root; is theta above theta_c?")
    z = np.sqrt(alpha_y * (1.0 + delta) * Z ** 2)
    g, h, e1, _ = inner(z, rho_x, alpha_x)
    return Z, delta, z, g, h, e1, s2, na


def transforms(alpha_x, alpha_y, rho_x, rho_y, theta):
    """Overlaps R_x^2 and R_y^2 of Eq. (12), from the transforms (E.3) at the outlier."""
    Z, delta, z, g, h, e1, s2, na = solve_supercritical(
        alpha_x, alpha_y, rho_x, rho_y, theta)
    K = Z * alpha_y * (1.0 + delta)
    m1 = K * (z * h - 1.0) / na
    m1x = K * np.sqrt(rho_x) * g / ((z - e1) * np.sqrt(na))
    m2 = 1.0 / (Z * (1.0 + delta))

    def Dbar(Zq):
        def res(d):
            d = float(np.asarray(d).reshape(-1)[0])
            zz = np.sqrt(alpha_y * (1.0 + d) * Zq ** 2)
            gg, _, _, _ = inner(zz, rho_x, alpha_x)
            return d + (1.0 + d) * (alpha_y / alpha_x) * (zz * gg - 1.0)
        d = float(_fsolve_quiet(res, [delta], xtol=1e-14)[0])
        zz = np.sqrt(alpha_y * (1.0 + d) * Zq ** 2)
        _, hh, _, _ = inner(zz, rho_x, alpha_x)
        return alpha_y * (zz * hh - 1.0) / na

    step = 1e-5
    Dp = (Dbar(Z + step) - Dbar(Z - step)) / (2.0 * step)
    R_y2 = -2.0 * m2 / (s2 * Dp)
    R_x2 = -2.0 * m1 / (s2 * Dp) * (m1x / m1) ** 2
    return R_x2, R_y2


# One trace serves all response retentions and signal strengths.
class _BranchTrace(NamedTuple):
    theta1: np.ndarray   # theta at rho_y = 1, increasing from the edge
    Rx2: np.ndarray      # Eq. (12) along the branch
    Ry2: np.ndarray
    z_hat: np.ndarray    # predicted outlier position along the branch
    delta: np.ndarray    # solution of (E.2) along the branch
    z_edge: float        # refined top of the deformed bulk
    zeta_edge: float     # auxiliary spectral coordinate at the edge
    Dbar_edge: float     # Dbar at the refined edge, theta*^{-2} in Eq. (11)


def _delta_of(zeta: float, g: float, alpha_x: float,
              alpha_y: float) -> float:
    """The explicit Y-side fixed point (E.2) on the physical branch."""
    A = (alpha_y / alpha_x) * (zeta * g - 1.0)
    return -A / (1.0 + A)


def _deriv_uniform(f: np.ndarray, h: float) -> np.ndarray:
    """Fourth-order derivative on a uniform grid."""
    d = np.empty_like(f)
    d[2:-2] = (f[:-4] - 8.0 * f[1:-3] + 8.0 * f[3:-1] - f[4:]) / (12.0 * h)
    d[0] = (-25.0 * f[0] + 48.0 * f[1] - 36.0 * f[2]
            + 16.0 * f[3] - 3.0 * f[4]) / (12.0 * h)
    d[1] = (-3.0 * f[0] - 10.0 * f[1] + 18.0 * f[2]
            - 6.0 * f[3] + f[4]) / (12.0 * h)
    d[-2] = (3.0 * f[-1] + 10.0 * f[-2] - 18.0 * f[-3]
             + 6.0 * f[-4] - f[-5]) / (12.0 * h)
    d[-1] = (25.0 * f[-1] - 48.0 * f[-2] + 36.0 * f[-3]
             - 16.0 * f[-4] + 3.0 * f[-5]) / (12.0 * h)
    return d


@lru_cache(maxsize=64)
def _branch_trace(alpha_x: float, alpha_y: float, rho_x: float) -> _BranchTrace:
    """Trace and cache the physical branch of Proposition E.7."""

    na = n_a(alpha_x, rho_x)
    sqrt_na = np.sqrt(na)
    rows, warm = [], (0.0, 0.0)
    for zeta in np.linspace(_ZETA_HI, 0.05, _N_GRID):
        g, h, e1, e2 = inner(zeta, rho_x, alpha_x, warm)
        warm = (e1, e2)
        if _jacobian_det(zeta, e1, e2, rho_x, alpha_x) <= 0.0:
            break
        delta = _delta_of(zeta, g, alpha_x, alpha_y)
        if not -1.0 < delta < 0.0:
            break
        z_hat = zeta / np.sqrt(alpha_y * (1.0 + delta))
        Dbar = alpha_y * (zeta * h - 1.0) / na
        K = z_hat * alpha_y * (1.0 + delta)
        m1 = K * (zeta * h - 1.0) / na
        m1x = K * np.sqrt(rho_x) * g / ((zeta - e1) * sqrt_na)
        m2 = 1.0 / (z_hat * (1.0 + delta))
        rows.append((zeta, z_hat, Dbar, m1, m1x, m2, e1, e2, delta))
    rows = np.asarray(rows)

    # Refine the edge inside the coarse bracket.
    i_edge = int(np.argmin(rows[:, 1]))
    warm_edge = (float(rows[i_edge, 6]), float(rows[i_edge, 7]))
    lo = float(rows[min(i_edge + 1, len(rows) - 1), 0])
    hi = float(rows[max(i_edge - 1, 0), 0])

    def _z_hat_of(zeta):
        g, _, _, _ = inner(zeta, rho_x, alpha_x, warm_edge)
        delta = _delta_of(zeta, g, alpha_x, alpha_y)
        if not -1.0 < delta < 0.0:
            return np.inf
        return zeta / np.sqrt(alpha_y * (1.0 + delta))

    res = minimize_scalar(_z_hat_of, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-10})
    zeta_edge = float(res.x)
    g_e, h_e, _, _ = inner(zeta_edge, rho_x, alpha_x, warm_edge)
    delta_e = _delta_of(zeta_edge, g_e, alpha_x, alpha_y)
    z_edge_val = zeta_edge / np.sqrt(alpha_y * (1.0 + delta_e))
    Dbar_edge = alpha_y * (zeta_edge * h_e - 1.0) / na

    # Physical branch, ordered from the edge.
    br = rows[: i_edge + 1][::-1]
    z_hat = br[:, 1]
    Dbar = br[:, 2]
    delta_arr = br[:, 8]
    # Dbar' along the branch, chain-ruled through the uniformly spaced
    # zeta parameter: near the edge dz_hat/dzeta -> 0 and Dbar'(z_hat)
    # diverges, so differentiating in zeta and taking the ratio is far
    # better conditioned than differencing in z_hat directly.
    zeta_br = br[:, 0]
    h_grid = float(zeta_br[1] - zeta_br[0])
    Dbar_prime = (_deriv_uniform(Dbar, h_grid)
                  / _deriv_uniform(z_hat, h_grid))
    s2 = 1.0 / Dbar                       # theta_eff^2 at each branch point
    m1, m1x, m2 = br[:, 3], br[:, 4], br[:, 5]
    Ry2 = -2.0 * m2 / (s2 * Dbar_prime)
    Rx2 = -2.0 * m1 / (s2 * Dbar_prime) * (m1x / m1) ** 2
    theta1 = np.sqrt(1.0 / (na * Dbar))   # theta axis at rho_y = 1

    keep = (np.isfinite(theta1) & np.isfinite(Rx2) & np.isfinite(Ry2))
    theta1, Rx2, Ry2, z_hat, delta_arr = (
        a[keep] for a in (theta1, Rx2, Ry2, z_hat, delta_arr))
    # np.interp needs an increasing abscissa; drop any numerical backtracks.
    mono = np.concatenate(([True], np.diff(theta1) > 0))
    theta1, Rx2, Ry2, z_hat, delta_arr = (
        a[mono] for a in (theta1, Rx2, Ry2, z_hat, delta_arr))
    return _BranchTrace(theta1, Rx2, Ry2, z_hat, delta_arr,
                        float(z_edge_val), float(zeta_edge), float(Dbar_edge))


def z_edge(alpha_x: float, alpha_y: float, rho_x: float) -> float:
    """Top edge of the deformed noise bulk."""
    tr = _branch_trace(float(alpha_x), float(alpha_y), float(rho_x))
    return tr.z_edge


def theta_star(alpha_x: float, alpha_y: float, rho_x: float) -> float:
    """Edge level ``theta* = Dbar(z_edge)^{-1/2}``."""
    alpha_x, alpha_y, rho_x = float(alpha_x), float(alpha_y), float(rho_x)
    if rho_x == 1.0:   # without design masking the edge is the isotropic one
        return float((alpha_x * alpha_y) ** -0.25)
    tr = _branch_trace(alpha_x, alpha_y, rho_x)
    return float(tr.Dbar_edge ** -0.5)


def theta_c(alpha_x: float, alpha_y: float, rho_x: float,
            rho_y: float) -> float:
    """Phase boundary, Eq. (11)."""
    return theta_star(alpha_x, alpha_y, rho_x) * float(
        np.sqrt(kappa(alpha_x, rho_x) / (rho_x * rho_y)))


def overlaps_at(alpha_x: float, alpha_y: float, rho_x: float, rho_y: float,
                theta: float):
    """Predicted squared overlaps at one signal strength."""
    if theta <= theta_c(alpha_x, alpha_y, rho_x, rho_y):
        return 0.0, 0.0
    R_x2, R_y2 = transforms(alpha_x, alpha_y, rho_x, rho_y, theta)
    return float(R_x2), float(R_y2)


def overlap_curves(alpha_x: float, alpha_y: float, rho_x: float,
                   rho_y: float, theta_grid) -> dict:
    """Predicted overlap curves on a signal-strength grid."""
    theta_grid = np.asarray(theta_grid, dtype=float)
    tr = _branch_trace(float(alpha_x), float(alpha_y), float(rho_x))
    tc = theta_c(alpha_x, alpha_y, rho_x, rho_y)
    theta_branch = tr.theta1 / np.sqrt(rho_y)
    th = np.clip(theta_grid, theta_branch[0], theta_branch[-1])
    rx2 = PchipInterpolator(theta_branch, tr.Rx2)(th)
    ry2 = PchipInterpolator(theta_branch, tr.Ry2)(th)
    sub = theta_grid < tc
    rx2 = np.where(sub, 0.0, np.clip(rx2, 0.0, 1.0))
    ry2 = np.where(sub, 0.0, np.clip(ry2, 0.0, 1.0))
    return {"Rx2": rx2, "Ry2": ry2, "theta_c": tc}
