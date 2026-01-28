"""
Core model definitions and theoretical predictions.

This module contains:
- ModelParams: dataclass encapsulating model parameters
- theoretical_overlaps: replica theory predictions
- inv_sqrtm_psd: matrix inverse square root helper
- theoretical_sigma1: predicted top singular value
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelParams:
    """Parameters for the spiked two-view model with missingness.

    Attributes:
        N: Number of samples
        Dx: Dimension of X
        Dy: Dimension of Y
        theta: Signal strength
        mx: Missingness rate in X
        my: Missingness rate in Y
        u0: True direction in X (generated if None)
        v0: True direction in Y (generated if None)
    """
    N: int
    Dx: int
    Dy: int
    theta: float
    mx: float
    my: float
    u0: Optional[np.ndarray] = None
    v0: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.u0 is None:
            self.u0 = np.random.randn(self.Dx)
            self.u0 /= np.linalg.norm(self.u0)
        if self.v0 is None:
            self.v0 = np.random.randn(self.Dy)
            self.v0 /= np.linalg.norm(self.v0)

    @property
    def alpha_x(self) -> float:
        """Aspect ratio N/Dx."""
        return self.N / self.Dx

    @property
    def alpha_y(self) -> float:
        """Aspect ratio N/Dy."""
        return self.N / self.Dy

    @property
    def rho_x(self) -> float:
        """Retention probability in X."""
        return 1 - self.mx

    @property
    def rho_y(self) -> float:
        """Retention probability in Y."""
        return 1 - self.my

    @property
    def rho(self) -> float:
        """Joint retention probability."""
        return self.rho_x * self.rho_y

    @property
    def theta_eff(self) -> float:
        """Effective signal strength under dual masking."""
        return np.sqrt(self.rho) * self.theta

    @property
    def theta_crit(self) -> float:
        """Critical signal strength for phase transition."""
        return 1 / ((self.alpha_x * self.alpha_y)**0.25 * np.sqrt(self.rho))

    @property
    def is_supercritical(self) -> bool:
        """Check if in supercritical regime."""
        return self.alpha_x * self.alpha_y * self.rho**2 * self.theta**4 > 1


def theoretical_overlaps(params: ModelParams) -> tuple[float, float]:
    """
    Compute theoretical overlaps from replica theory.

    Args:
        params: Model parameters

    Returns:
        rx2: Theoretical squared overlap in X direction
        ry2: Theoretical squared overlap in Y direction
    """
    alpha_x = params.alpha_x
    alpha_y = params.alpha_y
    rho = params.rho
    theta = params.theta

    # Check if supercritical
    discriminant = alpha_x * alpha_y * rho**2 * theta**4

    if discriminant <= 1:
        # Subcritical: no recovery
        return 0.0, 0.0

    # Supercritical: use formulas from Theorem 1
    rx2 = (discriminant - 1) / (alpha_y * rho * theta**2 * (alpha_x * rho * theta**2 + 1))
    ry2 = (discriminant - 1) / (alpha_x * rho * theta**2 * (alpha_y * rho * theta**2 + 1))

    return rx2, ry2


def inv_sqrtm_psd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute the inverse square root of a positive semi-definite matrix.

    For symmetric PSD matrix A, returns A^(-1/2) via eigendecomposition.

    Args:
        A: Symmetric positive semi-definite matrix
        eps: Small constant for numerical stability

    Returns:
        A_inv_sqrt: Inverse square root of A
    """
    # Symmetrize (in case of numerical errors)
    A_sym = (A + A.T) / 2

    # Eigendecomposition
    w, V = np.linalg.eigh(A_sym)

    # Regularize small eigenvalues
    w = np.maximum(w, eps)

    # A^(-1/2) = V @ diag(1/sqrt(w)) @ V^T
    A_inv_sqrt = (V / np.sqrt(w)) @ V.T

    return A_inv_sqrt


def theoretical_sigma1(params: ModelParams) -> float:
    """
    Compute theoretical top singular value of cross-covariance.

    In the supercritical regime, the top singular value has a deterministic
    limit related to the signal strength.

    Args:
        params: Model parameters

    Returns:
        sigma1_theory: Theoretical top singular value
    """
    rx2, ry2 = theoretical_overlaps(params)

    if rx2 == 0 or ry2 == 0:
        # Subcritical: sigma_1 is just the bulk edge
        return 1 / np.sqrt(min(params.alpha_x, params.alpha_y))

    # Supercritical: spike emerges
    rx, ry = np.sqrt(rx2), np.sqrt(ry2)
    return np.sqrt(params.rho) * params.theta * rx * ry
