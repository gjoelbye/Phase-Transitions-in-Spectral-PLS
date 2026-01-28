"""
PLS-SVD with missing data: Phase transition analysis.

This package provides tools for analyzing phase transitions in Partial Least
Squares with MCAR (Missing Completely At Random) missingness.

Modules:
    core: Model parameters and theoretical predictions
    methods: PLS-SVD estimation and baseline methods
    data: Data generation utilities
    runners: Experiment runners and parallel workers
"""

from .core import (
    ModelParams,
    theoretical_overlaps,
    inv_sqrtm_psd,
    theoretical_sigma1,
)

from .methods import (
    pls_svd,
    compute_overlaps,
    complete_case_analysis,
    mean_imputation_pls,
)

from .data import (
    whiten_to_identity,
    apply_mcar,
    generate_data,
    generate_data_non_gaussian,
    generate_semi_synthetic,
)

from .runners import (
    run_single_trial,
    run_multiple_trials,
    run_single_trial_pls_only,
    run_multiple_trials_pls_only,
    _run_experiment_worker,
    _run_grid_worker,
    _run_n_worker,
    run_single_trial_non_gaussian,
    run_multiple_trials_non_gaussian,
    _run_non_gaussian_worker,
    _run_diagnostics_worker,
    compute_sigma_ratio,
    split_half_stability,
    bootstrap_direction_variance,
)

__all__ = [
    # Core
    'ModelParams',
    'theoretical_overlaps',
    'inv_sqrtm_psd',
    'theoretical_sigma1',
    # Methods
    'pls_svd',
    'compute_overlaps',
    'complete_case_analysis',
    'mean_imputation_pls',
    # Data
    'whiten_to_identity',
    'apply_mcar',
    'generate_data',
    'generate_data_non_gaussian',
    'generate_semi_synthetic',
    # Runners
    'run_single_trial',
    'run_multiple_trials',
    'run_single_trial_pls_only',
    'run_multiple_trials_pls_only',
    '_run_experiment_worker',
    '_run_grid_worker',
    '_run_n_worker',
    'run_single_trial_non_gaussian',
    'run_multiple_trials_non_gaussian',
    '_run_non_gaussian_worker',
    '_run_diagnostics_worker',
    'compute_sigma_ratio',
    'split_half_stability',
    'bootstrap_direction_variance',
]
