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
    mean_imputation_pls,
    em_pls,
    iterative_svd_pls,
    oracle_pls,
)

from .data import (
    whiten_to_identity,
    apply_mcar,
    generate_data,
    generate_data_non_gaussian,
    generate_semi_synthetic,
    generate_data_mar,
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
    # MAR runners
    run_single_trial_mar,
    run_multiple_trials_mar,
    _run_mar_worker,
    _run_mar_grid_worker,
    # All-methods comparison runners
    run_single_trial_all_methods,
    run_multiple_trials_all_methods,
    _run_all_methods_worker,
    _run_all_methods_missingness_worker,
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
    'mean_imputation_pls',
    'em_pls',
    'iterative_svd_pls',
    'oracle_pls',
    # Data
    'whiten_to_identity',
    'apply_mcar',
    'generate_data',
    'generate_data_non_gaussian',
    'generate_semi_synthetic',
    'generate_data_mar',
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
    # MAR runners
    'run_single_trial_mar',
    'run_multiple_trials_mar',
    '_run_mar_worker',
    '_run_mar_grid_worker',
    # All-methods comparison runners
    'run_single_trial_all_methods',
    'run_multiple_trials_all_methods',
    '_run_all_methods_worker',
    '_run_all_methods_missingness_worker',
]
