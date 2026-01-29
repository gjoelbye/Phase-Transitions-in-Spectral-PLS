#!/usr/bin/env python3
"""
Quick test script to verify experimental setup.

Run this before executing the notebooks to ensure all dependencies
are installed and the core functionality works correctly.
"""

import sys


def test_imports():
    """Test that all required packages are available."""
    print("Testing imports...")
    try:
        import numpy as np
        print("  [OK] numpy")
    except ImportError:
        print("  [FAIL] numpy - please install: pip install numpy")
        return False

    try:
        import matplotlib.pyplot as plt
        print("  [OK] matplotlib")
    except ImportError:
        print("  [FAIL] matplotlib - please install: pip install matplotlib")
        return False

    try:
        import seaborn as sns
        print("  [OK] seaborn")
    except ImportError:
        print("  [FAIL] seaborn - please install: pip install seaborn")
        return False

    try:
        from src import ModelParams, generate_data, pls_svd, theoretical_overlaps
        print("  [OK] src (local module)")
    except ImportError as e:
        print(f"  [FAIL] src - {e}")
        return False

    return True


def test_basic_functionality():
    """Test basic model functionality."""
    print("\nTesting basic functionality...")

    try:
        from src import ModelParams, generate_data, pls_svd, compute_overlaps, theoretical_overlaps
        import numpy as np

        # Create simple model
        params = ModelParams(N=100, Dx=30, Dy=25, theta=2.0, mx=0.2, my=0.2)
        print(f"  [OK] Created ModelParams (N={params.N}, Dx={params.Dx}, Dy={params.Dy})")

        # Generate data
        X, Y, Sx, Sy = generate_data(params, seed=42)
        print(f"  [OK] Generated data: X shape {X.shape}, Y shape {Y.shape}")

        # Check missingness
        frac_missing_x = 1 - np.mean(Sx)
        frac_missing_y = 1 - np.mean(Sy)
        print(f"    Observed missingness: X={frac_missing_x:.2%}, Y={frac_missing_y:.2%}")

        # Estimate
        u_hat, v_hat, sigma1 = pls_svd(X, Y)
        print(f"  [OK] PLS-SVD estimate (top singular value = {sigma1:.3f})")

        # Compute overlaps
        Rx2, Ry2 = compute_overlaps(u_hat, v_hat, params.u0, params.v0)
        print(f"    Empirical overlaps: Rx2={Rx2:.3f}, Ry2={Ry2:.3f}")

        # Theoretical prediction
        Rx2_theory, Ry2_theory = theoretical_overlaps(params)
        print(f"    Theoretical overlaps: rx2={Rx2_theory:.3f}, ry2={Ry2_theory:.3f}")

        # Check if close (should be reasonably close even for small N)
        if abs(Rx2 - Rx2_theory) < 0.3:
            print("  [OK] Theory and empirical are reasonably close")
        else:
            print(f"  [WARN] Large gap (expected for small N={params.N})")

        return True

    except Exception as e:
        print(f"  [FAIL] Error in basic functionality: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase_transition():
    """Test that phase transition is detected correctly."""
    print("\nTesting phase transition...")

    try:
        from src import ModelParams, theoretical_overlaps

        # Create two scenarios: subcritical and supercritical
        params_sub = ModelParams(N=500, Dx=200, Dy=200, theta=0.5, mx=0.4, my=0.4)
        params_sup = ModelParams(N=500, Dx=200, Dy=200, theta=3.0, mx=0.4, my=0.4)

        print(f"  Critical threshold theta_crit = {params_sub.theta_crit:.3f}")

        # Subcritical
        Rx2_sub, Ry2_sub = theoretical_overlaps(params_sub)
        print(f"  theta={params_sub.theta:.1f} < theta_crit: rx2={Rx2_sub:.3f}, ry2={Ry2_sub:.3f}", end="")
        if Rx2_sub < 0.01 and Ry2_sub < 0.01:
            print(" [OK] (correctly subcritical)")
        else:
            print(" [FAIL] (should be ~0)")
            return False

        # Supercritical
        Rx2_sup, Ry2_sup = theoretical_overlaps(params_sup)
        print(f"  theta={params_sup.theta:.1f} > theta_crit: rx2={Rx2_sup:.3f}, ry2={Ry2_sup:.3f}", end="")
        if Rx2_sup > 0.3 and Ry2_sup > 0.3:
            print(" [OK] (correctly supercritical)")
        else:
            print(" [FAIL] (should be >0)")
            return False

        print("  [OK] Phase transition working correctly")
        return True

    except Exception as e:
        print(f"  [FAIL] Error in phase transition test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_baselines():
    """Test that baseline methods run without errors."""
    print("\nTesting baseline methods...")

    try:
        from src import (ModelParams, generate_data,
                               mean_imputation_pls, compute_overlaps)

        params = ModelParams(N=100, Dx=30, Dy=25, theta=2.0, mx=0.2, my=0.2)
        X, Y, Sx, Sy = generate_data(params, seed=42)

        # Mean imputation
        u_mi, v_mi = mean_imputation_pls(X, Y, Sx, Sy)
        Rx2_mi, _ = compute_overlaps(u_mi, v_mi, params.u0, params.v0)
        print(f"  [OK] Mean imputation (Rx2={Rx2_mi:.3f})")

        return True

    except Exception as e:
        print(f"  [FAIL] Error in baseline methods: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("PLS-SVD Experimental Setup Test")
    print("="*60)

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False
        print("\n[FAIL] Some dependencies are missing. Please install them.")
        print("   Run: pip install numpy matplotlib seaborn")
        return 1

    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False

    # Test phase transition
    if not test_phase_transition():
        all_passed = False

    # Test baselines
    if not test_baselines():
        all_passed = False

    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("[OK] ALL TESTS PASSED!")
        print("\nYou're ready to run the notebooks!")
    else:
        print("[FAIL] SOME TESTS FAILED")
        print("\nPlease fix the errors above before running the notebooks.")
        return 1
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
