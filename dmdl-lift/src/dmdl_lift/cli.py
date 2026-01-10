"""Command-line interface for dmdl-lift validation."""
import matplotlib
matplotlib.use("Agg")  # Headless backend for CI

import sys
from dmdl_lift.validate import validate_method
from dmdl_lift.plot import plot_results
import matplotlib.pyplot as plt


def main():
    """Run Phase 0 validation and generate report figure."""
    print("Running hardened Phase 0 validation...")
    results = validate_method(L=6)
    if results is not None:
        fig = plot_results(results)
        fig.savefig("phase0_hardened_validation.png", dpi=150, bbox_inches='tight')
        print("\nValidation output above. Figure saved: phase0_hardened_validation.png")
        if results['success']:
            print("Validation PASSED.")
        else:
            print("Validation FAILED.")
        plt.close(fig)
    sys.exit(0 if results and results['success'] else 1)


if __name__ == "__main__":
    main()
