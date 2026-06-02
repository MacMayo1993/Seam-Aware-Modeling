#!/usr/bin/env python3
"""
reproduce.py — End-to-end reproduction script for Seam-Aware Modeling (MASS/SMASH).

Runs the full validation pipeline in sequence and regenerates all outputs:
  1. Basic demo          : examples/mass_smash.py
  2. Baseline comparison : validation/strong_baselines.py
  3. Ablation study      : validation/ablation.py      (if present)
  4. SNR sweep           : validation/snr_sweep.py     (if present)

Usage:
  python reproduce.py          # full run (~5-10 min)
  python reproduce.py --quick  # fast smoke-test (~1 min)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()


def build_env() -> dict:
    """Return an env dict with PROJECT_ROOT prepended to PYTHONPATH."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + ((":" + existing) if existing else "")
    return env


def run_step(
    label: str,
    cmd: list[str],
    *,
    supports_quick: bool = False,
    quick: bool = False,
) -> tuple[bool, float]:
    """
    Run a single pipeline step.

    Returns (success, elapsed_seconds).
    """
    if supports_quick and quick:
        cmd = cmd + ["--quick"]

    print()
    print("=" * 70)
    print(f"STEP: {label}")
    print(f"CMD : {' '.join(cmd)}")
    print("=" * 70)

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            env=build_env(),
            capture_output=False,   # let output stream live to terminal
            text=True,
        )
        elapsed = time.monotonic() - t0
        success = result.returncode == 0
        if not success:
            print(f"\n[FAILED] Exit code {result.returncode} after {elapsed:.1f}s")
        else:
            print(f"\n[PASSED] Completed in {elapsed:.1f}s")
        return success, elapsed
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(f"\n[ERROR] {exc}")
        return False, elapsed


def optional_step(
    label: str,
    script: Path,
    *,
    extra_args: list[str] | None = None,
    supports_quick: bool = False,
    quick: bool = False,
) -> tuple[bool | None, float]:
    """Run an optional step; return (None, 0) if the script doesn't exist."""
    if not script.exists():
        print(f"\n[SKIP] {label} — {script.name} not found")
        return None, 0.0
    cmd = [sys.executable, str(script)] + (extra_args or [])
    return run_step(label, cmd, supports_quick=supports_quick, quick=quick)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce all MASS/SMASH validation outputs."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Pass --quick to any script that supports it (fast smoke-test).",
    )
    args = parser.parse_args()

    print()
    print("*" * 70)
    print("  Seam-Aware Modeling — Full Validation Pipeline")
    print("  -----------------------------------------------")
    print("  Runs each validation step in sequence, captures output, and")
    print("  prints a summary at the end.")
    if args.quick:
        print("  Mode: --quick (smoke-test, reduced iterations)")
    else:
        print("  Mode: full (may take 5-10 minutes)")
    print("*" * 70)

    wall_t0 = time.monotonic()
    results: list[tuple[str, bool | None, float]] = []

    # ------------------------------------------------------------------ #
    # Step 1: Basic demo
    # ------------------------------------------------------------------ #
    ok, dt = run_step(
        "Basic demo (examples/mass_smash.py)",
        [
            sys.executable,
            str(PROJECT_ROOT / "examples" / "mass_smash.py"),
            "--no-plot",
            "--seed", "42",
        ],
    )
    results.append(("Basic demo", ok, dt))

    # ------------------------------------------------------------------ #
    # Step 2: Strong baselines comparison table
    # ------------------------------------------------------------------ #
    ok, dt = run_step(
        "Strong baselines (validation/strong_baselines.py)",
        [sys.executable, str(PROJECT_ROOT / "validation" / "strong_baselines.py")],
    )
    results.append(("Strong baselines", ok, dt))

    # ------------------------------------------------------------------ #
    # Step 3: Ablation study (optional)
    # ------------------------------------------------------------------ #
    ok, dt = optional_step(
        "Ablation study (validation/ablation.py)",
        PROJECT_ROOT / "validation" / "ablation.py",
        supports_quick=False,
        quick=args.quick,
    )
    results.append(("Ablation study", ok, dt))

    # ------------------------------------------------------------------ #
    # Step 4: SNR sweep (optional)
    # ------------------------------------------------------------------ #
    ok, dt = optional_step(
        "SNR sweep (validation/snr_sweep.py)",
        PROJECT_ROOT / "validation" / "snr_sweep.py",
        supports_quick=False,
        quick=args.quick,
    )
    results.append(("SNR sweep", ok, dt))

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    total = time.monotonic() - wall_t0
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = failed = skipped = 0
    for name, status, elapsed in results:
        if status is None:
            tag = "SKIP "
            skipped += 1
        elif status:
            tag = "PASS "
            passed += 1
        else:
            tag = "FAIL "
            failed += 1
        print(f"  [{tag}]  {name:<40}  {elapsed:6.1f}s")

    print("-" * 70)
    print(f"  Passed: {passed}   Failed: {failed}   Skipped: {skipped}")
    print(f"  Total wall time: {total:.1f}s")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
