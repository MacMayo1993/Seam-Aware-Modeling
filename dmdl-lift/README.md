# dmdl-lift

Hardened validation pipeline for detecting rank-2 lift in time series using ΔMDL model selection.
Validates that rank-2 (phase-conditioned AR(1)) is preferred only for rich drives (Fibonacci word) and not for simple periodic ones.

## Installation

```bash
pip install -e .
```

## Usage

```bash
dmdl-validate
```

This runs the hardened Phase 0, prints results, and saves phase0_hardened_validation.png in the current directory.

## Why ΔMDL?

ΔMDL compares rank-1 (global AR(1)) vs. rank-2 (per-phase AR(1)) models. Positive ΔMDL indicates preference for lift/structure.

## Pass criteria (Frozen L=6)

- **Periodic**: μ < 0, phases = 2, ESS/phase > 50
- **Fibonacci**: μ > 50, phases ≥ 6, ESS/phase > 15, separation > 40

## Example Output

![Validation Output](phase0_hardened_validation.png)
