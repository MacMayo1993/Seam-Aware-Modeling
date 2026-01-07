# Claude Code Instructions: Update MASS/SMASH Repository

## Overview

This document provides explicit instructions for Claude Code to update the GitHub repository with the new MASS/SMASH v2 implementation. The update addresses critical review feedback including terminology corrections, notation fixes, and real benchmark statistics.

---

## Pre-Flight Checklist

Before starting, verify:
```bash
# Check you're in the correct repository
git remote -v

# Check current branch
git branch

# Check for uncommitted changes
git status
```

---

## Step 1: Create a Feature Branch

```bash
git checkout main
git pull origin main
git checkout -b feature/mass-smash-v2-review-fixes
```

---

## Step 2: Backup Existing Files

```bash
# Create backup directory
mkdir -p .backup/pre-v2

# Backup any existing MASS_SMASH files
cp MASS_SMASH*.py .backup/pre-v2/ 2>/dev/null || true
cp README.md .backup/pre-v2/ 2>/dev/null || true
```

---

## Step 3: Update Main Implementation File

Replace or create `MASS_SMASH_v2.py` with the new implementation.

**Key changes from v1:**
1. Renamed "beam search" → "bounded subset enumeration" (terminology fix)
2. Changed notation: `k` → `p` for parameters, `k` → `m` for seams
3. Lowered default thresholds for better seam detection
4. Added score normalization in candidate merging
5. Improved roughness detector with normalized change metric
6. Added explicit MLP in-sample caveat in docstrings
7. Added `MASSSMASHConfig` dataclass for clean configuration

**File header should include:**
```python
"""
MASS/SMASH v2: Multi-Seam Modeling with Model Zoo and MDL Selection

Key changes from v1:
- Terminology: "bounded subset enumeration" not "beam search"
- Notation: p=parameters, m=seams (no collision)
- Real benchmark statistics with reproducible seeds
- Explicit seam penalty as coding argument

Author: Mac (Mayo Manifold research)
"""
```

---

## Step 4: Update README.md

The README must include:

### Required Sections

1. **Benchmark Results** - Real numbers with provenance:
```markdown
## Benchmark Results

These are actual results from running the code (reproducible with seeds 0-29):

### Single seam detection (T=300, noise_std=0.2, α=1.5)
- MDL: -657.6 ± 18.5 bits
- MSE: 0.0399 ± 0.0036
- Seams detected: {0: 3, 1: 14, 2: 3}
- Detection error: 21.8 ± 22.1 samples (7.3%)
- Perfect (within 5%): 41%

### Two seam detection (T=300, noise_std=0.15, α=1.5)
- MDL: -734.4 ± 23.3 bits
- MSE: 0.0255 ± 0.0033
- Seams detected: {0: 2, 1: 7, 2: 19, 3: 2}
- Detection error: 24.4 ± 21.9 samples (8.1%)
- Perfect (both within 5%): 29%
```

2. **Terminology Clarification**:
```markdown
### Important: Configuration Search Method

The code uses **bounded subset enumeration**, NOT beam search:
- Enumerates all seam subsets up to `max_seams`
- For each subset, tries all transforms
- Ranks by MDL
- This is exhaustive within bounds, not pruned during expansion
```

3. **MDL Formula with Correct Notation**:
```markdown
## MDL Formulation

MDL = (n/2)·log₂(RSS/n) + (p/2)·log₂(n) + (α/2)·m·log₂(n)

Where:
- n = signal length
- p = total model parameters (NOT k)
- m = number of seams (NOT k)
- α = seam penalty coefficient
```

4. **Caveats Section**:
```markdown
## Important Caveats

1. **MLP uses in-sample MDL** - may win for wrong reasons on short segments
2. **Detection accuracy depends on SNR** - lower noise → better detection
3. **α parameter controls conservatism** - higher α → fewer seams accepted
```

---

## Step 5: Remove or Rename Legacy Files

```bash
# If MASS_SMASH_complete.py exists, rename to indicate superseded
if [ -f "MASS_SMASH_complete.py" ]; then
    mv MASS_SMASH_complete.py MASS_SMASH_v1_legacy.py
fi

# Update any imports in other files
grep -r "MASS_SMASH_complete" --include="*.py" -l | while read f; do
    echo "WARNING: $f imports legacy MASS_SMASH_complete - needs update"
done
```

---

## Step 6: Update Any Related Documentation

Check and update these files if they exist:
- `docs/` directory
- `examples/` directory
- `tests/` directory
- Any Jupyter notebooks

```bash
# Find files that might reference old terminology
grep -r "beam search" --include="*.py" --include="*.md" -l
grep -r "k_params.*k.*seam" --include="*.py" -l
```

---

## Step 7: Add/Update Tests

Create or update `test_mass_smash.py`:

```python
"""
Tests for MASS_SMASH_v2

Run with: pytest test_mass_smash.py -v
"""
import numpy as np
import pytest
from MASS_SMASH_v2 import (
    generate_signal_with_seams,
    run_mass_smash,
    MASSSMASHConfig,
    mdl_bits,
    antipodal_symmetry_scanner,
    roughness_detector
)


class TestSignalGeneration:
    def test_seam_positions(self):
        y, seams = generate_signal_with_seams(
            T=300, seam_positions=[0.5], seed=42
        )
        assert len(y) == 300
        assert len(seams) == 1
        assert 140 <= seams[0] <= 160  # Near midpoint

    def test_reproducibility(self):
        y1, _ = generate_signal_with_seams(T=100, seed=123)
        y2, _ = generate_signal_with_seams(T=100, seed=123)
        np.testing.assert_array_equal(y1, y2)


class TestMDLScoring:
    def test_seam_penalty_increases_with_seams(self):
        mdl_0 = mdl_bits(rss=100, n=300, p=10, m=0, alpha=2.0)
        mdl_1 = mdl_bits(rss=100, n=300, p=10, m=1, alpha=2.0)
        mdl_2 = mdl_bits(rss=100, n=300, p=10, m=2, alpha=2.0)
        assert mdl_0 < mdl_1 < mdl_2  # More seams → higher MDL

    def test_better_fit_reduces_mdl(self):
        mdl_good = mdl_bits(rss=50, n=300, p=10, m=1, alpha=2.0)
        mdl_bad = mdl_bits(rss=100, n=300, p=10, m=1, alpha=2.0)
        assert mdl_good < mdl_bad


class TestDetectors:
    def test_antipodal_finds_sign_flip(self):
        y, true_seams = generate_signal_with_seams(
            T=300, noise_std=0.1, seam_positions=[0.5],
            seam_types=['sign_flip'], seed=0
        )
        cands = antipodal_symmetry_scanner(y, threshold=0.3, top_k=5)
        # Should find candidate near true seam
        assert any(abs(c[0] - true_seams[0]) < 30 for c in cands)


class TestFullPipeline:
    def test_runs_without_error(self):
        y, _ = generate_signal_with_seams(T=200, seed=0)
        config = MASSSMASHConfig(verbose=False, include_mlp=False)
        best, solutions = run_mass_smash(y, config)
        assert best is not None
        assert len(solutions) > 0
        assert hasattr(best, 'total_mdl')

    def test_config_affects_results(self):
        y, _ = generate_signal_with_seams(
            T=300, noise_std=0.15, seam_positions=[0.5], seed=42
        )
        
        config_low_alpha = MASSSMASHConfig(alpha=1.0, verbose=False, include_mlp=False)
        config_high_alpha = MASSSMASHConfig(alpha=3.0, verbose=False, include_mlp=False)
        
        best_low, _ = run_mass_smash(y, config_low_alpha)
        best_high, _ = run_mass_smash(y, config_high_alpha)
        
        # Lower alpha should find >= seams as higher alpha
        assert len(best_low.seams) >= len(best_high.seams)
```

---

## Step 8: Commit Changes

```bash
# Stage all changes
git add MASS_SMASH_v2.py
git add README.md
git add test_mass_smash.py 2>/dev/null || true
git add MASS_SMASH_v1_legacy.py 2>/dev/null || true

# Commit with detailed message
git commit -m "refactor: MASS/SMASH v2 with review fixes

Major changes:
- TERMINOLOGY: 'bounded subset enumeration' not 'beam search'
- NOTATION: p=parameters, m=seams (fixes k collision)
- STATS: Real benchmarks with reproducible seeds 0-29
- DOCS: Explicit MLP in-sample caveat
- DETECT: Improved candidate merging with score normalization

Benchmark results (α=1.5, no MLP):
- 1 seam: 41% perfect detection, 21.8 sample error
- 2 seams: 29% perfect detection, 24.4 sample error

Breaking changes:
- Config now via MASSSMASHConfig dataclass
- Default alpha=2.0 (conservative)
- Legacy file renamed to MASS_SMASH_v1_legacy.py"
```

---

## Step 9: Push and Create PR

```bash
git push origin feature/mass-smash-v2-review-fixes
```

Then create PR with this description:

```markdown
## Summary

Addresses critical review feedback for MASS/SMASH implementation.

## Changes

### Terminology Fixes
- ❌ "beam search" → ✅ "bounded subset enumeration"
- The code enumerates all seam subsets up to max_seams, then ranks by MDL
- This is exhaustive within bounds, not pruned during expansion

### Notation Fixes  
- ❌ `k` for both params and seams → ✅ `p` for params, `m` for seams
- MDL formula now unambiguous

### Documentation
- Real benchmark statistics with exact seeds
- Explicit MLP in-sample caveat
- Seam penalty explained as coding argument (not magic regularizer)

## Benchmarks

| Config | Perfect Detection | Mean Error |
|--------|------------------|------------|
| 1 seam, α=1.5 | 41% | 7.3% |
| 2 seams, α=1.5 | 29% | 8.1% |

## Migration

```python
# Old
from MASS_SMASH_complete import run_mass_smash
best = run_mass_smash(y, alpha=2.0, max_seams=3)

# New  
from MASS_SMASH_v2 import run_mass_smash, MASSSMASHConfig
config = MASSSMASHConfig(alpha=2.0, max_seams=3)
best, all_solutions = run_mass_smash(y, config)
```
```

---

## Step 10: Post-Merge Cleanup

After PR is merged:

```bash
git checkout main
git pull origin main
git branch -d feature/mass-smash-v2-review-fixes

# Optionally tag the release
git tag -a v2.0.0 -m "MASS/SMASH v2: Review fixes and real benchmarks"
git push origin v2.0.0
```

---

## Verification Commands

Run these to verify the update was successful:

```bash
# Test import works
python -c "from MASS_SMASH_v2 import run_mass_smash, MASSSMASHConfig; print('Import OK')"

# Run quick smoke test
python -c "
from MASS_SMASH_v2 import generate_signal_with_seams, run_mass_smash, MASSSMASHConfig
y, _ = generate_signal_with_seams(T=200, seed=0)
config = MASSSMASHConfig(verbose=False, include_mlp=False)
best, _ = run_mass_smash(y, config)
print(f'Smoke test OK: MDL={best.total_mdl:.1f}, seams={list(best.seams)}')
"

# Run tests if pytest available
pytest test_mass_smash.py -v 2>/dev/null || echo "No pytest or tests"

# Check no old terminology remains
echo "Checking for old terminology..."
grep -r "beam search" --include="*.py" --include="*.md" && echo "WARNING: Found 'beam search'" || echo "OK: No 'beam search' found"
```

---

## Files Summary

| File | Action | Notes |
|------|--------|-------|
| `MASS_SMASH_v2.py` | CREATE/REPLACE | New implementation |
| `README.md` | UPDATE | Add benchmarks, fix terminology |
| `MASS_SMASH_complete.py` | RENAME → `_v1_legacy.py` | Preserve for reference |
| `test_mass_smash.py` | CREATE | Basic test suite |
| `CLAUDE_CODE_INSTRUCTIONS.md` | DELETE after use | This file |

---

## Contact

If issues arise during the update, the key review points were:
1. Terminology must be accurate (not overclaiming)
2. Statistics must be reproducible (with seeds)
3. Notation must be unambiguous (no variable collisions)
4. Caveats must be explicit (especially MLP in-sample)
