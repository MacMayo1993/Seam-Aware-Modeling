"""
Performance tests for seam detection.

These tests verify that optimizations maintain O(n) complexity.
Run with: pytest tests/test_performance.py -v -s
Skip slow tests: pytest -m "not slow"
"""
import time

import numpy as np
import pytest

from seamaware.core.detection import detect_seam_cusum


def benchmark_detection(signal_length: int, num_trials: int = 3) -> tuple:
    """
    Benchmark detection on signal of given length.

    Parameters
    ----------
    signal_length : int
        Length of signal to generate
    num_trials : int
        Number of trials to average over

    Returns
    -------
    tuple
        (result, avg_time_seconds)
    """
    np.random.seed(42)

    # Create signal with seam at midpoint
    signal = np.concatenate([
        np.sin(np.linspace(0, 4 * np.pi, signal_length // 2)),
        -np.sin(np.linspace(0, 4 * np.pi, signal_length // 2))  # Sign flip
    ])
    signal += 0.1 * np.random.randn(signal_length)

    times = []
    result = None
    for _ in range(num_trials):
        start = time.perf_counter()
        result = detect_seam_cusum(signal)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    return result, avg_time


class TestDetectionPerformance:
    """Test suite for seam detection performance."""

    def test_correctness_known_seam(self):
        """Verify detection finds seams at known positions."""
        # Known seam at position 500
        signal = np.concatenate([np.ones(500), -np.ones(500)])
        signal += 0.05 * np.random.randn(1000)

        result = detect_seam_cusum(signal, min_segment_length=10)

        # Should find seam near position 500
        error = abs(result.position - 500)
        assert error < 50, f"Detection error too large: {error} samples from true seam"
        assert result.confidence > 0.5, f"Low confidence: {result.confidence}"

    @pytest.mark.slow
    def test_scales_linearly_small(self):
        """Verify O(n) scaling on small to medium signals."""
        # Test on 1k, 5k, 10k samples
        results = []
        for n in [1000, 5000, 10000]:
            _, time_sec = benchmark_detection(n, num_trials=3)
            time_ms = time_sec * 1000
            results.append((n, time_ms))

            # Each should complete in reasonable time
            assert time_ms < 100, f"Signal length {n} took {time_ms:.1f}ms (should be <100ms)"

        # Verify approximately linear scaling
        # time(5k) / time(1k) should be ~5× (not ~25× for O(n²))
        _, t1k = results[0]
        _, t5k = results[1]
        ratio = t5k / t1k if t1k > 0 else float('inf')

        # Allow some overhead, but ratio should be closer to 5 than 25
        assert ratio < 15, f"Scaling ratio {ratio:.1f}× suggests worse than O(n) complexity"

    @pytest.mark.slow
    def test_scales_linearly_large(self):
        """Verify O(n) scaling on large signals (50k samples)."""
        n = 50000
        result, time_sec = benchmark_detection(n, num_trials=3)
        time_ms = time_sec * 1000

        # Old O(n²) implementation would take ~25 seconds
        # New O(n) should take <1 second
        assert time_ms < 1000, (
            f"Large signal (n={n}) took {time_ms:.0f}ms (should be <1000ms). "
            "This suggests O(n²) complexity instead of O(n)."
        )

        # Should still find seam correctly
        assert result.position > 0
        assert result.confidence > 0

    def test_minimum_segment_enforcement(self):
        """Verify min_segment_length is respected."""
        signal = np.concatenate([np.ones(100), -np.ones(100)])
        min_seg = 20

        result = detect_seam_cusum(signal, min_segment_length=min_seg)

        # Detected position should respect boundaries
        assert result.position >= min_seg
        assert result.position <= len(signal) - min_seg

    @pytest.mark.slow
    def test_multiple_trials_consistent(self):
        """Verify detection is deterministic across runs."""
        signal = np.random.randn(1000)
        signal[500:] *= -1  # Clear seam at 500

        positions = []
        for _ in range(5):
            result = detect_seam_cusum(signal, min_segment_length=10)
            positions.append(result.position)

        # All runs should find same position (deterministic)
        assert len(set(positions)) == 1, f"Non-deterministic results: {positions}"

    def test_min_segment_length_validation(self):
        """Verify min_segment_length is validated."""
        signal = np.ones(100)

        # min_segment_length < 1 should raise
        with pytest.raises(ValueError, match="min_segment_length must be >= 1"):
            detect_seam_cusum(signal, min_segment_length=0)

        with pytest.raises(ValueError, match="min_segment_length must be >= 1"):
            detect_seam_cusum(signal, min_segment_length=-1)

        # min_segment_length = 1 should work
        result = detect_seam_cusum(signal, min_segment_length=1)
        assert result.position > 0
