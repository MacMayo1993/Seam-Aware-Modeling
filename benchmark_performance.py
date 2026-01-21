#!/usr/bin/env python3
"""
Performance Benchmark Suite for SeamAware Optimizations

This script comprehensively tests all performance optimizations implemented
in the SeamAware codebase, comparing optimized vs baseline implementations.
"""

import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# Add examples to path for mass_smash
sys.path.insert(0, "examples")

from seamaware import MASSFramework
from seamaware.core.detection import detect_seam
from seamaware.core.seam_detection import compute_roughness

from mass_smash import MASSSMASHConfig, run_mass_smash


class BenchmarkResult:
    """Store benchmark results for a single test."""

    def __init__(
        self, name: str, size: int, time_ms: float, operations: int = None
    ):
        self.name = name
        self.size = size
        self.time_ms = time_ms
        self.operations = operations

    def throughput(self) -> float:
        """Operations per second."""
        if self.operations:
            return (self.operations / self.time_ms) * 1000
        return (self.size / self.time_ms) * 1000

    def __str__(self):
        if self.operations:
            return (
                f"{self.name}: {self.time_ms:.2f}ms "
                f"({self.throughput():.0f} ops/sec, n={self.size})"
            )
        return (
            f"{self.name}: {self.time_ms:.2f}ms "
            f"({self.throughput():.0f} samples/sec, n={self.size})"
        )


def generate_test_signal(n: int, seam_pos: int = None, noise: float = 0.1):
    """Generate synthetic test signal with optional seam."""
    np.random.seed(42)
    signal = np.sin(np.linspace(0, 4 * np.pi, n))

    if seam_pos:
        signal[seam_pos:] *= -1

    signal += np.random.normal(0, noise, n)
    return signal


def benchmark_cusum_detection() -> List[BenchmarkResult]:
    """Benchmark CUSUM detection at various scales."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: CUSUM Detection (Vectorized)")
    print("=" * 70)

    results = []
    sizes = [100, 500, 1000, 5000, 10000]

    for n in sizes:
        signal = generate_test_signal(n, seam_pos=n // 2)

        # Warmup
        _ = detect_seam(signal, method="cusum")

        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = detect_seam(signal, method="cusum")
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        bench = BenchmarkResult("CUSUM Detection", n, avg_time)
        results.append(bench)

        print(f"n={n:>5}: {avg_time:.2f} ± {std_time:.2f} ms "
              f"({bench.throughput():.0f} samples/sec)")

    return results


def benchmark_roughness_computation() -> List[BenchmarkResult]:
    """Benchmark roughness computation with Savitzky-Golay filter."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Roughness Computation (Savitzky-Golay)")
    print("=" * 70)

    results = []
    sizes = [100, 500, 1000, 5000, 10000]

    for n in sizes:
        signal = generate_test_signal(n, seam_pos=n // 2)

        # Warmup
        _ = compute_roughness(signal, window=20, poly_degree=1)

        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            roughness = compute_roughness(signal, window=20, poly_degree=1)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        bench = BenchmarkResult("Roughness Computation", n, avg_time)
        results.append(bench)

        print(f"n={n:>5}: {avg_time:.2f} ± {std_time:.2f} ms "
              f"({bench.throughput():.0f} samples/sec)")

    return results


def benchmark_mass_framework() -> List[BenchmarkResult]:
    """Benchmark MASSFramework with detection candidates."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: MASSFramework (Detection-Guided Search)")
    print("=" * 70)

    results = []
    sizes = [100, 200, 500, 1000, 2000]

    for n in sizes:
        signal = generate_test_signal(n, seam_pos=n // 2, noise=0.15)

        mass = MASSFramework(
            baseline="fourier",
            baseline_params={"K": 3},
            detection_method="cusum",
            atoms=["sign_flip"],
            min_confidence=0.1,
        )

        # Warmup
        _ = mass.fit(signal)

        # Benchmark
        times = []
        for _ in range(5):
            start = time.perf_counter()
            result = mass.fit(signal)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        # Count candidate evaluations (atoms × candidates)
        detection = detect_seam(signal, method="cusum")
        n_candidates = len(detection.all_candidates)
        n_evaluations = n_candidates * len(mass.atoms)

        bench = BenchmarkResult(
            "MASSFramework", n, avg_time, operations=n_evaluations
        )
        results.append(bench)

        print(f"n={n:>4}: {avg_time:.2f} ± {std_time:.2f} ms "
              f"({n_evaluations} evals, {bench.throughput():.0f} ops/sec)")

    return results


def benchmark_mass_smash_beam_search() -> List[BenchmarkResult]:
    """Benchmark MASS/SMASH with beam search."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: MASS/SMASH Beam Search")
    print("=" * 70)

    results = []
    test_configs = [
        (200, 3, 3),  # (signal_size, max_seams, candidates)
        (300, 3, 5),
        (500, 3, 5),
        (300, 4, 8),
    ]

    for n, max_seams, n_candidates in test_configs:
        signal = generate_test_signal(n, seam_pos=n // 2, noise=0.15)

        # Beam search config
        config_beam = MASSSMASHConfig(
            top_k_candidates=n_candidates,
            max_seams=max_seams,
            use_beam_search=True,
            beam_width=5,
            verbose=False,
            include_mlp=False,
        )

        # Exhaustive search config
        config_exhaustive = MASSSMASHConfig(
            top_k_candidates=n_candidates,
            max_seams=max_seams,
            use_beam_search=False,
            verbose=False,
            include_mlp=False,
        )

        # Warmup
        _ = run_mass_smash(signal, config_beam)

        # Benchmark beam search
        times_beam = []
        for _ in range(3):
            start = time.perf_counter()
            best_beam, all_beam = run_mass_smash(signal, config_beam)
            elapsed = (time.perf_counter() - start) * 1000
            times_beam.append(elapsed)

        # Benchmark exhaustive
        times_exhaustive = []
        for _ in range(3):
            start = time.perf_counter()
            best_exh, all_exh = run_mass_smash(signal, config_exhaustive)
            elapsed = (time.perf_counter() - start) * 1000
            times_exhaustive.append(elapsed)

        avg_beam = np.mean(times_beam)
        avg_exh = np.mean(times_exhaustive)
        speedup = avg_exh / avg_beam

        bench_beam = BenchmarkResult(
            f"Beam Search (k={n_candidates}, m={max_seams})",
            n,
            avg_beam,
            operations=len(all_beam),
        )
        bench_exh = BenchmarkResult(
            f"Exhaustive (k={n_candidates}, m={max_seams})",
            n,
            avg_exh,
            operations=len(all_exh),
        )

        results.extend([bench_beam, bench_exh])

        print(f"\nn={n}, max_seams={max_seams}, candidates={n_candidates}")
        print(f"  Beam:       {avg_beam:.2f}ms ({len(all_beam)} configs)")
        print(f"  Exhaustive: {avg_exh:.2f}ms ({len(all_exh)} configs)")
        print(f"  Speedup:    {speedup:.1f}×")
        print(f"  Same result: {best_beam.total_mdl == best_exh.total_mdl}")

    return results


def benchmark_scalability() -> Dict[str, List[Tuple[int, float]]]:
    """Test scalability of optimizations."""
    print("\n" + "=" * 70)
    print("BENCHMARK 5: Scalability Analysis")
    print("=" * 70)

    sizes = [100, 500, 1000, 5000, 10000, 20000]
    results = {
        "cusum": [],
        "roughness": [],
        "mass_framework": [],
    }

    for n in sizes:
        signal = generate_test_signal(n, seam_pos=n // 2)

        # CUSUM
        start = time.perf_counter()
        _ = detect_seam(signal, method="cusum")
        cusum_time = (time.perf_counter() - start) * 1000
        results["cusum"].append((n, cusum_time))

        # Roughness
        start = time.perf_counter()
        _ = compute_roughness(signal, window=20)
        rough_time = (time.perf_counter() - start) * 1000
        results["roughness"].append((n, rough_time))

        # MASSFramework (only up to 5000 to keep runtime reasonable)
        if n <= 5000:
            mass = MASSFramework(
                baseline="fourier",
                baseline_params={"K": 3},
                detection_method="cusum",
                atoms=["sign_flip"],
            )
            start = time.perf_counter()
            _ = mass.fit(signal)
            mass_time = (time.perf_counter() - start) * 1000
            results["mass_framework"].append((n, mass_time))

        print(f"n={n:>5}: CUSUM={cusum_time:.2f}ms, "
              f"Roughness={rough_time:.2f}ms" +
              (f", MASS={mass_time:.2f}ms" if n <= 5000 else ""))

    return results


def generate_performance_report(all_results: Dict):
    """Generate markdown performance report."""
    print("\n" + "=" * 70)
    print("GENERATING PERFORMANCE REPORT")
    print("=" * 70)

    report = []
    report.append("# Performance Optimization Results")
    report.append("")
    report.append("## Executive Summary")
    report.append("")
    report.append("This report presents comprehensive benchmarking results for ")
    report.append("performance optimizations implemented in the SeamAware library.")
    report.append("")

    # CUSUM Detection
    cusum_results = all_results["cusum"]
    report.append("## 1. CUSUM Detection (Vectorized)")
    report.append("")
    report.append("| Signal Size | Time (ms) | Throughput (samples/sec) |")
    report.append("|-------------|-----------|--------------------------|")
    for result in cusum_results:
        report.append(
            f"| {result.size:>11} | {result.time_ms:>9.2f} | "
            f"{result.throughput():>24,.0f} |"
        )
    report.append("")

    # Roughness
    rough_results = all_results["roughness"]
    report.append("## 2. Roughness Computation (Savitzky-Golay Filter)")
    report.append("")
    report.append("| Signal Size | Time (ms) | Throughput (samples/sec) |")
    report.append("|-------------|-----------|--------------------------|")
    for result in rough_results:
        report.append(
            f"| {result.size:>11} | {result.time_ms:>9.2f} | "
            f"{result.throughput():>24,.0f} |"
        )
    report.append("")

    # MASSFramework
    mass_results = all_results["mass"]
    report.append("## 3. MASSFramework (Detection-Guided Search)")
    report.append("")
    report.append("| Signal Size | Time (ms) | Evaluations | Ops/sec |")
    report.append("|-------------|-----------|-------------|---------|")
    for result in mass_results:
        report.append(
            f"| {result.size:>11} | {result.time_ms:>9.2f} | "
            f"{result.operations:>11} | {result.throughput():>7,.0f} |"
        )
    report.append("")

    # Save report
    report_text = "\n".join(report)
    with open("PERFORMANCE_RESULTS.md", "w") as f:
        f.write(report_text)

    print("✓ Report saved to PERFORMANCE_RESULTS.md")


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("SeamAware Performance Benchmark Suite")
    print("=" * 70)
    print("Testing all critical and high-priority optimizations")
    print()

    all_results = {}

    # Run benchmarks
    all_results["cusum"] = benchmark_cusum_detection()
    all_results["roughness"] = benchmark_roughness_computation()
    all_results["mass"] = benchmark_mass_framework()
    all_results["mass_smash"] = benchmark_mass_smash_beam_search()
    all_results["scalability"] = benchmark_scalability()

    # Generate report
    generate_performance_report(all_results)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
