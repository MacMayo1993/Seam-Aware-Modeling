# Performance Optimization Results

## Executive Summary

This report presents comprehensive benchmarking results for 
performance optimizations implemented in the SeamAware library.

## 1. CUSUM Detection (Vectorized)

| Signal Size | Time (ms) | Throughput (samples/sec) |
|-------------|-----------|--------------------------|
|         100 |      0.07 |                1,493,514 |
|         500 |      0.10 |                4,836,975 |
|        1000 |      0.15 |                6,485,173 |
|        5000 |      0.94 |                5,335,715 |
|       10000 |      1.79 |                5,583,949 |

## 2. Roughness Computation (Savitzky-Golay Filter)

| Signal Size | Time (ms) | Throughput (samples/sec) |
|-------------|-----------|--------------------------|
|         100 |      0.27 |                  375,877 |
|         500 |      0.84 |                  597,522 |
|        1000 |      1.05 |                  948,381 |
|        5000 |      5.18 |                  964,862 |
|       10000 |     10.09 |                  991,381 |

## 3. MASSFramework (Detection-Guided Search)

| Signal Size | Time (ms) | Evaluations | Ops/sec |
|-------------|-----------|-------------|---------|
|         100 |      0.48 |           5 |  10,511 |
|         200 |      0.46 |           5 |  10,862 |
|         500 |      0.58 |           5 |   8,598 |
|        1000 |      0.71 |           5 |   7,047 |
|        2000 |      1.01 |           5 |   4,956 |
