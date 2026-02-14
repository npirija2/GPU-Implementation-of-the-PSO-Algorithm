# CUDA PSO – GPU Accelerated Particle Swarm Optimization

CUDA-based implementation of Particle Swarm Optimization (PSO) optimized for NVIDIA GPUs using massive parallelism. The project progresses from a sequential CPU version to highly optimized GPU kernels using Structure of Arrays (SoA), queue reductions, warp-level shuffle, and bit-packing techniques.

## Features

- Sequential CPU PSO baseline
- Parallel CUDA PSO implementations
- Ring (Lbest) topology
- Queue and Queue-Lock reductions
- Warp-level optimized cuPSO
- High-dimensional benchmarks (up to 120D)

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit
- C++ compiler (nvcc, g++)

## Build

```bash
nvcc -O3 main.cu -o pso


