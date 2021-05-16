# kmeanspp

Implements the kmeans++ algorithm for generating a good initial codebook for the kmeans iterations.

## Required python packages
- NumPy
- CuPy for running on NVIDIA GPU
- Sphinx for generating documentation

## Benchmarks
The CuPy version runs, for 10 000 000 training vectors and codebook size of 1000, in about 466 seconds.

The NumPy version runs, on a single thread, for the same problem size, in about 789 seconds.

## Documentation
Full documentation of the functions provided in the module is available https://mkurop.github.io/kmeanspp
