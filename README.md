# FACTOR
The repository contains the core code to reproduce the paper "FACTOR: Fairness-Aligned Conformal Transport for Optimal Regions". FACTOR is optimal transport-based conformal inference for multivariate outcomes with fairness contraints. The outcomes can be continuous, discrete or mixed and the training of optimal transport is mostly adapted from [multi-output-conformal-regression](https://github.com/Vekteur/multi-output-conformal-regression). 

## Benchmark analysis
- `sim_moc.py`: benchmark analysis on synthetic data, used to validate FACTOR under controlled settings and known data-generating mechanisms.
- `real_moc.py`: benchmark analysis on real datasets with continuous outcomes, demonstrating empirical performance and fairness properties in practical regression tasks.
- `real_moc_discrete.py`: benchmark analysis on real datasets with mixed-type outcomes (continuous and discrete/categorical), showcasing FACTOR’s flexibility beyond purely continuous responses.

## Supporting utility
- `utils_moc.py`: helper functions for computation and plotting, used to evaluate coverage, fairness metrics, and visualize multivariate prediction sets.
- `utils_data.py`: routines for data generation, including toy examples and data preparation for different training pipelines.
- `utils_conformalizer.py`: a collection of candidate conformalizers and baselines against which FACTOR is benchmarked, enabling systematic and reproducible comparisons.
