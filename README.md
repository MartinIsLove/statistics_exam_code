# Random Variate Generation from the Uniform Distribution

This project accompanies a thesis on generating random variables from
standard uniform draws. NumPy supplies pseudorandom uniform inputs,
SciPy supplies the theoretical distributions for comparison, and
Matplotlib produces the figures.

## Sampling methods

| Target distribution | Sampling method |
| --- | --- |
| Exponential with rate `ln(2)` | Inverse transform sampling |
| Standard normal `N(0, 1)` | Box–Muller transform |
| `Gamma(3, 1)` | Acceptance–rejection with an exponential proposal |
| `Beta(2, 5)` | Ratio `A / (A + B)` using independent `Gamma(2, 1)` and `Gamma(5, 1)` draws |

The Gamma parameters are given in shape–rate order.

## Run the experiment

Run these commands from the directory containing `test.py`.

```bash
python -m pip install numpy scipy matplotlib
python test.py
```

The script uses NumPy's PCG64 bit generator with seed `2026` and generates
10,000 samples per target distribution. The reference run used NumPy
2.3.5 and SciPy 1.17.0.

## Outputs

- The terminal table compares theoretical and observed means, variances,
  and Gamma sampling costs. The Kolmogorov–Smirnov distance measures the
  largest vertical gap between the empirical and theoretical CDFs. The
  table expresses that distance in percentage points as `100 * D_n`.
- `ecdf_comparison.pdf` overlays the empirical and theoretical CDFs
  for all four distributions.
- `density_comparison.pdf` overlays the normalized histograms and
  theoretical densities for all four distributions.

The PDFs are saved in the directory from which the script is run.

