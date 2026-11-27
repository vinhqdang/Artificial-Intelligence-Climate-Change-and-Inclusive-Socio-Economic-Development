# Artificial-Intelligence-Climate-Change-and-Inclusive-Socio-Economic-Development
https://www.emerald.com/ijse/calls-for-submissions/2192/Artificial-Intelligence-Climate-Change-and

## Reproduce the agricultural exposure comparison

From the repository root, run the following commands with the Python environment for
the agricultural analysis. The comparison uses the committed processed panel.

```bash
python code/31_application_agri.py --exposure-shares 0.3
python code/31_application_agri.py --exposure-shares 0.2
python code/31_application_agri.py --exposure-shares 0.5
python code/31_application_agri.py --exposure-shares 0.3 --equal-country
```

The commands write CSV to standard output. The first run reproduces the exposure-coverage
table in the manuscript. The remaining runs check alternative budgets and equal country
weights. The share for random selection equals the number selected divided by the number
of eligible countries.

## Benchmark the robust allocation algorithm

The new benchmark compares quota-preserving swaps from an exposure ranking. It uses
independent simulated test countries and a nested evaluation on the processed
agricultural panel, with country and region holdouts.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python code/40_robust_allocation.py --reps 500 --jobs 4
```

The simulation summaries, paired Monte Carlo comparisons, agricultural estimates and
held-out rules and manuscript tables are written to `results/robust_allocation/`. The `--mode simulation` and
`--mode empirical` switches run the two analyses separately. Use `--output-dir` to
write a replication run to a separate directory.

## Benchmark the ellipsoidal portfolio optimiser

Run the portfolio comparison with the same Python environment and processed panel.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python code/41_ellipsoidal_allocation.py --reps 200 --jobs 4
```

The command writes simulation summaries, paired comparisons, held-out estimates and
per-fold optimisation certificates. The `--mode` switch selects one comparison.
The simulation records lower–upper gaps for runs stopped before the solver tolerance.

## Reproduce the exchange-grid solver benchmark

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python code/42_exchange_grid_benchmark.py --reps 16
```

This runs 48 synthetic problems and 12 fitted agricultural objectives. It writes
feasible values, certified upper bounds, gaps and timings to
`results/exchange_grid/`. The exchange-grid optimiser does not call a mixed-integer
solver. Integer cutting planes are run separately as a benchmark.

## Rebuild the manuscript figures

```bash
python code/43_manuscript_figures.py
bash paper/build.sh
```

The figure script reads the processed agricultural panel and the saved simulation,
holdout and solver results. For the maps it retrieves Natural Earth 1:50m country
boundaries (v5.1.2) and checks the SHA-256 before using or caching them under
`data/raw/natural_earth/`. To use a local copy of the same file, pass
`--boundaries /path/to/ne_50m_admin_0_countries.geojson`.
