"""Compare exchange-grid certificates against generic integer cutting planes."""

import argparse
from importlib import import_module
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ewdrpl.ellipsoid import moderation_surface, robust_portfolio
from ewdrpl.exchange_grid import exchange_grid
from ewdrpl.heldout import score_training
from ewdrpl.nuisance import make_folds

benchmark = import_module("40_robust_allocation")
OUT = benchmark.ROOT / "results" / "exchange_grid"
RADIUS = 1.645


def compare(mean, factor, quota, priority, **metadata):
    reference = np.zeros(len(mean), dtype=bool)
    reference[np.argsort(-priority, kind="stable")[:quota]] = True
    methods = [
        ("Exchange grid", lambda: exchange_grid(mean, factor, reference, RADIUS)),
        ("Grid without tilt", lambda: exchange_grid(mean, factor, reference, RADIUS, tilt_updates=False)),
        ("Integer cutting planes", lambda: robust_portfolio(mean, factor, reference, RADIUS, rounds=30, seconds=0.3)),
    ]
    rows = []
    for method, solve in methods:
        start = perf_counter()
        result = solve()
        elapsed = perf_counter() - start
        assert result.selected.sum() == quota
        delta = result.selected.astype(float) - reference
        primal = mean @ delta - RADIUS * np.linalg.norm(delta @ factor)
        np.testing.assert_allclose(primal, result.value, atol=1e-8)
        rows.append({
            **metadata, "countries": len(mean), "quota": quota, "method": method,
            "lower": result.value, "upper": result.upper_bound, "gap": result.gap,
            "certified": result.gap <= 1e-5, "seconds": elapsed, "steps": result.iterations,
        })
    best = max(row["lower"] for row in rows)
    assert all(row["upper"] + 1e-6 >= best for row in rows)
    return rows


def synthetic(reps):
    rows = []
    designs = (
        ("One-factor lattice", 64, 1, True),
        ("Two-factor lattice", 40, 2, True),
        ("Six continuous factors", 32, 6, False),
    )
    for label, n, rank, discrete in designs:
        for rep in range(reps):
            rng = np.random.default_rng(6300 + rep)
            mean = rng.normal(size=n)
            priority = rng.uniform(size=n)
            if discrete:
                factor = rng.integers(-3, 4, size=(n, rank)).astype(float)
                factor[0, :], factor[1, :] = -3, 3  # common grid span across designs
            else:
                factor = rng.normal(scale=0.7, size=(n, rank))
            rows.extend(compare(mean, factor, int(0.3 * n), priority, design=label, rep=rep))
    return rows


def agricultural():
    data, countries, z, exposure, row_regions = benchmark.agri_frame()
    ids = countries.index.to_numpy()
    country_folds = make_folds(ids, K=5, seed=42)
    row_country = pd.Series(country_folds, index=ids).reindex(data.country, fill_value=-1).to_numpy()
    regions, region_folds = np.unique(countries["region"].to_numpy(), return_inverse=True)
    row_region = pd.Series(np.arange(len(regions)), index=regions).reindex(row_regions, fill_value=-1).to_numpy()
    rows = []
    for label, groups, row_groups in (
        ("Agriculture country folds", country_folds, row_country),
        ("Agriculture region folds", region_folds, row_region),
    ):
        quotas = np.diff(np.floor(benchmark.BUDGET * np.r_[0, np.bincount(groups).cumsum()]).astype(int))
        for fold, quota in enumerate(quotas):
            test = groups == fold
            train_data = data.subset(row_groups != fold)
            scores = benchmark.country_scores(
                train_data, score_training(train_data, *benchmark.learners(), seed=420 + fold), ids[~test],
            )
            slopes, factor = moderation_surface(z[~test], scores, z[test])
            rows.extend(compare(exposure[test] * slopes, exposure[test, None] * factor,
                                quota, exposure[test], design=label, rep=fold))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=16, help="runs per synthetic design")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    opts = parser.parse_args()
    if opts.reps < 1:
        parser.error("use at least one replication")
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    timings = pd.DataFrame(synthetic(opts.reps) + agricultural())
    timings.to_csv(opts.output_dir / "benchmarks.csv", index=False, float_format="%.10g")
    summary = timings.groupby(["design", "method"], sort=False).agg(
        problems=("rep", "count"), certified=("certified", "sum"), median_gap=("gap", "median"),
        maximum_gap=("gap", "max"), mean_lower=("lower", "mean"), median_seconds=("seconds", "median"),
    ).reset_index()
    summary.to_csv(opts.output_dir / "summary.csv", index=False, float_format="%.10g")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
