"""Evaluate covariance-aware integer portfolios on independent test countries."""

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ewdrpl.ellipsoid import moderation_surface, robust_portfolio
from ewdrpl.heldout import SCENARIOS, score_holdout, score_training
from ewdrpl.inference import cluster_bootstrap
from ewdrpl.nuisance import make_folds

benchmark = import_module("40_robust_allocation")
RADIUS = 1.645
METHODS = ("Exposure baseline", "Robust swaps", "Pooled mean", "Ellipsoidal")
OUT = benchmark.ROOT / "results" / "ellipsoidal_allocation"


def policies(z_train, exposure_train, scores, z_test, exposure_test, quota, fast):
    rules, choices = benchmark.fit_rules(z_train, exposure_train, (exposure_train[:, None] * scores).T)
    reference = rules[0].assign(z_test, exposure_test, quota)
    swaps = rules[choices["Robust swaps"]].assign(z_test, exposure_test, quota)
    slope, factor = moderation_surface(z_train, scores, z_test)
    mean = exposure_test * slope
    loadings = exposure_test[:, None] * factor
    neutral = robust_portfolio(mean, loadings, reference, 0)
    settings = {"rounds": 12, "seconds": 0.15} if fast else {"rounds": 40, "seconds": 1.0}
    robust = robust_portfolio(mean, loadings, reference, RADIUS, **settings)
    masks = np.array([reference, swaps, neutral.selected, robust.selected])
    assert np.all(masks.sum(axis=1) == quota)
    return masks, robust


def simulated_run(design, overlap, heterogeneous, rep):
    rng = np.random.default_rng(410_000 + rep)
    train, z_train, exposure_train, _ = benchmark.simulate(150, rng, overlap, heterogeneous)
    _, z_test, exposure_test, slope_test = benchmark.simulate(60, rng, overlap, heterogeneous)
    scores = score_training(train, *benchmark.learners(), seed=42).reshape(
        len(z_train), 6, len(SCENARIOS),
    ).mean(axis=1)
    quota = int(benchmark.BUDGET * len(z_test))
    masks, certificate = policies(z_train, exposure_train, scores, z_test, exposure_test, quota, fast=True)
    true_values = exposure_test * slope_test
    values = masks @ true_values / len(z_test)
    oracle = np.sort(true_values)[-quota:].sum() / len(z_test)
    return [{
        "design": design, "rep": rep, "method": method, "gain": value - values[0],
        "regret": oracle - value, "loss": value < values[0] - 1e-12,
        "changed": (mask & ~masks[0]).sum(),
        "solver_gap": certificate.gap / len(z_test) if method == "Ellipsoidal" else np.nan,
        "closed_gap": certificate.termination == "gap tolerance" if method == "Ellipsoidal" else np.nan,
    } for method, value, mask in zip(METHODS, values, masks)]


def simulation(reps, jobs, out):
    batches = Parallel(n_jobs=jobs, verbose=5)(
        delayed(simulated_run)(design, overlap, heterogeneous, rep)
        for design, (overlap, heterogeneous) in benchmark.DGP.items() for rep in range(reps)
    )
    rows = pd.DataFrame([row for batch in batches for row in batch])
    summary = rows.groupby(["design", "method"], sort=False).agg(
        gain=("gain", "mean"), regret=("regret", "mean"), loss_rate=("loss", "mean"),
        changed=("changed", "mean"), mean_gap=("solver_gap", "mean"),
        max_gap=("solver_gap", "max"), closed_rate=("closed_gap", "mean"), reps=("rep", "count"),
    ).reset_index()
    summary.to_csv(out / "simulation.csv", index=False, float_format="%.8f")
    pivot = rows.pivot(index=["design", "rep"], columns="method", values="gain")
    comparisons = []
    for design, gains in pivot.groupby(level="design", sort=False):
        for comparator in METHODS[:-1]:
            paired = gains["Ellipsoidal"] - gains[comparator]
            comparisons.append({
                "design": design, "comparator": comparator, "difference": paired.mean(),
                "mc_se": paired.std(ddof=1) / np.sqrt(len(paired)), "reps": len(paired),
            })
    pd.DataFrame(comparisons).to_csv(out / "paired.csv", index=False, float_format="%.8f")
    print(summary.to_string(index=False), flush=True)


def empirical_fold(data, z, exposure, ids, folds, row_folds, fold, quota):
    test = folds == fold
    train_data = data.subset(row_folds != fold)
    test_data = data.subset(np.isin(data.country, ids[test]))
    learners = benchmark.learners()
    train_score = benchmark.country_scores(
        train_data, score_training(train_data, *learners, seed=420 + fold), ids[~test],
    )
    test_score = benchmark.country_scores(
        test_data, score_holdout(train_data, test_data, *learners, seed=520 + fold), ids[test],
    )
    masks, certificate = policies(z[~test], exposure[~test], train_score, z[test], exposure[test], quota, fast=False)
    return ids[test], exposure[test, None] * test_score[:, :2], masks, {
        "fold": fold, "countries": int(test.sum()), "quota": quota, "iterations": certificate.iterations,
        "lower_bound": certificate.value / test.sum(), "upper_bound": certificate.upper_bound / test.sum(),
        "solver_gap": certificate.gap / test.sum(), "termination": certificate.termination,
    }


def empirical(jobs, out):
    data, countries, z, exposure, row_regions = benchmark.agri_frame()
    ids = countries.index.to_numpy()
    country_folds = make_folds(ids, K=5, seed=42)
    row_country = pd.Series(country_folds, index=ids).reindex(data.country, fill_value=-1).to_numpy()
    regions, region_folds = np.unique(countries["region"], return_inverse=True)
    row_region = pd.Series(np.arange(len(regions)), index=regions).reindex(row_regions, fill_value=-1).to_numpy()
    results, bounds = [], []
    for design, folds, row_folds in (
        ("Country folds", country_folds, row_country), ("Region folds", region_folds, row_region),
    ):
        sizes = np.bincount(folds)
        quotas = np.diff(np.floor(benchmark.BUDGET * np.r_[0, sizes.cumsum()]).astype(int))
        batches = Parallel(n_jobs=jobs, verbose=5)(
            delayed(empirical_fold)(data, z, exposure, ids, folds, row_folds, fold, quota)
            for fold, quota in enumerate(quotas)
        )
        test_ids, scores, assignments, certificates = zip(*batches)
        cid, values = np.concatenate(test_ids), np.concatenate(scores)
        masks = np.concatenate(assignments, axis=1)
        assert len(np.unique(cid)) == len(ids)
        assert np.all(masks.sum(axis=1) == int(benchmark.BUDGET * len(ids)))
        bounds.extend(dict(row, design=design) for row in certificates)
        for method, mask in zip(METHODS, masks):
            for label, welfare in zip(SCENARIOS[:2], values.T):
                paired = (mask.astype(float) - masks[0]) * welfare
                stats = cluster_bootstrap(paired, cid, n_boot=4000, seed=42)
                results.append({
                    "design": design, "method": method, "evaluation": label,
                    "gain": stats["estimate"], "lower": stats["ci"][0], "upper": stats["ci"][1],
                    "selected": mask.sum(), "changed": (mask & ~masks[0]).sum(),
                })
    application = pd.DataFrame(results)
    application.to_csv(out / "application.csv", index=False, float_format="%.8f")
    pd.DataFrame(bounds).to_csv(out / "certificates.csv", index=False, float_format="%.10g")
    print(application.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=200, help="replications per design")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--mode", choices=("both", "simulation", "empirical"), default="both")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.reps < 2 or args.jobs < 1:
        parser.error("use at least two replications and one job")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("simulation", "both"):
        simulation(args.reps, args.jobs, args.output_dir)
    if args.mode in ("empirical", "both"):
        empirical(args.jobs, args.output_dir)


if __name__ == "__main__":
    main()
