"""Benchmark robust budget-neutral swaps in simulation and held-out countries."""

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ewdrpl.heldout import ModerationData, SCENARIOS, score_holdout, score_training
from ewdrpl.inference import cluster_bootstrap
from ewdrpl.nuisance import make_folds
from ewdrpl.swaps import assignment_matrix, robust_choice, scenario_gains, swap_candidates
from ewdrpl.welfare import atkinson_weights, benefit

ROOT = Path(__file__).resolve().parent.parent
BUDGET = 0.30
METHODS = ("Exposure baseline", "DR swaps", "Plug-in swaps", "Mean-score swaps", "Robust swaps")
POLICY_FEATURES = ("ln_gdppc", "gini", "internet", "agri_share", "base_temp", "sov")
DGP = {
    "Moderate overlap": (1.0, True),
    "Weak overlap": (3.0, True),
    "Homogeneous effects": (1.0, False),
}


def learners():
    settings = {"max_iter": 50, "max_depth": 3, "learning_rate": 0.08, "min_samples_leaf": 15}
    return HistGradientBoostingRegressor(**settings), HistGradientBoostingClassifier(**settings)


def fit_rules(z, priority, training_benefits):
    rules = swap_candidates(z)
    assignments = assignment_matrix(rules, z, priority, int(BUDGET * len(priority)))
    gains = scenario_gains(assignments, training_benefits)
    best_dr = np.argmax(gains[:, 0])
    best_plugin = np.argmax(gains[:, -1])
    best_mean = np.argmax(gains.mean(axis=1))
    best_robust = robust_choice(gains)
    return rules, dict(zip(METHODS, (0, best_dr, best_plugin, best_mean, best_robust)))


def simulate(n, rng, overlap, heterogeneous):
    z = rng.normal(size=(n, 4))
    income, internet, agriculture, climate = z.T
    # High exposure is correlated with poverty; treatment responds to development.
    incomes = np.exp(0.7 * income[:, None] + np.linspace(-1.0, 1.0, 10))
    country = np.repeat(np.arange(n), 6)
    x = z[country]
    sigma = np.exp(0.3 * agriculture + 0.15 * climate)
    exposed = 1 / (1 + np.exp(income - agriculture))
    coverage = benefit(exposed[:, None] * np.ones((1, 10)),
                       atkinson_weights(incomes, 1.0), sigma)
    propensity = 1 / (1 + np.exp(-overlap * (income + 0.7 * internet)))
    treatment = rng.binomial(1, propensity[country])
    shock = rng.normal(0.20 * x[:, 3], sigma[country])
    if heterogeneous:
        rho = 0.10 + 0.50 * (internet > 0.0) - 0.35 * ((climate > 0.5) & (agriculture > 0))
    else:
        rho = np.full(n, 0.30)
    slope = -0.9 - 0.35 * agriculture + 0.25 * climate
    outcome = ((slope[country] + treatment * rho[country]) * shock
               + 0.4 * np.sin(x[:, 0]) + 0.35 * treatment + rng.normal(0, 0.65, len(shock)))
    data = ModerationData(x, shock, treatment, outcome, country)
    return data, z, coverage, rho


def simulation_rep(design, overlap, heterogeneous, rep):
    rng = np.random.default_rng(410_000 + rep)
    data, z_train, exposure_train, _ = simulate(150, rng, overlap, heterogeneous)
    _, z_test, exposure_test, rho_test = simulate(300, rng, overlap, heterogeneous)
    training_scores = score_training(data, *learners(), seed=42)
    b_train = training_scores.reshape(len(z_train), 6, len(SCENARIOS)).mean(axis=1)
    rules, choices = fit_rules(z_train, exposure_train, (exposure_train[:, None] * b_train).T)
    assignments = assignment_matrix(rules, z_test, exposure_test, int(BUDGET * len(z_test)))
    gains = scenario_gains(assignments, exposure_test * rho_test)[:, 0]
    best = gains.max()
    return [{
        "design": design, "rep": rep, "method": method, "gain": gains[i],
        "regret": best - gains[i], "loss": gains[i] < -1e-12, "fallback": i == 0,
    } for method, i in choices.items()]


def simulate_many(reps, jobs):
    runs = Parallel(n_jobs=jobs, verbose=5)(
        delayed(simulation_rep)(design, overlap, heterogeneous, rep)
        for design, (overlap, heterogeneous) in DGP.items()
        for rep in range(reps)
    )
    raw = pd.DataFrame([row for run in runs for row in run])
    summary = (raw.groupby(["design", "method"], sort=False).agg(
        gain=("gain", "mean"), gain_sd=("gain", "std"), regret=("regret", "mean"),
        loss_rate=("loss", "mean"), fallback_rate=("fallback", "mean"), reps=("rep", "count"),
    ).reset_index())
    summary["gain_mc_se"] = summary.pop("gain_sd") / np.sqrt(summary["reps"])
    pairs = raw.pivot(index=["design", "rep"], columns="method", values="gain")
    comparisons = []
    for design, benefits in pairs.groupby(level="design"):
        for method in METHODS[1:-1]:
            difference = benefits["Robust swaps"] - benefits[method]
            comparisons.append({
                "design": design, "comparator": method, "difference": difference.mean(),
                "mc_se": difference.std(ddof=1) / np.sqrt(len(difference)), "reps": len(difference),
            })
    return summary, pd.DataFrame(comparisons)


def agri_frame():
    app = import_module("31_application_agri")
    panel = app.load_panel()
    frame, regions = app.make_frame(panel)
    countries = app.policy_country_frame(frame, panel)
    data = ModerationData(
        frame[app.xlist(regions)].to_numpy(float), frame["S_spell"].to_numpy(float),
        frame["T"].to_numpy(float), frame["Y"].to_numpy(float), frame["iso3"].to_numpy(),
    )
    z = countries[list(POLICY_FEATURES)].to_numpy(float)
    w = countries["pop"].to_numpy(float) / countries["pop"].sum()
    omega = atkinson_weights(countries[app.DEC].to_numpy(float), 1.0, w)
    e = np.clip(10 * countries["agri"].to_numpy(float)[:, None] - np.arange(10), 0, 1)
    exposure = benefit(e, omega, countries["s_bar"].to_numpy(float), w)
    return data, countries, z, exposure, frame["region"].to_numpy()


def country_scores(data, scores, ids):
    return pd.DataFrame(scores, index=data.country).groupby(level=0).mean().loc[ids].to_numpy()


def evaluate_fold(data, z, exposure, ids, folds, row_folds, fold, budget):
    held_out = folds == fold
    training_countries, testing_countries = ids[~held_out], ids[held_out]
    test_rows = np.isin(data.country, testing_countries)
    train, test = data.subset(row_folds != fold), data.subset(test_rows)
    regressor, classifier = learners()
    train_scores = country_scores(
        train, score_training(train, regressor, classifier, seed=420 + fold), training_countries,
    )
    test_scores = country_scores(
        test, score_holdout(train, test, regressor, classifier, seed=520 + fold), testing_countries,
    )
    rules, choices = fit_rules(z[~held_out], exposure[~held_out], (exposure[~held_out, None] * train_scores).T)
    selected = assignment_matrix([rules[i] for i in choices.values()], z[held_out], exposure[held_out], budget)
    assert np.all(selected.sum(axis=1) == budget)
    robust_rule = rules[choices["Robust swaps"]]
    column, threshold = robust_rule.split if robust_rule.split is not None else (None, None)
    meta = {
        "fold": fold, "eligible": len(testing_countries), "budget": budget,
        "robust_feature": POLICY_FEATURES[column] if column is not None else None,
        "robust_threshold": threshold,
        "preferred_side": ("low" if robust_rule.prefer_low else "high") if column is not None else None,
        "swap_cap_fraction": robust_rule.fraction,
        "robust_swaps": int((selected[METHODS.index("Robust swaps")] & ~selected[0]).sum()),
    }
    return testing_countries, exposure[held_out], test_scores, selected, meta


def evaluate_application(data, countries, z, exposure, folds, row_folds, design, jobs):
    ids = countries.index.to_numpy()
    sizes = np.bincount(folds)
    # Allocate the pooled quota proportionally with cumulative rounding down.
    quotas = np.diff(np.floor(BUDGET * np.r_[0, sizes.cumsum()]).astype(int))
    evaluations = Parallel(n_jobs=jobs, verbose=5)(
        delayed(evaluate_fold)(data, z, exposure, ids, folds, row_folds, fold, quotas[fold])
        for fold in range(len(sizes))
    )
    test_ids, exposure, test_scores, masks, meta = zip(*evaluations)
    cid = np.concatenate(test_ids)
    assert len(np.unique(cid)) == len(ids)
    b = np.concatenate(exposure)[:, None] * np.concatenate(test_scores)
    masks = np.concatenate(masks, axis=1)
    assert np.all(masks.sum(axis=1) == int(BUDGET * len(ids)))
    rows = []
    for method, mask in zip(METHODS, masks):
        delta = mask.astype(float) - masks[0]
        for scenario, values in zip(SCENARIOS[:2], b.T):
            stat = cluster_bootstrap(delta * values, cid, n_boot=4000, seed=42)
            rows.append({
                "design": design, "method": method, "evaluation": scenario,
                "value": np.mean(mask * values),
                "gain": stat["estimate"], "lower": stat["ci"][0], "upper": stat["ci"][1],
                "treated": mask.sum(), "swaps_from_baseline": (mask & ~masks[0]).sum(),
            })
    return pd.DataFrame(rows), pd.DataFrame(meta).assign(design=design)


def empirical(jobs):
    data, countries, z, exposure, row_regions = agri_frame()
    country_folds = make_folds(countries.index.to_numpy(), K=5, seed=42)
    row_country_folds = pd.Series(country_folds, index=countries.index).reindex(
        data.country, fill_value=-1,
    ).to_numpy()
    regions, region_folds = np.unique(countries["region"].to_numpy(), return_inverse=True)
    row_region_folds = pd.Series(np.arange(len(regions)), index=regions).reindex(
        row_regions, fill_value=-1,
    ).to_numpy()
    primary, primary_rules = evaluate_application(
        data, countries, z, exposure, country_folds, row_country_folds, "Country folds", jobs,
    )
    regional, regional_rules = evaluate_application(
        data, countries, z, exposure, region_folds, row_region_folds, "Region folds", jobs,
    )
    regional_rules["region"] = regions[regional_rules["fold"].to_numpy()]
    return pd.concat([primary, regional], ignore_index=True), pd.concat(
        [primary_rules, regional_rules], ignore_index=True,
    )


def write_simulation_table(summary, path):
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Rule & Gain (MC SE) & Regret & Loss rate (\%) \\",
    ]
    for design, group in summary.groupby("design", sort=False):
        lines += [r"\midrule", rf"\multicolumn{{4}}{{l}}{{\emph{{{design}}}}} \\"]
        for row in group.itertuples():
            lines.append(
                f"{row.method} & {row.gain:.4f} ({row.gain_mc_se:.4f}) & "
                f"{row.regret:.4f} & {100 * row.loss_rate:.1f}" + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def write_application_table(application, path):
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Rule & Gain [95\% interval] & Replacements \\",
    ]
    primary = application[application["evaluation"] == "DR 0.10"]
    for design, group in primary.groupby("design", sort=False):
        lines += [r"\midrule", rf"\multicolumn{{3}}{{l}}{{\emph{{{design}}}}} \\"]
        for row in group.itertuples():
            gain = abs(row.gain) if abs(row.gain) < 0.0005 else row.gain
            lines.append(
                f"{row.method} & {gain:.3f} [{row.lower:.3f}, {row.upper:.3f}] & "
                f"{row.swaps_from_baseline}" + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=500, help="replications per simulation design")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--mode", choices=("both", "simulation", "empirical", "tables"), default="both")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "robust_allocation")
    args = parser.parse_args()
    if args.reps < 2 or args.jobs < 1:
        parser.error("use at least two replications and at least one job")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("both", "simulation"):
        simulation, pairs = simulate_many(args.reps, args.jobs)
        pairs.to_csv(args.output_dir / "simulation_pairs.csv", index=False, float_format="%.8f")
        simulation.to_csv(args.output_dir / "simulation.csv", index=False, float_format="%.8f")
        write_simulation_table(simulation, args.output_dir / "simulation.tex")
        print(simulation.to_string(index=False), flush=True)
    if args.mode in ("both", "empirical"):
        application, folds = empirical(args.jobs)
        application.to_csv(args.output_dir / "application.csv", index=False, float_format="%.8f")
        folds.to_csv(args.output_dir / "folds.csv", index=False)
        write_application_table(application, args.output_dir / "application.tex")
        print(application.to_string(index=False), flush=True)
        print(folds.to_string(index=False), flush=True)

    if args.mode == "tables":
        write_simulation_table(pd.read_csv(args.output_dir / "simulation.csv"),
                               args.output_dir / "simulation.tex")
        write_application_table(pd.read_csv(args.output_dir / "application.csv"),
                                args.output_dir / "application.tex")


if __name__ == "__main__":
    main()
