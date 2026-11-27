"""Monte Carlo study for EW-DRPL (design section 6).

Outputs (results/montecarlo/):
  mc_results.csv   raw results, one row per (N, rep, eps, method) and ATE rows
  mc_summary.csv   mean, sd and MC SE by (N, eps, method, metric)
  mc_table.tex     booktabs LaTeX tables (regret, recall, ATE bias/RMSE)
  fig_regret.pdf/.png  regret by method, eps and N

DGP (section 6, with the specified choices made explicit):
  * N countries, J = 4 spells each, D = 10 deciles.
  * Covariates X (6): lgdp, gini, internet, agri, btemp, ksov (correlated normals).
  * T_ij ~ Bernoulli(expit(1.5 lgdp_i + 0.5 internet_i)), drawn per spell.
  * S_ij ~ N(0, sigma_i^2), sigma_i = exp(0.3 agri_i)  (s_bar_i = sigma_i, known exposure).
  * beta_d(X,T) = -(1 + 0.5 agri - 0.1 d) + T rho_d(X),
    rho_d(X) = 0.8 (1 - d/10) 1{internet > 0}.
  * Y_ijd = beta_d(X_i,T_ij) S_ij + g_d(X_i,T_ij) + e_ijd, e ~ N(0, 0.5^2),
    g_d = 0.3 tanh(lgdp) + 0.2 sin(gini) + 0.1 agri^2 - 0.05 d/10 + 0.4 T.
  * Decile incomes y_id = exp(mu_i + s_i z_d), mu_i = 1.5 + 0.6 lgdp, s_i = 0.4 exp(0.5 gini),
    z_d = Phi^{-1}((d - 0.5)/10); constant within country.
Simplifications: equal country weights (no population weights); regret and recall are computed
in-sample against the true benefit of the same N countries; the ATE truth is the sample average
of rho over countries and deciles.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ewdrpl import EWDRPL  # noqa: E402
from ewdrpl.nuisance import make_folds  # noqa: E402
from ewdrpl.policy import crossfit_policy, policy_value, search_tree, topk_policy  # noqa: E402
from ewdrpl.welfare import atkinson_weights, benefit as benefit_from_gamma  # noqa: E402

OUT = HERE.parent / "results" / "montecarlo"
COV = ["lgdp", "gini", "internet", "agri", "btemp", "ksov"]
J = 4
D = 10
K_BUDGET = 0.3
EPS_GRID = (0.0, 1.0, 2.0)
N_GRID = (150, 500)
R = 200
BASE_SEED = 20260101

METHODS = ["EW-DRPL", "EW-DRPL-crossfit", "Utilitarian-DR", "Plug-in", "IPW-only",
           "Raw-psi-DR", "Naive-TWFE", "Oracle"]
ATE_METHODS = ["ATE-DR", "ATE-DR-raw", "ATE-plugin", "ATE-IPW", "ATE-naive"]


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


# covariance of the latent normals (lgdp, gini, internet, agri, btemp, ksov)
_C = np.eye(6)
for (a, b), r in {(0, 1): -0.3, (0, 2): 0.5, (0, 5): 0.4, (1, 3): 0.3, (3, 4): 0.3,
                  (2, 5): 0.3}.items():
    _C[a, b] = _C[b, a] = r
_L = np.linalg.cholesky(_C)
_dec = np.arange(1, D + 1)
_z = norm.ppf((_dec - 0.5) / D)


def simulate(N, rng):
    Zl = rng.normal(size=(N, 6))
    X = Zl @ _L.T
    lgdp, gini, internet, agri = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    p = _expit(1.5 * lgdp + 0.5 * internet)
    sigma = np.exp(0.3 * agri)
    rho_d = 0.8 * (1 - _dec / 10)
    rho = (internet > 0)[:, None] * rho_d[None, :]                  # (N, D)
    s_inc = 0.4 * np.exp(0.5 * gini)
    mu = 1.5 + 0.6 * lgdp
    income = np.exp(mu[:, None] + s_inc[:, None] * _z[None, :])      # (N, D)
    gd = (0.3 * np.tanh(lgdp)[:, None] + 0.2 * np.sin(gini)[:, None]
          + 0.1 * agri[:, None] ** 2 - 0.05 * _dec[None, :] / 10)     # (N, D)

    n = N * J
    cid = np.repeat(np.arange(N), J)
    T = rng.binomial(1, p[cid])
    S = rng.normal(0.0, sigma[cid])
    beta = -(1 + 0.5 * agri[cid][:, None] - 0.1 * _dec[None, :]) + T[:, None] * rho[cid]
    g = gd[cid] + 0.4 * T[:, None]
    Y = beta * S[:, None] + g + rng.normal(0.0, 0.5, size=(n, D))

    df = pd.DataFrame({
        "spell": np.repeat(np.arange(n), D),
        "country": np.repeat(cid, D),
        "d": np.tile(_dec, n),
        "Y": Y.ravel(),
        "S": np.repeat(S, D),
        "T": np.repeat(T, D),
        "income": income[cid].ravel(),
        "exposure": np.repeat(sigma[cid], D),
    })
    for j, c in enumerate(COV):
        df[c] = np.repeat(X[cid, j], D)
    return df, {"X": X, "rho": rho, "sbar": sigma, "income": income, "Y": Y, "S": S, "T": T,
                "cid": cid}


def naive_twfe(sim, N, k):
    """Pooled OLS of Y on decile dummies, S, S*T and S*T*Z (Z standardised country covariates)."""
    X = sim["X"]
    Zc = (X - X.mean(0)) / X.std(0)
    cid = sim["cid"]
    n = N * J
    Ys = sim["Y"].ravel()
    Srow = np.repeat(sim["S"], D)
    Trow = np.repeat(sim["T"], D)
    dummies = np.tile(np.eye(D), (n, 1))
    Zrow = np.repeat(Zc[cid], D, axis=0)
    ST = Srow * Trow
    design = np.column_stack([dummies, Srow, ST, ST[:, None] * Zrow])
    coef, *_ = np.linalg.lstsq(design, Ys, rcond=None)
    c = coef[D + 1]
    delta = coef[D + 2:]
    pred = c + Zc @ delta
    return topk_policy(pred, k), float(pred.mean())


def _row(N, rep, eps, method, pi, V, V_or, pi_or, B_true):
    n_or = pi_or.sum()
    recall = float((pi & pi_or).sum() / n_or) if n_or > 0 else np.nan
    return {"N": N, "rep": rep, "eps": eps, "method": method, "regret": V_or - V,
            "recall": recall, "value": V, "oracle_value": V_or, "n_treated": int(pi.sum()),
            "ate_hat": np.nan, "ate_true": np.nan, "err": np.nan}


def run_rep(N, rep):
    seed = BASE_SEED + 100_000 * (0 if N == N_GRID[0] else 1) + rep
    rng = np.random.default_rng(seed)
    df, sim = simulate(N, rng)
    est = EWDRPL(K=5, seed=42).fit(
        df, unit="spell", group="country", decile="d", outcome="Y", shock="S", treat="T",
        covariates=COV, income="income", exposure="exposure")
    assert np.array_equal(est.countries_, np.arange(N))
    Z = est.Z_
    folds = est.fold_
    k = K_BUDGET
    rho = sim["rho"]
    sbar = sim["sbar"]
    income = sim["income"]
    rows = []

    # ATE rows
    ate_true = float(rho.mean())
    pi_naive, ate_naive = naive_twfe(sim, N, k)
    ate_vals = {"ATE-DR": est.mats_["dr"].mean(), "ATE-DR-raw": est.mats_["dr_raw"].mean(),
                "ATE-plugin": est.mats_["plugin"].mean(), "ATE-IPW": est.mats_["ipw"].mean(),
                "ATE-naive": ate_naive}
    for m, v in ate_vals.items():
        rows.append({"N": N, "rep": rep, "eps": np.nan, "method": m, "regret": np.nan,
                     "recall": np.nan, "value": np.nan, "oracle_value": np.nan,
                     "n_treated": np.nan, "ate_hat": float(v), "ate_true": ate_true,
                     "err": float(v) - ate_true})

    B_u = est.benefit(0.0, "dr").to_numpy()
    pi_u = search_tree(Z, B_u, k).predict(Z)

    for eps in EPS_GRID:
        omega = atkinson_weights(income, eps)
        B_true = benefit_from_gamma(rho, omega, sbar)
        pi_or = search_tree(Z, B_true, k).predict(Z)
        V_or = policy_value(pi_or, B_true)
        B_dr = est.benefit(eps, "dr").to_numpy()
        pi_dr = search_tree(Z, B_dr, k).predict(Z)
        pi_cf, _, _ = crossfit_policy(Z, B_dr, k, folds)
        pi_plug = search_tree(Z, est.benefit(eps, "plugin").to_numpy(), k).predict(Z)
        pi_ipw = search_tree(Z, est.benefit(eps, "ipw").to_numpy(), k).predict(Z)
        pi_raw = search_tree(Z, est.benefit(eps, "dr_raw").to_numpy(), k).predict(Z)
        cand = {"EW-DRPL": pi_dr, "EW-DRPL-crossfit": pi_cf, "Utilitarian-DR": pi_u,
                "Plug-in": pi_plug, "IPW-only": pi_ipw, "Raw-psi-DR": pi_raw,
                "Naive-TWFE": pi_naive, "Oracle": pi_or}
        for m in METHODS:
            pi = cand[m]
            V = policy_value(pi, B_true)
            rows.append(_row(N, rep, eps, m, pi, V, V_or, pi_or, B_true))
    return rows


def _flatten(results):
    out = []
    for r in results:
        out.extend(r)
    return pd.DataFrame(out)


def summarise(raw):
    pol = raw[~raw["method"].str.startswith("ATE")]
    rows = []
    for (N, eps, m), g in pol.groupby(["N", "eps", "method"], dropna=False):
        for metric in ["regret", "recall", "value", "n_treated"]:
            x = g[metric].dropna()
            n = len(x)
            sd = x.std(ddof=1)
            rows.append({"N": N, "eps": eps, "method": m, "metric": metric, "mean": x.mean(),
                         "sd": sd, "mc_se": sd / np.sqrt(n), "n_reps": n})
    ate = raw[raw["method"].str.startswith("ATE")]
    for (N, m), g in ate.groupby(["N", "method"]):
        err = g["err"]
        n = len(err)
        rows.append({"N": N, "eps": np.nan, "method": m, "metric": "ate_bias",
                     "mean": err.mean(), "sd": err.std(ddof=1),
                     "mc_se": err.std(ddof=1) / np.sqrt(n), "n_reps": n})
        rows.append({"N": N, "eps": np.nan, "method": m, "metric": "ate_rmse",
                     "mean": float(np.sqrt((err ** 2).mean())), "sd": np.nan, "mc_se": np.nan,
                     "n_reps": n})
    return pd.DataFrame(rows)


def latex_tables(summ, n_reps):
    def cell(N, eps, m, metric, fmt="{:.3f}"):
        r = summ[(summ.N == N) & (summ.eps == eps) & (summ.method == m) & (summ.metric == metric)]
        if r.empty:
            return "--"
        r = r.iloc[0]
        return f"{fmt.format(r['mean'])} ({fmt.format(r['mc_se'])})"

    lines = []
    lines.append("% Monte Carlo summary for EW-DRPL. Requires \\usepackage{booktabs}.")
    lines.append("% Cells: mean (MC standard error) over %d replications." % n_reps)
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append(" & \\multicolumn{3}{c}{$N=150$} & \\multicolumn{3}{c}{$N=500$} \\\\")
    lines.append("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}")
    lines.append("Method & $\\varepsilon=0$ & $\\varepsilon=1$ & $\\varepsilon=2$ & "
                 "$\\varepsilon=0$ & $\\varepsilon=1$ & $\\varepsilon=2$ \\\\")
    for metric, title in [("regret", "Panel A: regret vs.\\ oracle tree (true welfare)"),
                          ("recall", "Panel B: share of oracle-treated countries recovered")]:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for m in METHODS:
            vals = [cell(N, e, m, metric) for N in N_GRID for e in EPS_GRID]
            lines.append(m.replace("-", "-") + " & " + " & ".join(vals) + " \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{7}{l}{\\emph{Panel C: ATE of $\\rho$ (average over countries and deciles): bias (MC SE) and RMSE}} \\\\")
    for m in ATE_METHODS:
        cells = []
        for N in N_GRID:
            b = summ[(summ.N == N) & (summ.method == m) & (summ.metric == "ate_bias")]
            r = summ[(summ.N == N) & (summ.method == m) & (summ.metric == "ate_rmse")]
            cells += [f"{b['mean'].iloc[0]:.3f} ({b['mc_se'].iloc[0]:.3f})",
                      f"{r['mean'].iloc[0]:.3f}", ""]
        lines.append(m + " & " + " & ".join(cells[:3]) + " & " + " & ".join(cells[3:]) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def make_figure(summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {0.0: "#0072B2", 1.0: "#E69F00", 2.0: "#009E73"}
    labels = {0.0: "eps = 0 (utilitarian)", 1.0: "eps = 1 (log)", 2.0: "eps = 2"}
    show = ["EW-DRPL", "EW-DRPL-crossfit", "Utilitarian-DR", "Plug-in", "IPW-only",
            "Raw-psi-DR", "Naive-TWFE"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, N in zip(axes, N_GRID):
        for i, m in enumerate(show):
            for e_i, e in enumerate(EPS_GRID):
                r = summ[(summ.N == N) & (summ.eps == e) & (summ.method == m) & (summ.metric == "regret")]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = i + (e_i - 1) * 0.22
                ax.errorbar(r["mean"], y, xerr=1.96 * r["mc_se"], fmt="o", color=colors[e],
                            ms=5, capsize=2, label=labels[e] if i == 0 else None)
        ax.set_yticks(range(len(show)))
        ax.set_yticklabels(show)
        ax.invert_yaxis()
        ax.set_title(f"N = {N}")
        ax.set_xlabel("Regret vs. oracle tree (mean, 95% MC CI)")
        ax.grid(axis="x", color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_regret.pdf")
    fig.savefig(OUT / "fig_regret.png", dpi=200)
    plt.close(fig)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=R)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        raw = pd.read_csv(OUT / "mc_results.csv")
    else:
        t0 = time.time()
        jobs = [(N, r) for N in N_GRID for r in range(args.reps)]
        results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
            delayed(run_rep)(N, r) for N, r in jobs)
        raw = _flatten(results)
        raw.to_csv(OUT / "mc_results.csv", index=False)
        print(f"Replications finished in {(time.time() - t0) / 60:.1f} min")
    summ = summarise(raw)
    summ.to_csv(OUT / "mc_summary.csv", index=False)
    n_reps = int(summ["n_reps"].max())
    (OUT / "mc_table.tex").write_text(latex_tables(summ, n_reps))
    make_figure(summ)
    print(summ[summ.metric.isin(["regret", "recall"])].round(3).to_string())


if __name__ == "__main__":
    main()
