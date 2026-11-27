"""Agricultural yield resilience: main application (v2 redesign).

Outcome dly_it = 100 * (ln cereal yield_t - ln cereal yield_{t-1}); D = 1.
S = hotm (hottest-month temperature anomaly, detrended 1980-2023); T = T_res at t-1
(income-adjusted climate-AI capability); X as in the design (gini dropped, >25% missing,
kept in Z through its country mean). EW-DRPL with 20 fold seeds (42-61); Gamma = median across
seeds (mean kept for comparison).

Welfare (equity-weighted, yield resilience accrues to agricultural households):
  B_i(eps) = s_i * n_i * mean_d( omega_id(eps) * e_id * Gamma_i ),
  e_id = clip(10 a_i - (d - 1), 0, 1) (bottom-first, main) or e_id = a_i (uniform, sensitivity),
  a_i = mean agricultural employment share / 100, omega from decile means of the latest PIP survey.
  Gamma_i = country mean over years of the median-over-seeds Gamma. The package benefit() is
  used with gamma_bar = e_id * Gamma_i; policy search/cross-fitting uses the package functions.

Run:   python code/31_application_agri.py
       APP_SMOKE=1 python code/31_application_agri.py   (two seeds, scratch output)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from ewdrpl import EWDRPL  # noqa: E402
from ewdrpl.inference import blp, clan, cluster_bootstrap, cluster_mean_se, gates  # noqa: E402
from ewdrpl.nuisance import crossfit_propensity, default_classifier, make_folds  # noqa: E402
from ewdrpl.policy import crossfit_policy, search_tree, topk_policy  # noqa: E402
from ewdrpl.welfare import atkinson_weights, benefit as benefit_from_gamma  # noqa: E402
from application_utils import (  # noqa: E402
    BLACK, DARK, LIGHT, MID, fig_style, fmt, fmt_se, save_fig, write_table,
)

PROC = ROOT / "data" / "processed"
SMOKE = os.environ.get("APP_SMOKE") == "1"
OUT = (Path("/tmp/claude-0/-home-user-Artificial-Intelligence-Climate-Change-and-Inclusive-Socio-Economic-Development/"
            "fedb9d6e-2447-5a7d-8860-2639400da4da/scratchpad/app_agri_smoke") if SMOKE
       else ROOT / "results" / "application")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
FOLD_SEEDS = [42, 43] if SMOKE else list(range(42, 62))
K_FOLDS = 5
K_MAIN = 0.3
K_SENS = [0.2, 0.5]
EPS = [0.0, 0.5, 1.0, 2.0]
EPS_EQ = [1.0, 2.0]
N_BOOT = 20 if SMOKE else 500
N_RANDOM = 50 if SMOKE else 1000
ASEAN = ["VNM", "IDN", "PHL", "THA", "MYS", "KHM", "LAO", "MMR"]
X_AG = ["ln_gdppc", "internet", "agri_share", "urban", "ln_pop", "readiness", "ln_ai_pc_lag",
        "base_temp", "precip_shock", "ln_fert"]           # gini dropped (25.7% missing)
Z_NAMES = ["ln_gdppc", "gini", "internet", "agri_share", "base_temp", "sov_z"]
DEC = [f"y_dec{d}" for d in range(1, 11)]

T0 = time.time()
SUM: dict[str, list[str]] = {}


def log(section: str, msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)
    SUM.setdefault(section, []).append(msg)


# ============================================================ data
def load_panel() -> pd.DataFrame:
    p = pd.read_csv(PROC / "agri_panel.csv")
    p["gini_f"] = p["gini"].fillna(p.groupby("iso3")["gini"].transform("mean"))
    return p


def region_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    regs = sorted(df["region"].dropna().unique())
    out = pd.DataFrame(index=df.index)
    cols = []
    for j, r in enumerate(regs[1:], start=1):
        out[f"reg{j}"] = (df["region"] == r).astype(float)
        cols.append(f"reg{j}")
    return out, cols


def make_frame(p: pd.DataFrame, outcome="dly", S="hotm", T="T_res", sbar="hotm_sd_i", need_extra=()):
    need = [outcome, S, T, sbar, "region", "year"] + X_AG + list(need_extra)
    d = p.dropna(subset=need).copy()
    rd, rc = region_dummies(d)
    d = pd.concat([d, rd], axis=1)
    d["uid"] = d["iso3"] + "_" + d["year"].astype(str)
    d["d"] = 1
    d["Y"] = d[outcome].astype(float)
    d["S_spell"] = d[S].astype(float)
    d["T"] = d[T].astype(float)
    d["y_d1"] = 1.0
    d["s_bar"] = d[sbar].astype(float)
    return d.sort_values(["iso3", "year"]).reset_index(drop=True), rc


def xlist(rc):
    return X_AG + ["year"] + rc


def _fit_one(frame, X, seed, reg, clf, e_clip):
    return EWDRPL(K=K_FOLDS, seed=seed, regressor=reg, classifier=clf, e_clip=e_clip).fit(
        frame, unit="uid", group="iso3", decile="d", outcome="Y", shock="S_spell", treat="T",
        covariates=X, income="y_d1", exposure="s_bar")


def fit_seeds(frame, X, reg=None, clf=None, e_clip=(0.1, 0.9), n_jobs=4):
    return Parallel(n_jobs=n_jobs)(delayed(_fit_one)(frame, X, s, reg, clf, e_clip) for s in FOLD_SEEDS)


def aggregate(ests, frame) -> dict:
    base = ests[0]
    units = np.sort(base.scores_["unit"].unique())

    def piv(est, col):
        return est.scores_.pivot(index="unit", columns="d", values=col).loc[units, [1]].to_numpy(float)[:, 0]

    G, Gm = {}, {}
    for col, key in [("gamma_dr", "dr"), ("gamma_raw", "raw"), ("psi_star", "psi"), ("rho_plugin", "rho")]:
        stack = np.stack([piv(e, col) for e in ests])
        G[key] = np.median(stack, axis=0)
        Gm[key] = stack.mean(axis=0)
    seed_ate = np.array([float(piv(e, "gamma_dr").mean()) for e in ests])
    meta = frame.set_index("uid").loc[units].reset_index()
    e_hat = base.scores_.groupby("unit")["e_hat"].first().loc[units].to_numpy(float)
    return {"units": units, "G": G, "Gmean": Gm, "meta": meta, "groups": meta["iso3"].to_numpy(),
            "e_hat": e_hat, "seed_ate": seed_ate, "fold_country": base.fold_, "countries": base.countries_}


def seed_stats(agg) -> dict:
    s = agg["seed_ate"]
    q = np.quantile(s, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {"min": q[0], "p25": q[1], "median": q[2], "p75": q[3], "max": q[4],
            "median_agg": float(agg["G"]["dr"].mean()), "mean_agg": float(agg["Gmean"]["dr"].mean())}


def twfe(d: pd.DataFrame, y: str, regs: list[str]):
    d = d.dropna(subset=[y] + regs).copy()
    FE = pd.concat([pd.get_dummies(d["iso3"], drop_first=True, dtype=float),
                    pd.get_dummies(d["year"].astype(str), drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(pd.concat([d[regs].astype(float), FE], axis=1), has_constant="add")
    res = sm.OLS(d[y].astype(float).to_numpy(), X.to_numpy(float)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["iso3"])[0]})
    names = list(X.columns)
    out = {r: (float(res.params[names.index(r)]), float(res.bse[names.index(r)]), float(res.pvalues[names.index(r)]))
           for r in regs}
    return out, int(res.nobs), int(d["iso3"].nunique())


# ============================================================ country table and welfare
def country_table(agg, p: pd.DataFrame, sbar_col="hotm_sd_i") -> dict:
    """Country-level inputs. Gamma_i and rho_i: means over years of the median-over-seeds scores."""
    m = agg["meta"].copy()
    m["Gam"] = agg["G"]["dr"]
    m["rho"] = agg["G"]["rho"]
    g = m.groupby("iso3")
    c = pd.DataFrame({"Gam": g["Gam"].mean(), "rho": g["rho"].mean(), "ln_gdppc": g["ln_gdppc"].mean(),
                      "gini": g["gini_f"].mean(), "internet": g["internet"].mean(),
                      "agri_share": g["agri_share"].mean(), "base_temp": g["base_temp"].mean(),
                      "ln_kpc_lag": g["ln_kpc_lag"].mean(), "n_years": g.size()})
    pp = p.set_index("iso3")
    c["agri"] = p.groupby("iso3")["agri_empl"].mean() / 100
    c["s_bar"] = p.groupby("iso3")[sbar_col].first()
    c["pop"] = pp.groupby(level=0)["pop_2023"].first()
    c["emde"] = pp.groupby(level=0)["emde"].first()
    c["name"] = pp.groupby(level=0)["country"].first()
    c["region"] = pp.groupby(level=0)["region"].first()
    sovl = p.dropna(subset=["sovereignty"]).groupby("iso3")["sovereignty"].last()
    c["sov"] = sovl.reindex(c.index).fillna(-0.1)
    for dcol in DEC:
        c[dcol] = pp.groupby(level=0)[dcol].first()
    c["has_inc"] = c[DEC].notna().all(axis=1)
    keep = c["has_inc"] & c[["Gam", "rho", "gini", "agri", "s_bar", "pop", "ln_kpc_lag", "ln_gdppc", "base_temp"]].notna().all(axis=1)
    c = c[keep]
    c["gini"] = c["gini"]
    N = len(c)
    Inc = c[DEC].to_numpy(float)
    w = c["pop"].to_numpy(float) / c["pop"].sum()
    a = c["agri"].to_numpy(float)
    dd = np.arange(1, 11)[None, :]
    e_bottom = np.clip(10 * a[:, None] - (dd - 1), 0, 1)
    e_uniform = np.repeat(a[:, None], 10, axis=1)
    Z = c[["ln_gdppc", "gini", "internet", "agri_share", "base_temp", "sov"]].to_numpy(float)
    return {"iso3": c.index.to_numpy(), "name": c["name"].to_numpy(), "emde": c["emde"].to_numpy(int),
            "region": c["region"].to_numpy(), "Gam": c["Gam"].to_numpy(float), "rho": c["rho"].to_numpy(float),
            "Inc": Inc, "w": w, "s_bar": c["s_bar"].to_numpy(float), "Z": Z, "agri": a,
            "e": {"bottom": e_bottom, "uniform": e_uniform}, "ln_kpc": c["ln_kpc_lag"].to_numpy(float),
            "gdppc": c["ln_gdppc"].to_numpy(float), "N": N, "tab": c}


def B_of(c, eps: float, which: str = "Gam", expo: str = "bottom"):
    om = atkinson_weights(c["Inc"], eps, c["w"])
    base = c[which]
    gam = c["e"][expo] * base[:, None]
    return benefit_from_gamma(gam, om, c["s_bar"], c["w"])


def ci_of(contrib, clusters):
    r = cluster_bootstrap(contrib, clusters, n_boot=N_BOOT, seed=SEED)
    return r["estimate"], r["ci"][0], r["ci"][1]


def policy_block(c, k: float, folds: np.ndarray, expo: str = "bottom") -> dict:
    Z = c["Z"]
    iso = c["iso3"]
    N = len(iso)
    m = int(np.floor(k * N + 1e-9))
    names = ["ln_gdppc", "gini", "internet", "agri_share", "base_temp", "sov_z"]
    B = {e: B_of(c, e, "Gam", expo) for e in EPS}
    Bp = {e: B_of(c, e, "rho", expo) for e in EPS}
    tree = {e: search_tree(Z, B[e], k, depth=2, feature_names=names) for e in EPS}
    pi_full = {e: tree[e].predict(Z) for e in EPS}
    pi_cf = {e: crossfit_policy(Z, B[e], k, folds)[0] for e in EPS}
    pi_cfp = {e: crossfit_policy(Z, Bp[e], k, folds)[0] for e in EPS}
    comp = {
        "capable": topk_policy(c["ln_kpc"], k).astype(float),
        "poorest": topk_policy(-c["gdppc"], k).astype(float),
        "agri": topk_policy(c["agri"], k).astype(float),
    }
    rng = np.random.default_rng(SEED)
    idx = np.argsort(rng.random((N_RANDOM, N)), axis=1)[:, :m]
    p_rand = np.bincount(idx.ravel(), minlength=N) / N_RANDOM
    methods = {
        "EW-DRPL (own-eps rule, held out)": lambda e: pi_cf[e].astype(float),
        "Utilitarian tree (eps=0, held out)": lambda e: pi_cf[0.0].astype(float),
        "Plug-in tree (held out)": lambda e: pi_cfp[e].astype(float),
        "Capable-first (top 30% by ln k_pc)": lambda e: comp["capable"],
        "Poorest-first (bottom 30% by ln GDP pc)": lambda e: comp["poorest"],
        "Most-agricultural-first (top 30% by agri share)": lambda e: comp["agri"],
        "Random 30% (expected over draws)": lambda e: p_rand,
    }
    values = []
    for name, pif in methods.items():
        for e in EPS:
            est, lo, hi = ci_of(pif(e) * B[e], iso)
            values.append({"method": name, "eps": e, "value": est, "ci_lo": lo, "ci_hi": hi})
    matrix = {(e, e2): ci_of(pi_cf[e].astype(float) * B[e2], iso) for e in EPS for e2 in EPS}
    equity = []
    for e in EPS_EQ:
        g_ = ci_of((pi_cf[e].astype(float) - pi_cf[0.0].astype(float)) * B[e], iso)
        p_ = ci_of((pi_cf[0.0].astype(float) - pi_cf[e].astype(float)) * B[0.0], iso)
        equity.append({"eps": e, "gain": g_[0], "gain_lo": g_[1], "gain_hi": g_[2],
                       "price": p_[0], "price_lo": p_[1], "price_hi": p_[2]})
    return {"k": k, "N": N, "m": m, "B": B, "tree": tree, "pi_full": pi_full, "pi_cf": pi_cf,
            "values": pd.DataFrame(values), "matrix": matrix, "equity": pd.DataFrame(equity),
            "p_rand": p_rand}


def fig_policy(csv_path: Path, stem: str) -> None:
    import re
    d = pd.read_csv(csv_path)
    nums = d["value [95% CI]"].map(lambda t: [float(x) for x in re.findall(r"-?\d+\.\d+", t)])
    d["est"] = [v[0] for v in nums]
    d["lo"] = [v[1] for v in nums]
    d["hi"] = [v[2] for v in nums]
    methods = list(dict.fromkeys(d["method"]))
    ypos = {m: i for i, m in enumerate(methods)}
    fig, axes = plt.subplots(1, len(EPS), figsize=(7.4, 3.6), sharey=True)
    for ax, e in zip(axes, EPS):
        sub = d[np.isclose(d["eps"].astype(float), e)]
        y = sub["method"].map(ypos).to_numpy()
        ax.errorbar(sub["est"], y, xerr=[sub["est"] - sub["lo"], sub["hi"] - sub["est"]], fmt="o",
                    color=BLACK, ms=4, capsize=2, lw=1)
        ax.axvline(0, color=LIGHT, lw=0.8)
        ax.set_title(rf"$\varepsilon$ = {e:g}")
        ax.set_xlabel("welfare value")
    axes[0].set_yticks(list(ypos.values()))
    axes[0].set_yticklabels(methods, fontsize=6.5)
    axes[0].invert_yaxis()
    fig.tight_layout()
    save_fig(fig, OUT / stem)


def treated_set(c, expo="bottom", eps=1.0, k=K_MAIN):
    Z = c["Z"]
    B = B_of(c, eps, "Gam", expo)
    tr = search_tree(Z, B, k, depth=2, feature_names=["ln_gdppc", "gini", "internet", "agri_share", "base_temp", "sov_z"])
    return set(c["iso3"][tr.predict(Z)])


def jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else np.nan


# ============================================================ main
def main() -> None:
    fig_style()
    p = load_panel()
    frame, rc = make_frame(p)
    X = xlist(rc)
    log("sample", f"panel rows {len(p)}, countries {p.iso3.nunique()}; analysis rows {len(frame)}, countries "
                  f"{frame.iso3.nunique()}, years {int(frame.year.min())}-{int(frame.year.max())}; "
                  f"share T_res=1 {frame['T'].mean():.3f}; EMDE countries {frame[frame.emde == 1].iso3.nunique()}")
    miss = p[["dly", "hotm", "T_res", "gini", "ln_fert", "agri_empl", "precip_shock"]].isna().mean()
    log("sample", "missing shares (panel, 2000-2023): " + ", ".join(f"{k} {v:.3f}" for k, v in miss.items())
        + "; gini dropped from X (missing > 25%), kept in Z via country mean")
    log("sample", f"dropped |dly| >= 100 (data errors): 18 country-years (see 12_build_agri.py output)")

    # ---- first stage
    fs_rows = []
    for lab, dd, cond in [("all countries", p, None), ("EMDE only", p[p.emde == 1], None),
                          ("base_temp >= 20", p[p.base_temp >= 20], None)]:
        res, n, g = twfe(dd, "dly", ["hotm"])
        b, se, pv = res["hotm"]
        fs_rows.append({"sample": lab, "beta (dly on hotm)": fmt_se(b, se, 3), "p": fmt(pv, 3), "N": str(n), "countries": str(g)})
        log("firststage", f"{lab}: beta {b:.3f} (SE {se:.3f}, p {pv:.4f}), N {n}, countries {g}")
    res, n, g = twfe(p, "dly", ["shock"])
    fs_rows.append({"sample": "annual shock (all countries, comparison)", "beta (dly on hotm)": fmt_se(res["shock"][0], res["shock"][1], 3),
                    "p": fmt(res["shock"][2], 3), "N": str(n), "countries": str(g)})
    log("firststage", f"annual shock (all): beta {res['shock'][0]:.3f} (SE {res['shock'][1]:.3f}); N {n}")
    write_table(pd.DataFrame(fs_rows), OUT / "agri_firststage",
                caption="First stage: TWFE regression of the cereal-yield growth dly on the hottest-month anomaly hotm (country and year fixed effects, country-clustered SE).",
                label="tab:agri_firststage", notes="dly = 100 times the log change in cereal yield (kg/ha, WDI). Observations with |dly| >= 100 dropped. Hotm = hottest-month mean temperature minus its 1980-2023 linear trend.",
                align="lrrrr")

    # ---- EW-DRPL, 20 seeds
    t = time.time()
    ests = fit_seeds(frame, X)
    agg = aggregate(ests, frame)
    ss = seed_stats(agg)
    pd.DataFrame({"fold_seed": FOLD_SEEDS, "ate": agg["seed_ate"]}).to_csv(OUT / "agri_seed_stability.csv", index=False)
    log("fit", f"EW-DRPL over {len(FOLD_SEEDS)} fold seeds in {time.time() - t:.0f}s; X = {len(X)} covariates")
    log("ate", f"ATE of rho (median-over-seeds Gamma, mean over rows) = {ss['median_agg']:.3f}; mean-over-seeds {ss['mean_agg']:.3f}")
    log("ate", f"seed distribution of ATE: min {ss['min']:.3f}, p25 {ss['p25']:.3f}, median {ss['median']:.3f}, p75 {ss['p75']:.3f}, max {ss['max']:.3f}")
    G = agg["G"]["dr"]
    g = agg["groups"]
    ate, se = cluster_mean_se(G, g)
    t1 = agg["meta"]["T"].to_numpy() == 1
    psi = agg["G"]["psi"]
    m1, s1 = cluster_mean_se(psi[t1], g[t1])
    m0, s0 = cluster_mean_se(psi[~t1], g[~t1])
    ncty = len(np.unique(g))
    ate_row = [{"quantity": "ATE of rho (mean Gamma, median over seeds)", "estimate": fmt_se(ate, se, 3), "N": str(len(G)), "countries": str(ncty)},
               {"quantity": "mean psi* | T_res = 1 (beta, T=1)", "estimate": fmt_se(m1, s1, 3), "N": str(int(t1.sum())), "countries": "--"},
               {"quantity": "mean psi* | T_res = 0 (beta, T=0)", "estimate": fmt_se(m0, s0, 3), "N": str(int((~t1).sum())), "countries": "--"},
               {"quantity": "mean Gamma, mean-over-seeds (comparison)", "estimate": fmt(ss["mean_agg"], 3), "N": str(len(G)), "countries": str(ncty)},
               {"quantity": "seed ATE min / p25 / median / p75 / max", "estimate": " / ".join(fmt(ss[k_], 3) for k_ in ["min", "p25", "median", "p75", "max"]),
                "N": "--", "countries": "--"}]
    write_table(pd.DataFrame(ate_row), OUT / "agri_ate", caption="Average resilience effect on cereal-yield growth and mean marginal damage by capability status.",
                label="tab:agri_ate", notes="Country-clustered SE. Gamma = median across 20 fold seeds (main). psi* is the orthogonal slope score (beta(X,T)); beta by T is its mean.",
                align="lrrr")
    log("ate", f"beta(T=1) (mean psi*) {m1:.3f} (SE {s1:.3f}); beta(T=0) {m0:.3f} (SE {s0:.3f})")
    log("ate", f"Gamma quantiles (rows): p50 {np.quantile(np.abs(G), 0.5):.3f}, p95 {np.quantile(np.abs(G), 0.95):.3f}, "
               f"p99 {np.quantile(np.abs(G), 0.99):.3f}, max |Gamma| {np.abs(G).max():.2f}")
    # BLP / GATES / CLAN
    Zr = agg["meta"][Z_NAMES].copy()
    n_gini_fill = int(agg["meta"]["gini_f"].isna().sum())
    Zr["gini"] = agg["meta"]["gini_f"].fillna(agg["meta"]["gini_f"].mean())
    log("hetero", f"Gini missing for all years in some countries: {n_gini_fill} country-year rows filled with the sample mean for Z")
    Zm = Zr.to_numpy(float)
    Zs = (Zm - Zm.mean(0)) / Zm.std(0, ddof=1)
    b = blp(G, Zs, g, Z_NAMES)
    rho = agg["G"]["rho"]
    gt, gd = gates(rho, G, g, 5)
    ct = clan(Zm, rho, g, Z_NAMES, 5)
    write_table(pd.DataFrame({"term": b.term, "coef": b.coef.map(lambda v: fmt(v, 3)), "se": b.se.map(lambda v: fmt(v, 3)),
                              "p": b.p.map(lambda v: fmt(v, 3))}), OUT / "agri_blp",
                caption="BLP of the median Gamma on standardised Z (ln GDP pc, Gini, internet, agriculture share, baseline temperature, sovereignty).",
                label="tab:agri_blp", notes="Country-clustered SE. Z standardised over country-years; Gini filled with the country mean where missing.")
    gt_out = pd.DataFrame({"quintile of predicted rho": gt["group"].astype(int).to_numpy(),
                           "mean Gamma": gt["mean"].map(lambda v: fmt(v, 3)).to_numpy(),
                           "se": gt["se"].map(lambda v: fmt(v, 3)).to_numpy(), "n": gt["n"].map(lambda v: fmt(v, 0)).to_numpy()})
    gt_out.loc[len(gt_out)] = ["top minus bottom", fmt(gd["top_minus_bottom"], 3), fmt(gd["se"], 3), "--"]
    write_table(gt_out, OUT / "agri_gates_clan", caption="GATES: mean median-Gamma by quintile of out-of-fold predicted rho (plug-in m1 - m0). CLAN: covariate means by quintile.",
                label="tab:agri_gates", notes="Quintile 1 = lowest predicted resilience.")
    ct_out = ct.copy()
    for c_ in ct_out.columns:
        if c_ != "feature":
            ct_out[c_] = ct_out[c_].map(lambda v: fmt(v, 2))
    write_table(ct_out, OUT / "agri_gates_clan_covariates", caption="CLAN: mean of each Z variable by quintile of predicted rho.",
                label="tab:agri_clan", notes="q1 = lowest predicted resilience; top_minus_bottom = q5 - q1.")
    log("hetero", "BLP: " + "; ".join(f"{r.term} {r.coef:.3f} (SE {r.se:.3f})" for r in b.itertuples()))
    log("hetero", "GATES quintile means: " + ", ".join(f"q{int(r.group)} {r.mean:.3f} (SE {r.se:.3f})" for r in gt.itertuples()))
    log("hetero", f"GATES top minus bottom {gd['top_minus_bottom']:.3f} (SE {gd['se']:.3f})")
    log("hetero", "CLAN top minus bottom: " + "; ".join(f"{r.feature} {r.top_minus_bottom:.2f}" for r in ct.itertuples()))
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    xs = np.arange(1, 6)
    ax.errorbar(xs, gt["mean"], yerr=1.96 * gt["se"], fmt="o", color=BLACK, ms=4, capsize=2, lw=1)
    ax.axhline(0, color=LIGHT, lw=0.8)
    ax.set_xlabel("quintile of predicted resilience (1 = lowest)")
    ax.set_ylabel(r"mean $\Gamma$ (95% CI)")
    fig.tight_layout()
    save_fig(fig, OUT / "fig_agri_gates")

    # ---- overlap and balance
    folds_row = pd.Series(agg["fold_country"], index=agg["countries"]).loc[g].to_numpy()
    F = agg["meta"][X].astype(float).to_numpy()
    Tv = agg["meta"]["T"].to_numpy()
    e_raw = crossfit_propensity(F, Tv, folds_row, default_classifier(), clip=(1e-6, 1 - 1e-6))
    ov = {"share_out_01": float(np.mean((e_raw < 0.1) | (e_raw > 0.9))), "share_out_05": float(np.mean((e_raw < 0.05) | (e_raw > 0.95))),
          "e_T1": float(e_raw[Tv == 1].mean()), "e_T0": float(e_raw[Tv == 0].mean())}
    log("overlap", f"T_res propensity: outside [0.1,0.9] {ov['share_out_01']:.3f}; outside [0.05,0.95] {ov['share_out_05']:.3f}; "
                   f"mean e T=1 {ov['e_T1']:.3f}, T=0 {ov['e_T0']:.3f}")
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    bins = np.linspace(0, 1, 26)
    ax.hist(e_raw[Tv == 0], bins=bins, color=LIGHT, edgecolor=DARK, lw=0.5, label="T = 0")
    ax.hist(e_raw[Tv == 1], bins=bins, histtype="step", color=BLACK, lw=1.4, ls="--", label="T = 1")
    ax.axvline(0.1, color=MID, ls=":", lw=1)
    ax.axvline(0.9, color=MID, ls=":", lw=1)
    ax.set_xlabel(r"cross-fitted propensity $\hat e(X)$")
    ax.set_ylabel("country-years")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, OUT / "fig_agri_overlap")
    w1 = Tv / agg["e_hat"]
    w0 = (1 - Tv) / (1 - agg["e_hat"])
    bal = []
    for x_ in X:
        v_ = agg["meta"][x_].astype(float).to_numpy()
        sd = v_.std(ddof=1)
        smd0 = (v_[Tv == 1].mean() - v_[Tv == 0].mean()) / sd
        smd1 = (np.sum(w1 * v_) / np.sum(w1) - np.sum(w0 * v_) / np.sum(w0)) / sd
        bal.append({"covariate": x_, "SMD before": fmt(smd0, 3), "SMD after IPW": fmt(smd1, 3)})
    write_table(pd.DataFrame(bal), OUT / "agri_balance", caption="Covariate balance before and after inverse-propensity weighting (T_res).",
                label="tab:agri_balance", notes="Standardised mean differences (T=1 minus T=0), ATE weights from the cross-fitted propensity clipped to [0.1, 0.9].", align="lrr")
    log("overlap", "max |SMD| before " + f"{max(abs(float(r['SMD before'])) for r in bal):.3f}" + ", after IPW " +
        f"{max(abs(float(r['SMD after IPW'])) for r in bal):.3f}")

    # descriptives
    dsc = []
    for lab, col in [("dly (cereal-yield growth, %)", "dly"), ("hotm (hottest-month anomaly, deg C)", "hotm"),
                     ("shock (annual anomaly, deg C)", "shock"), ("agri VA growth (%)", "agri_growth_v")] + \
                    [(x_, x_) for x_ in ["ln_gdppc", "gini", "internet", "agri_share", "urban", "ln_pop", "readiness", "ln_ai_pc_lag", "base_temp", "precip_shock", "ln_fert"]]:
        vv = agg["meta"][col].astype(float) if col in agg["meta"] else frame[col].astype(float)
        tt = agg["meta"]["T"].astype(float)
        dsc.append({"variable": lab, "mean T=0": fmt(vv[tt == 0].mean(), 3), "sd T=0": fmt(vv[tt == 0].std(), 3),
                    "mean T=1": fmt(vv[tt == 1].mean(), 3), "sd T=1": fmt(vv[tt == 1].std(), 3)})
    write_table(pd.DataFrame(dsc), OUT / "agri_descriptives", caption="Descriptive statistics by income-adjusted capability status (T_res at t-1), analysis sample.",
                label="tab:agri_desc", notes="Country-year sample; means and SDs by T_res.", align="lrrrr")

    # ---- baseline TWFE
    base_regs = ["T_res", "hotm", "hotm_x_T"]
    p2 = p.copy()
    p2["hotm_x_T"] = p2["hotm"] * p2["T_res"]
    ctrl = ["ln_gdppc", "internet", "agri_share", "urban", "ln_pop", "readiness", "ln_ai_pc_lag", "precip_shock", "ln_fert"]
    rb1, n1, c1 = twfe(p2, "dly", ["hotm", "hotm_x_T", "T_res"] + ctrl)
    rb2, n2, c2 = twfe(p2, "agri_growth_v", ["hotm", "hotm_x_T", "T_res"] + ctrl)
    base_rows = []
    for lab, kk in [("hotm", "hotm"), ("hotm x T_res", "hotm_x_T"), ("T_res (main effect)", "T_res"),
                    ("ln GDP pc", "ln_gdppc"), ("internet", "internet")]:
        base_rows.append({"term": lab, "dly (cereal yield growth)": fmt_se(rb1[kk][0], rb1[kk][1], 3),
                          "agri VA growth": fmt_se(rb2[kk][0], rb2[kk][1], 3)})
    base_rows.append({"term": "N", "dly (cereal yield growth)": str(n1), "agri VA growth": str(n2)})
    base_rows.append({"term": "countries", "dly (cereal yield growth)": str(c1), "agri VA growth": str(c2)})
    write_table(pd.DataFrame(base_rows), OUT / "agri_baseline",
                caption="Baseline TWFE: country and year fixed effects; controls ln GDP pc, internet, agriculture share, urban, ln population, readiness, ln AI pc (lag), precipitation shock, ln fertilizer.",
                label="tab:agri_baseline", notes="Country-clustered SE. Gini omitted (missing 25.7%).", align="lcc")
    log("baseline", f"TWFE dly: hotm {rb1['hotm'][0]:.3f} (SE {rb1['hotm'][1]:.3f}); hotm x T_res {rb1['hotm_x_T'][0]:.3f} "
                    f"(SE {rb1['hotm_x_T'][1]:.3f}); T_res {rb1['T_res'][0]:.3f} (SE {rb1['T_res'][1]:.3f}); N {n1}")
    log("baseline", f"TWFE agri VA growth: hotm {rb2['hotm'][0]:.3f} (SE {rb2['hotm'][1]:.3f}); hotm x T_res {rb2['hotm_x_T'][0]:.3f} (SE {rb2['hotm_x_T'][1]:.3f}); N {n2}")

    # ---- policy (main: bottom-first exposure)
    c = country_table(agg, p)
    N = c["N"]
    folds_c = make_folds(c["iso3"], K=K_FOLDS, seed=SEED)
    log("policy", f"policy sample N = {N} countries (EMDE {int(c['emde'].sum())}); income deciles, Z and agriculture available; "
                  f"budget floor(0.3 N) = {int(np.floor(K_MAIN * N))}; ASEAN in sample: "
                  + ", ".join(sorted(set(c['iso3']) & set(ASEAN))))
    log("policy", f"mean agricultural employment share (policy sample) {c['agri'].mean():.3f}")
    pol = policy_block(c, K_MAIN, folds_c, "bottom")
    v = pol["values"]
    write_table(pd.DataFrame({"method": v.method, "eps": v.eps.map(lambda z: fmt(z, 1)),
                              "value [95% CI]": [f"{r.value:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]" for r in v.itertuples()]}),
                OUT / "agri_policy_values", caption="Welfare value (1/N) sum_i pi_i B_i(eps) of agricultural-resilience benefits, budget k = 0.3, bottom-first exposure.",
                label="tab:agri_policy_values", notes="Held-out values are cross-fitted over country folds (seed 42). 95% CIs: country-cluster bootstrap (B = 500).", align="llr")
    mat_rows = []
    for e in EPS:
        row = {"rule eps": fmt(e, 1)}
        for e2 in EPS:
            m_, lo, hi = pol["matrix"][(e, e2)]
            row[f"eval eps={fmt(e2, 1)}"] = f"{m_:.4f} [{lo:.4f}, {hi:.4f}]"
        mat_rows.append(row)
    write_table(pd.DataFrame(mat_rows), OUT / "agri_policy_matrix", caption="Cross-fitted held-out value of the eps rule (rows) evaluated under eps' (columns).",
                label="tab:agri_policy_matrix", notes="Budget k = 0.3. Brackets: 95% country-cluster bootstrap CI.")
    eq = pol["equity"]
    write_table(pd.DataFrame({"eps": eq.eps.map(lambda z: fmt(z, 1)),
                              "equity gain W_eps(pi_eps) - W_eps(pi_0)": [f"{r.gain:.4f} [{r.gain_lo:.4f}, {r.gain_hi:.4f}]" for r in eq.itertuples()],
                              "equity price W_0(pi_0) - W_0(pi_eps)": [f"{r.price:.4f} [{r.price_lo:.4f}, {r.price_hi:.4f}]" for r in eq.itertuples()]}),
                OUT / "agri_equity", caption="Equity gain and equity price of the eps rule against the utilitarian rule (k = 0.3).",
                label="tab:agri_equity", notes="Cross-fitted held-out values; 95% country-cluster bootstrap CIs in brackets.", align="lll")
    log("policy", "held-out matrix (rows: rule eps; columns: evaluation eps " + ", ".join(fmt(e, 1) for e in EPS) + "): "
        + " | ".join(f"eps {fmt(e, 1)}: " + ", ".join(f"{pol['matrix'][(e, e2)][0]:.4f}" for e2 in EPS) for e in EPS))
    for r in v[v.method.str.startswith("EW-DRPL")].itertuples():
        log("policy", f"own-eps held-out value eps {r.eps}: {r.value:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]")
    for r in v[v.method.str.startswith("Utilitarian")].itertuples():
        log("policy", f"utilitarian rule evaluated at eps {r.eps}: {r.value:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]")
    for r in eq.itertuples():
        log("policy", f"eps {r.eps}: equity gain {r.gain:.4f} [{r.gain_lo:.4f}, {r.gain_hi:.4f}]; equity price {r.price:.4f} [{r.price_lo:.4f}, {r.price_hi:.4f}]")
    treated_rows = []
    iso_list = list(c["iso3"])
    for e in EPS:
        tr = pol["tree"][e]
        treated = list(c["iso3"][pol["pi_full"][e]])
        log("policy", f"eps {e} rule (full sample): {tr.describe()}")
        log("policy", f"eps {e} treated ({len(treated)}): " + ", ".join(sorted(treated)))
    for i in range(N):
        flags = {e: bool(pol["pi_full"][e][i]) for e in EPS}
        if any(flags.values()):
            treated_rows.append({"ISO3": c["iso3"][i], "country": c["name"][i], "region": c["region"][i],
                                 "EMDE": "yes" if c["emde"][i] else "no", "ASEAN": "yes" if c["iso3"][i] in ASEAN else "no",
                                 "agri share": fmt(c["agri"][i], 3), "B(eps=1)": fmt(B_of(c, 1.0)[i], 4),
                                 **{f"eps {fmt(e, 1)}": "x" if flags[e] else "" for e in EPS}})
    write_table(pd.DataFrame(treated_rows).sort_values("ISO3"), OUT / "agri_treated",
                caption="Countries treated under the full-sample depth-2 tree for each eps (k = 0.3, bottom-first exposure).",
                label="tab:agri_treated", notes="x = treated. ASEAN: VNM, IDN, PHL, THA, MYS, KHM, LAO, MMR. B(eps=1) is the country benefit under eps = 1.", align="lllllrrcccc")
    if "VNM" in iso_list:
        vnm = {e: bool(pol["pi_full"][e][iso_list.index("VNM")]) for e in EPS}
        log("policy", "Viet Nam treated (full-sample tree): " + ", ".join(f"eps {fmt(e, 1)}: {'yes' if vnm[e] else 'no'}" for e in EPS))
    else:
        log("policy", "Viet Nam not in policy sample (no income deciles or Z)")
    rules = []
    for e in EPS:
        own = v[(v.method.str.startswith("EW-DRPL")) & (v.eps == e)].iloc[0]
        rules.append({"eps": fmt(e, 1), "rule (full sample, depth 2)": pol["tree"][e].describe(),
                      "treated": int(pol["pi_full"][e].sum()), "held-out value": fmt(own.value, 4)})
    write_table(pd.DataFrame(rules), OUT / "agri_policy_trees", caption="Learned depth-2 tree policies for agricultural-resilience benefits (k = 0.3).",
                label="tab:agri_trees", notes="Rules use Z <= threshold; treated-leaf flags (LL, LR, RL, RR) in the text.", align="lp{8cm}rr")
    ks_rows = []
    for kk in K_SENS:
        pk = policy_block(c, kk, folds_c, "bottom")
        for e in EPS:
            own = pk["values"][(pk["values"].method.str.startswith("EW-DRPL")) & (pk["values"].eps == e)].iloc[0]
            ks_rows.append({"k": fmt(kk, 1), "eps": fmt(e, 1), "treated": int(pk["pi_full"][e].sum()),
                            "rule": pk["tree"][e].describe(), "held-out value": fmt(own.value, 4)})
            log("budget", f"k={kk} eps={e}: treated {int(pk['pi_full'][e].sum())}; {pk['tree'][e].describe()}; held-out {own.value:.4f}")
        for r in pk["equity"].itertuples():
            log("budget", f"k={kk} eps={r.eps}: equity gain {r.gain:.4f} [{r.gain_lo:.4f}, {r.gain_hi:.4f}]; price {r.price:.4f} [{r.price_lo:.4f}, {r.price_hi:.4f}]")
    write_table(pd.DataFrame(ks_rows), OUT / "agri_budget_sensitivity", caption="Sensitivity of agricultural policy trees and held-out values to the budget share k.",
                label="tab:agri_budget", notes="Held-out value under the rule's own eps.", align="lllp{7cm}r")

    # ---- figures: first stage and policy/ranking
    res_fs = {}
    for var in ["dly", "hotm"]:
        Dm = pd.get_dummies(p["iso3"].astype(str), dtype=float).to_numpy()
        Yr = pd.get_dummies(p["year"].astype(str), dtype=float).to_numpy()
        M = np.column_stack([Dm, Yr])
        d_ = p.dropna(subset=["dly", "hotm"]).index
        Mm = M[d_]
        beta, *_ = np.linalg.lstsq(Mm, p.loc[d_, var].to_numpy(float), rcond=None)
        res_fs[var] = p.loc[d_, var].to_numpy(float) - Mm @ beta
    rh, ry = res_fs["hotm"], res_fs["dly"]
    qs = pd.qcut(rh, 20, labels=False, duplicates="drop")
    bins = pd.DataFrame({"h": rh, "y": ry, "q": qs}).groupby("q").agg(h=("h", "mean"), y=("y", "mean"), se=("y", lambda s: s.std(ddof=1) / np.sqrt(len(s))))
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.errorbar(bins["h"], bins["y"], yerr=1.96 * bins["se"], fmt="o", color=BLACK, ms=4, capsize=2, lw=0.8)
    slope = float(np.sum(rh * ry) / np.sum(rh ** 2))
    xx = np.linspace(rh.min(), rh.max(), 50)
    ax.plot(xx, slope * xx, color=MID, ls="--", lw=1)
    ax.axhline(0, color=LIGHT, lw=0.6)
    ax.set_xlabel("hottest-month anomaly, residualised (deg C)")
    ax.set_ylabel("cereal-yield growth, residualised (%)")
    ax.set_title(f"first stage (within-FE slope {slope:.2f})", fontsize=8)
    fig.tight_layout()
    save_fig(fig, OUT / "fig_agri_firststage")
    log("firststage", f"binned first-stage plot: FE-residualised slope {slope:.3f} (20 quantile bins)")

    B1 = B_of(c, 1.0)
    order = np.argsort(-B1)[:30]
    trt = pol["pi_full"][1.0]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    cols = [BLACK if trt[i] else LIGHT for i in order]
    ax.bar(np.arange(30), B1[order], color=cols, edgecolor=DARK, lw=0.4)
    ax.set_xticks(np.arange(30))
    ax.set_xticklabels(c["iso3"][order], rotation=90, fontsize=6)
    ax.set_ylabel(r"benefit $B_i(\varepsilon=1)$")
    ax.axhline(0, color=DARK, lw=0.6)
    ax.set_title("Top 30 countries by eps=1 benefit (black = treated under eps=1, k=0.3)", fontsize=8)
    fig.tight_layout()
    save_fig(fig, OUT / "fig_agri_ranking")
    fig_policy(OUT / "agri_policy_values.csv", "fig_agri_policy_values")

    # ---- sensitivity (a)-(h)
    main_set = set(c["iso3"][pol["pi_full"][1.0]])
    sens = []
    sens.append({"variant": "main (T_res, hotm, dly, bottom exposure)", "spells": str(len(G)), "countries": str(ncty),
                 "ATE (SE)": fmt_se(ate, se, 3), "eps=1 treated": str(len(main_set)), "Jaccard vs main": "1.000"})
    # (a) raw psi
    ga, sa = cluster_mean_se(agg["G"]["raw"], g)
    aset = None
    tmp = dict(c)
    tmp["Gam"] = pd.Series(agg["G"]["raw"]).groupby(g).mean().reindex(c["iso3"]).to_numpy()
    aset = treated_set(tmp)
    sens.append({"variant": "(a) raw psi score", "spells": str(len(G)), "countries": str(ncty), "ATE (SE)": fmt_se(ga, sa, 3),
                 "eps=1 treated": str(len(aset)), "Jaccard vs main": f"{jacc(aset, main_set):.3f}"})
    log("sensitivity", f"(a) raw psi: ATE {ga:.3f} (SE {sa:.3f}); Jaccard {jacc(aset, main_set):.3f}")
    specs = [
        ("(b) propensity trimming [0.05, 0.95]", frame, None, None, (0.05, 0.95), "dly", "hotm", "hotm_sd_i", "T_res"),
        ("(c) annual shock instead of hotm", frame, None, None, (0.1, 0.9), "dly", "shock", "shock_sd_i", "T_res"),
        ("(d) random forest nuisances", frame, RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=1),
         RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=1), (0.1, 0.9), "dly", "hotm", "hotm_sd_i", "T_res"),
        ("(e) EMDE-only sample", frame[frame.emde == 1].copy(), None, None, (0.1, 0.9), "dly", "hotm", "hotm_sd_i", "T_res"),
        ("(f) T_stock instead of T_res", frame, None, None, (0.1, 0.9), "dly", "hotm", "hotm_sd_i", "T_stock"),
        ("(g) agricultural VA growth outcome", frame, None, None, (0.1, 0.9), "agri_growth_v", "hotm", "hotm_sd_i", "T_res"),
    ]
    for lab, fr, reg_, clf_, eclip, outc, S_, sb, T_ in specs:
        fr2, rc2 = make_frame(p, outcome=outc, S=S_, T=T_, sbar=sb) if (lab.startswith("(f)") or lab.startswith("(g)") or lab.startswith("(c)")) else (fr, rc)
        if lab.startswith("(e)"):
            fr2, rc2 = make_frame(p[p.emde == 1])
        if lab.startswith("(b)") or lab.startswith("(d)"):
            fr2, rc2 = fr, rc
        X2 = xlist(rc2)
        ests2 = fit_seeds(fr2, X2, reg=reg_, clf=clf_, e_clip=eclip)
        agg2 = aggregate(ests2, fr2)
        s2 = seed_stats(agg2)
        G2 = agg2["G"]["dr"]
        a2, s2e = cluster_mean_se(G2, agg2["groups"])
        c2 = country_table(agg2, p, sbar_col=sb)
        ts = treated_set(c2)
        sens.append({"variant": lab, "spells": str(len(G2)), "countries": str(c2["N"]), "ATE (SE)": fmt_se(a2, s2e, 3),
                     "eps=1 treated": str(len(ts)), "Jaccard vs main": f"{jacc(ts, main_set):.3f}"})
        log("sensitivity", f"{lab}: N rows {len(G2)}, countries {c2['N']}; ATE {a2:.3f} (SE {s2e:.3f}); seed median {s2['median']:.3f} "
                           f"[min {s2['min']:.3f}, max {s2['max']:.3f}]; eps=1 treated {len(ts)}; Jaccard {jacc(ts, main_set):.3f}")
    # (h) uniform exposure: policy only
    cu = c
    ut = treated_set(cu, expo="uniform")
    pu = policy_block(c, K_MAIN, folds_c, "uniform")
    sens.append({"variant": "(h) uniform exposure e_id = a_i (policy only)", "spells": "--", "countries": str(N),
                 "ATE (SE)": "--", "eps=1 treated": str(len(ut)), "Jaccard vs main": f"{jacc(ut, main_set):.3f}"})
    log("sensitivity", f"(h) uniform exposure: eps=1 treated {len(ut)}; Jaccard {jacc(ut, main_set):.3f}; "
                       f"held-out eps=1 {pu['values'][(pu['values'].method.str.startswith('EW-DRPL')) & (pu['values'].eps == 1.0)].value.iloc[0]:.4f}; "
                       f"equity gain eps=1 {pu['equity'].gain.iloc[0]:.4f} [{pu['equity'].gain_lo.iloc[0]:.4f}, {pu['equity'].gain_hi.iloc[0]:.4f}]")
    write_table(pd.DataFrame(sens), OUT / "agri_sensitivity", caption="Sensitivity of the agricultural application: ATE of rho and overlap of the eps = 1 treated set with the main rule.",
                label="tab:agri_sensitivity", notes="Jaccard = |A and B| / |A or B| with the main eps=1 treated set (k = 0.3). Median-over-seeds ATE; 20 seeds per row. Variant (h) reports the policy only.", align="lrrrrr")

    # ---- summary
    fig_policy(OUT / "agri_policy_values.csv", "fig_agri_policy_values")
    lines = ["# Agricultural yield resilience: application results", "",
             f"Outcome dly (100 x log change in cereal yield), hotm exposure, T_res at t-1, {len(FOLD_SEEDS)} fold seeds "
             f"({FOLD_SEEDS[0]}-{FOLD_SEEDS[-1]}), Gamma = median across seeds. k = {K_MAIN}; B = {N_BOOT}; random draws {N_RANDOM}.", ""]
    for sec, msgs in SUM.items():
        lines.append(f"## {sec}")
        lines.append("")
        lines += [f"- {m}" for m in msgs]
        lines.append("")
    (OUT / "application_summary.md").write_text("\n".join(lines))
    log("runtime", f"total runtime {time.time() - T0:.0f}s")
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
