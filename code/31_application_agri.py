"""Agricultural yield resilience: main application (v2 redesign, revision 1).

Outcome dly_it = 100 * (ln cereal yield_t - ln cereal yield_{t-1}); D = 1.
S = hotm (hottest-month anomaly, detrended 1980-2023); T = T_res at t-1 (income-adjusted
climate-AI capability). X: one-year lags of ln GDP pc, internet, agriculture share, urban,
ln population, ln fertilizer (pre-determined); readiness (already t-1 in the build), ln AI pc
(lag), base temperature, precipitation shock (weather, contemporaneous). Gini is dropped from X
(25.7% missing) and kept in Z through its country mean.
EW-DRPL with 20 fold seeds (42-61). Gamma = median across seeds (main); split-adjusted inference
over seeds (Chernozhukov et al. 2018); mean and winsorised aggregation as checks.

Welfare: B_i(eps) = s_i * n_i * mean_d( omega_id(eps) * e_id * Gamma_i ),
e_id = clip(10 a_i - (d-1), 0, 1) (bottom-first, main) or a_i (uniform), a_i = mean agri
employment share / 100, omega from decile means of the latest PIP survey, n_i = N w_i with
population weights (unweighted variant: w_i = 1/N and no n_i scaling).

Run:   python code/31_application_agri.py
       APP_SMOKE=1 python code/31_application_agri.py   (two seeds, scratch output)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from tempfile import gettempdir

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
from application_utils import BLACK, DARK, LIGHT, MID, fig_style, fmt, fmt_se, save_fig, write_table  # noqa: E402

PROC = ROOT / "data" / "processed"
SMOKE = os.environ.get("APP_SMOKE") == "1"
OUT = (Path(gettempdir()) / "climate-adaptation-smoke" / "agriculture"
       if SMOKE else ROOT / "results" / "application")

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
LAGGED = ["ln_gdppc", "internet", "agri_share", "urban", "ln_pop", "ln_fert"]
X_AG = ["ln_gdppc_l1", "internet_l1", "agri_share_l1", "urban_l1", "ln_pop_l1", "readiness",
        "ln_ai_pc_lag", "base_temp", "precip_shock", "ln_fert_l1"]
Z_NAMES = ["ln_gdppc", "gini", "internet", "agri_share", "base_temp", "sov_z"]
TREE_NAMES = ["ln_gdppc", "gini", "internet", "agri_share", "base_temp", "sov_z"]
DEC = [f"y_dec{d}" for d in range(1, 11)]

T0 = time.time()
SUM: dict[str, list[str]] = {}


def log(section: str, msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)
    SUM.setdefault(section, []).append(msg)


# ============================================================ data and fitting
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


def make_frame(p: pd.DataFrame, outcome="dly", S="hotm", T="T_res", sbar="hotm_sd_i"):
    need = [outcome, S, T, sbar, "region", "year"] + X_AG
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
            "years": meta["year"].to_numpy(), "e_hat": e_hat, "seed_ate": seed_ate,
            "fold_country": base.fold_, "countries": base.countries_, "ests": ests}


def seed_stats(agg) -> dict:
    s = agg["seed_ate"]
    q = np.quantile(s, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {"min": q[0], "p25": q[1], "median": q[2], "p75": q[3], "max": q[4],
            "median_agg": float(agg["G"]["dr"].mean()), "mean_agg": float(agg["Gmean"]["dr"].mean())}


def per_seed(ests, units, col, groups, mask=None):
    """Per-seed mean and country-clustered SE of a unit-level score (split-adjustment inputs)."""
    th, se = [], []
    for e in ests:
        x = e.scores_.pivot(index="unit", columns="d", values=col).loc[units, [1]].to_numpy(float)[:, 0]
        if mask is None:
            m_, s_ = cluster_mean_se(x, groups)
        else:
            m_, s_ = cluster_mean_se(x[mask], groups[mask])
        th.append(m_)
        se.append(s_)
    return np.array(th), np.array(se)


def split_adj(th, se, z=1.96):
    """Chernozhukov et al. (2018): theta_med = median_s theta_s; SE_adj = sqrt(median_s(SE_s^2 + (theta_s - theta_med)^2))."""
    tm = float(np.median(th))
    sa = float(np.sqrt(np.median(se ** 2 + (th - tm) ** 2)))
    return tm, sa, tm - z * sa, tm + z * sa


def twoway_se(u, groups, years):
    """Cameron-Gelbach-Miller two-way (country and year) SE of the mean of u."""
    N = len(u)
    Sg = np.bincount(pd.factorize(groups)[0], weights=u)
    St = np.bincount(pd.factorize(years)[0], weights=u)
    V = (np.sum(Sg ** 2) + np.sum(St ** 2) - np.sum(u ** 2)) / N ** 2
    return float(np.sqrt(max(V, 0.0)))


def driscoll_kraay_se(u, years):
    """Driscoll-Kraay SE of the mean of u from yearly sums (Bartlett kernel, L = floor(4 (T/100)^(2/9)))."""
    codes, _ = pd.factorize(years, sort=True)
    h = np.bincount(codes, weights=u)
    T_ = len(h)
    L = int(np.floor(4 * (T_ / 100) ** (2 / 9)))
    omega = float(np.sum(h ** 2))
    for lag in range(1, L + 1):
        omega += 2 * (1 - lag / (L + 1)) * float(np.sum(h[lag:] * h[:-lag]))
    return float(np.sqrt(max(omega, 0.0)) / len(u)), L


def twfe(d: pd.DataFrame, y: str, regs: list[str], fe: str = "both"):
    d = d.dropna(subset=[y] + regs).copy()
    parts = []
    if fe in ("both", "country"):
        parts.append(pd.get_dummies(d["iso3"], drop_first=True, dtype=float))
    if fe in ("both", "year"):
        parts.append(pd.get_dummies(d["year"].astype(str), drop_first=True, dtype=float))
    if fe == "year_region":
        parts.append(pd.get_dummies(d["year"].astype(str), drop_first=True, dtype=float))
        parts.append(pd.get_dummies(d["region"].astype(str), drop_first=True, dtype=float))
    FE = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=d.index)
    X = sm.add_constant(pd.concat([d[regs].astype(float), FE], axis=1), has_constant="add")
    res = sm.OLS(d[y].astype(float).to_numpy(), X.to_numpy(float)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["iso3"])[0]})
    names = list(X.columns)
    out = {r: (float(res.params[names.index(r)]), float(res.bse[names.index(r)]), float(res.pvalues[names.index(r)]))
           for r in regs}
    return out, int(res.nobs), int(d["iso3"].nunique())


# ============================================================ country table and welfare
def policy_country_frame(frame: pd.DataFrame, p: pd.DataFrame, sbar_col="hotm_sd_i") -> pd.DataFrame:
    """Assemble the observed inputs to country allocation rules."""
    g = frame.groupby("iso3")
    c = pd.DataFrame({"ln_gdppc": g["ln_gdppc_l1"].mean(),
                      "gini": g["gini_f"].mean(), "internet": g["internet_l1"].mean(),
                      "agri_share": g["agri_share_l1"].mean(), "base_temp": g["base_temp"].mean(),
                      "ln_kpc_lag": g["ln_kpc_lag"].mean()})
    pg = p.groupby("iso3")
    c["agri"] = pg["agri_empl"].mean() / 100
    c["s_bar"] = pg[sbar_col].first()
    c = c.join(pg[["pop_2023", "emde", "country", "region", *DEC]].first()
               .rename(columns={"pop_2023": "pop", "country": "name"}))
    sovl = p.dropna(subset=["sovereignty"]).groupby("iso3")["sovereignty"].last()
    c["sov"] = sovl.reindex(c.index).fillna(-0.1)
    c["has_inc"] = c[DEC].notna().all(axis=1)
    return c.dropna(subset=[*DEC, "gini", "agri", "s_bar", "pop", "ln_kpc_lag", "ln_gdppc", "base_temp"])


def country_table(agg, p: pd.DataFrame, sbar_col="hotm_sd_i") -> dict:
    m = agg["meta"].assign(Gam=agg["G"]["dr"], rho=agg["G"]["rho"])
    c = policy_country_frame(m, p, sbar_col)
    c = c.join(m.groupby("iso3")[["Gam", "rho"]].mean()).dropna(subset=["Gam", "rho"])
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


def B_of(c, eps: float, which: str = "Gam", expo: str = "bottom", weighted: bool = True):
    """Country benefit. weighted=True: population weights in Atkinson normalisation and n_i scaling.
    weighted=False: equal country weights and no n_i scaling (unweighted objective)."""
    if weighted:
        om = atkinson_weights(c["Inc"], eps, c["w"])
        return benefit_from_gamma(c["e"][expo] * c[which][:, None], om, c["s_bar"], c["w"])
    om = atkinson_weights(c["Inc"], eps, np.full(c["N"], 1.0 / c["N"]))
    return benefit_from_gamma(c["e"][expo] * c[which][:, None], om, c["s_bar"], None)


def exposure_shares(c: pd.DataFrame, k: float, population_weighted: bool = True) -> pd.DataFrame:
    """Share of sample exposure selected when every country has the same slope change."""
    w = c["pop"].to_numpy(float) if population_weighted else np.ones(len(c))
    w = w / w.sum()
    income = c[DEC].to_numpy(float)
    agri = c["agri"].to_numpy(float)
    shock = c["s_bar"].to_numpy(float)
    bottom = np.clip(10 * agri[:, None] - np.arange(len(DEC)), 0, 1)
    uniform = np.broadcast_to(agri[:, None], income.shape)

    def share_inputs(exposure, eps):
        mass = benefit_from_gamma(exposure, atkinson_weights(income, eps, w), shock, w)
        return mass / mass.sum()

    exposure_eps0 = share_inputs(uniform, 0.0)
    bottom_weighted = share_inputs(bottom, 1.0)
    uniform_weighted = share_inputs(uniform, 1.0)
    rules = {
        "Research-first": c["ln_kpc_lag"],
        "Poorest-first": -c["ln_gdppc"],
        "Most-agricultural-first": c["agri"],
    }
    rows = []
    for rule, rank in rules.items():
        selected = topk_policy(rank, k)
        rows.append({
            "rule": rule,
            "countries": int(selected.sum()),
            "emdes": int(c.loc[selected, "emde"].sum()),
            "exposure_eps0_pct": 100 * exposure_eps0[selected].sum(),
            "bottom_eps1_pct": 100 * bottom_weighted[selected].sum(),
            "uniform_eps1_pct": 100 * uniform_weighted[selected].sum(),
        })
    return pd.DataFrame(rows)


def ci_of(contrib, clusters):
    r = cluster_bootstrap(contrib, clusters, n_boot=N_BOOT, seed=SEED)
    return r["estimate"], r["ci"][0], r["ci"][1]


def policy_block(c, k: float, folds: np.ndarray, expo: str = "bottom", weighted: bool = True) -> dict:
    Z = c["Z"]
    iso = c["iso3"]
    N = len(iso)
    m = int(np.floor(k * N + 1e-9))
    B = {e: B_of(c, e, "Gam", expo, weighted) for e in EPS}
    Bp = {e: B_of(c, e, "rho", expo, weighted) for e in EPS}
    tree = {e: search_tree(Z, B[e], k, depth=2, feature_names=TREE_NAMES) for e in EPS}
    pi_full = {e: tree[e].predict(Z) for e in EPS}
    pi_cf = {e: crossfit_policy(Z, B[e], k, folds)[0] for e in EPS}
    pi_cfp = {e: crossfit_policy(Z, Bp[e], k, folds)[0] for e in EPS}
    comp = {"capable": topk_policy(c["ln_kpc"], k).astype(float),
            "poorest": topk_policy(-c["gdppc"], k).astype(float),
            "agri": topk_policy(c["agri"], k).astype(float)}
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
            "comp": comp, "p_rand": p_rand}


def treated_set(c, expo="bottom", eps=1.0, k=K_MAIN, weighted=True):
    Z = c["Z"]
    B = B_of(c, eps, "Gam", expo, weighted)
    tr = search_tree(Z, B, k, depth=2, feature_names=TREE_NAMES)
    return set(c["iso3"][tr.predict(Z)])


def jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else np.nan


def leaf_report(tree, c, B, label):
    """Leaves of a depth-2 tree: condition, treated flag, countries (ISO3 and names)."""
    Z = c["Z"]
    masks = tree.leaf_masks(Z)
    nm = TREE_NAMES

    def cond(s):
        return "none" if s is None else f"{nm[s[0]]} <= {s[1]:.4g}"

    rt, lf, rg = cond(tree.root), cond(tree.left), cond(tree.right)
    texts = [f"root [{rt}] AND left [{lf}]", f"root [{rt}] AND NOT left [{lf}]",
             f"NOT root [{rt}] AND right [{rg}]", f"NOT root [{rt}] AND NOT right [{rg}]"]
    labs = ["LL", "LR", "RL", "RR"]
    rows = []
    for j, (mk, tr) in enumerate(zip(masks, tree.treated_leaves)):
        members = list(zip(c["iso3"][mk], c["name"][mk]))
        rows.append({"tree": label, "leaf": labs[j], "condition": texts[j], "treated": "yes" if tr else "no",
                     "n countries": int(mk.sum()),
                     "sum B (leaf)": fmt(float(np.sum(B[mk])), 4),
                     "countries": "; ".join(f"{i} {n}" for i, n in members) if members else "--"})
    return rows, masks


# ============================================================ main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_style()
    p = load_panel()
    frame, rc = make_frame(p)
    X = xlist(rc)
    log("sample", f"panel rows {len(p)}, countries {p.iso3.nunique()}; analysis rows {len(frame)}, countries "
                  f"{frame.iso3.nunique()}, years {int(frame.year.min())}-{int(frame.year.max())}; "
                  f"share T_res=1 {frame['T'].mean():.3f}; EMDE countries {frame[frame.emde == 1].iso3.nunique()}")
    log("controls", "CONTROL TIMING CHECK (before revision): ln_gdppc, internet, agri_share, urban, ln_pop, ln_fert were "
                    "CONTEMPORANEOUS; ln_ai_pc_lag was lagged; readiness was already one-year lagged (t-1) in the build; "
                    "base_temp time-invariant; precip_shock contemporaneous (weather).")
    log("controls", "REVISION: ln_gdppc, internet, agri_share, urban, ln_pop and ln_fert enter as one-year lags "
                    "(_l1); readiness is not lagged again (already t-1; a further shift would make it t-2).")
    miss = p[["dly", "hotm", "T_res", "gini", "ln_fert_l1", "agri_empl", "precip_shock"]].isna().mean()
    log("sample", "missing shares (panel): " + ", ".join(f"{k} {v:.3f}" for k, v in miss.items())
        + "; gini dropped from X (missing > 25%), kept in Z via country mean")
    log("sample", "dropped |dly| >= 100 (data errors): 18 country-years (12_build_agri.py)")

    # ---- first stage (unchanged)
    fs_rows = []
    for lab, dd in [("all countries", p), ("EMDE only", p[p.emde == 1]), ("base_temp >= 20", p[p.base_temp >= 20])]:
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

    # ---- EW-DRPL main fit (20 seeds)
    t = time.time()
    ests = fit_seeds(frame, X)
    agg = aggregate(ests, frame)
    ss = seed_stats(agg)
    G = agg["G"]["dr"]
    g = agg["groups"]
    years = agg["years"]
    ate, se = cluster_mean_se(G, g)
    pd.DataFrame({"fold_seed": FOLD_SEEDS, "ate": agg["seed_ate"]}).to_csv(OUT / "agri_seed_stability.csv", index=False)
    log("fit", f"EW-DRPL over {len(FOLD_SEEDS)} fold seeds in {time.time() - t:.0f}s; X = {len(X)} covariates (lagged controls)")
    log("ate", f"ATE of rho (median-over-seeds Gamma, mean over rows) = {ate:.3f} (cluster SE {se:.3f}; N {len(G)}, countries {len(np.unique(g))})")
    log("ate", f"seed distribution of ATE: min {ss['min']:.3f}, p25 {ss['p25']:.3f}, median {ss['median']:.3f}, p75 {ss['p75']:.3f}, max {ss['max']:.3f}")

    # split-adjusted inference (Chernozhukov et al. 2018)
    th_d, se_d = per_seed(agg["ests"], agg["units"], "gamma_dr", g)
    tm_d, sa_d, lo_d, hi_d = split_adj(th_d, se_d)
    t1 = agg["meta"]["T"].to_numpy() == 1
    th_p1, se_p1 = per_seed(agg["ests"], agg["units"], "psi_star", g, mask=t1)
    th_p0, se_p0 = per_seed(agg["ests"], agg["units"], "psi_star", g, mask=~t1)
    tm1, sa1, lo1, hi1 = split_adj(th_p1, se_p1)
    tm0, sa0, lo0, hi0 = split_adj(th_p0, se_p0)
    log("split", f"split-adjusted ATE: median theta {tm_d:.3f}, SE_adj {sa_d:.3f}, 95% CI [{lo_d:.3f}, {hi_d:.3f}] "
                 f"(per-seed theta range {th_d.min():.3f} to {th_d.max():.3f}; per-seed SE median {np.median(se_d):.3f})")
    log("split", f"split-adjusted beta(T=1) (psi*): {tm1:.3f}, SE_adj {sa1:.3f}, CI [{lo1:.3f}, {hi1:.3f}]; "
                 f"beta(T=0): {tm0:.3f}, SE_adj {sa0:.3f}, CI [{lo0:.3f}, {hi0:.3f}]")
    RES_split = {"ate": (tm_d, sa_d, lo_d, hi_d), "b1": (tm1, sa1, lo1, hi1), "b0": (tm0, sa0, lo0, hi0)}

    # other inference: mean, winsorised, two-way, Driscoll-Kraay
    gm = agg["Gmean"]["dr"]
    ate_mean, se_mean = cluster_mean_se(gm, g)
    q1, q99 = np.quantile(G, [0.01, 0.99])
    Gw = np.clip(G, q1, q99)
    ate_w, se_w = cluster_mean_se(Gw, g)
    u = G - G.mean()
    se_2w = twoway_se(u, g, years)
    se_dk, L_dk = driscoll_kraay_se(u, years)
    inf_rows = [
        {"estimator": "median over seeds (main), cluster by country", "estimate": fmt(ate, 3), "SE": fmt(se, 3), "95% CI": f"[{ate - 1.96 * se:.3f}, {ate + 1.96 * se:.3f}]"},
        {"estimator": "split-adjusted (median theta over seeds)", "estimate": fmt(tm_d, 3), "SE": fmt(sa_d, 3), "95% CI": f"[{lo_d:.3f}, {hi_d:.3f}]"},
        {"estimator": "mean over seeds, cluster by country", "estimate": fmt(ate_mean, 3), "SE": fmt(se_mean, 3), "95% CI": f"[{ate_mean - 1.96 * se_mean:.3f}, {ate_mean + 1.96 * se_mean:.3f}]"},
        {"estimator": "median over seeds, winsorised 1/99 pct of Gamma", "estimate": fmt(ate_w, 3), "SE": fmt(se_w, 3), "95% CI": f"[{ate_w - 1.96 * se_w:.3f}, {ate_w + 1.96 * se_w:.3f}]"},
        {"estimator": "median over seeds, two-way cluster (country, year)", "estimate": fmt(ate, 3), "SE": fmt(se_2w, 3), "95% CI": f"[{ate - 1.96 * se_2w:.3f}, {ate + 1.96 * se_2w:.3f}]"},
        {"estimator": f"median over seeds, Driscoll-Kraay (lag {L_dk})", "estimate": fmt(ate, 3), "SE": fmt(se_dk, 3), "95% CI": f"[{ate - 1.96 * se_dk:.3f}, {ate + 1.96 * se_dk:.3f}]"},
    ]
    write_table(pd.DataFrame(inf_rows), OUT / "agri_inference",
                caption="ATE of the resilience score under alternative aggregation and inference.",
                label="tab:agri_inference", notes="Split-adjusted SE per Chernozhukov et al. (2018) over 20 fold seeds. Two-way SE: Cameron-Gelbach-Miller (country, year). Driscoll-Kraay on yearly sums with Bartlett kernel.",
                align="lrrr")
    log("inference", f"mean-over-seeds ATE {ate_mean:.3f} (SE {se_mean:.3f}); winsorised median ATE {ate_w:.3f} (SE {se_w:.3f}); "
                     f"winsorisation bounds {q1:.3f}, {q99:.3f}")
    log("inference", f"two-way cluster SE {se_2w:.3f}; Driscoll-Kraay SE {se_dk:.3f} (lag {L_dk}); cluster SE {se:.3f}")
    write_table(pd.DataFrame([
        {"quantity": "beta(T=1): mean psi* | T_res=1, median over seeds", "estimate": fmt(tm1, 3), "SE_adj": fmt(sa1, 3), "95% CI": f"[{lo1:.3f}, {hi1:.3f}]"},
        {"quantity": "beta(T=0): mean psi* | T_res=0, median over seeds", "estimate": fmt(tm0, 3), "SE_adj": fmt(sa0, 3), "95% CI": f"[{lo0:.3f}, {hi0:.3f}]"},
        {"quantity": "ATE of rho, median over seeds", "estimate": fmt(tm_d, 3), "SE_adj": fmt(sa_d, 3), "95% CI": f"[{lo_d:.3f}, {hi_d:.3f}]"},
    ]), OUT / "agri_split_adjusted", caption="Split-adjusted inference over 20 fold seeds (Chernozhukov et al. 2018).",
        label="tab:agri_split", notes="theta_s and SE_s are per seed (country-clustered); SE_adj = sqrt(median_s(SE_s^2 + (theta_s - median theta)^2)).", align="lrrr")

    # ATE and beta table (current)
    ate_row = [{"quantity": "ATE of rho (mean Gamma, median over seeds)", "estimate": fmt_se(ate, se, 3), "N": str(len(G)), "countries": str(len(np.unique(g)))},
               {"quantity": "mean psi* | T_res = 1 (beta, T=1)", "estimate": fmt_se(float(np.mean(agg["G"]["psi"][t1])), cluster_mean_se(agg["G"]["psi"][t1], g[t1])[1], 3), "N": str(int(t1.sum())), "countries": "--"},
               {"quantity": "mean psi* | T_res = 0 (beta, T=0)", "estimate": fmt_se(float(np.mean(agg["G"]["psi"][~t1])), cluster_mean_se(agg["G"]["psi"][~t1], g[~t1])[1], 3), "N": str(int((~t1).sum())), "countries": "--"},
               {"quantity": "seed ATE min / p25 / median / p75 / max", "estimate": " / ".join(fmt(ss[k_], 3) for k_ in ["min", "p25", "median", "p75", "max"]), "N": "--", "countries": "--"}]
    write_table(pd.DataFrame(ate_row), OUT / "agri_ate", caption="Average resilience effect on cereal-yield growth and mean marginal damage by capability status.",
                label="tab:agri_ate", notes="Country-clustered SE. Gamma = median across 20 fold seeds (main). psi* is the orthogonal slope score; beta by T is its mean.", align="lrrr")
    log("ate", f"beta(T=1) {float(np.mean(agg['G']['psi'][t1])):.3f}; beta(T=0) {float(np.mean(agg['G']['psi'][~t1])):.3f}; overall mean psi* {float(np.mean(agg['G']['psi'])):.3f}")
    log("ate", f"Gamma tails: p50 |Gamma| {np.quantile(np.abs(G), 0.5):.3f}, p95 {np.quantile(np.abs(G), 0.95):.3f}, "
               f"p99 {np.quantile(np.abs(G), 0.99):.3f}, max {np.abs(G).max():.2f}")

    # ---- damage reconciliation
    rec = []
    a_res, a_n, a_c = twfe(p.assign(**{}), "dly", ["hotm"], fe="both")  # placeholder, replaced below on the estimation sample
    est_sample = frame.copy()
    est_sample["hotm"] = est_sample["S_spell"]
    a_res, a_n, a_c = twfe(est_sample, "dly", ["S_spell"], fe="both")
    b_res, b_n, b_c = twfe(est_sample, "dly", ["S_spell"] + X_AG, fe="year_region")
    rec.append({"specification": "(a) TWFE dly on hotm, country + year FE (estimation sample)", "beta": fmt_se(a_res["S_spell"][0], a_res["S_spell"][1], 3), "N": str(a_n), "countries": str(a_c)})
    rec.append({"specification": "(b) year FE + region FE + X (no country FE), estimation sample", "beta": fmt_se(b_res["S_spell"][0], b_res["S_spell"][1], 3), "N": str(b_n), "countries": str(b_c)})
    rec.append({"specification": "(c) EW-DRPL: mean psi* overall (median over seeds), no FE", "beta": fmt_se(float(np.mean(agg["G"]["psi"])), cluster_mean_se(agg["G"]["psi"], g)[1], 3), "N": str(len(G)), "countries": str(len(np.unique(g)))})
    rec.append({"specification": "(c') EW-DRPL: split-adjusted mean psi* overall", "beta": fmt(float(np.median(per_seed(agg["ests"], agg["units"], "psi_star", g)[0])), 3), "N": str(len(G)), "countries": str(len(np.unique(g)))})
    write_table(pd.DataFrame(rec), OUT / "agri_damage_reconciliation", caption="Damage reconciliation: TWFE first stage versus the orthogonal EW-DRPL slope.",
                label="tab:agri_recon", notes="All TWFE rows use the 3,338-row estimation sample with the X of the main fit; (a) controls for country and year fixed effects only, (b) year and region fixed effects with X (lagged) and no country fixed effects, (c) mean of the median orthogonal score psi*.", align="lrrr")
    log("reconciliation", f"(a) TWFE country+year FE on estimation sample: {a_res['S_spell'][0]:.3f} (SE {a_res['S_spell'][1]:.3f}), N {a_n}")
    log("reconciliation", f"(b) year+region FE + X (no country FE): {b_res['S_spell'][0]:.3f} (SE {b_res['S_spell'][1]:.3f}), N {b_n}")
    log("reconciliation", f"(c) mean psi* overall (median over seeds): {float(np.mean(agg['G']['psi'])):.3f}; split-adjusted {float(np.median(per_seed(agg['ests'], agg['units'], 'psi_star', g)[0])):.3f}")

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

    # ---- BLP / GATES / CLAN
    Z_ROW = ["ln_gdppc_l1", "gini", "internet_l1", "agri_share_l1", "base_temp", "sov_z"]
    Zr = agg["meta"][Z_ROW].copy()
    n_gini_fill = int(agg["meta"]["gini_f"].isna().sum())
    Zr["gini"] = agg["meta"]["gini_f"].fillna(agg["meta"]["gini_f"].mean())
    Zm = Zr.to_numpy(float)
    Zs = (Zm - Zm.mean(0)) / Zm.std(0, ddof=1)
    b = blp(G, Zs, g, Z_NAMES)
    rho = agg["G"]["rho"]
    gt, gd = gates(rho, G, g, 5)
    ct = clan(Zm, rho, g, Z_NAMES, 5)
    write_table(pd.DataFrame({"term": b.term, "coef": b.coef.map(lambda v: fmt(v, 3)), "se": b.se.map(lambda v: fmt(v, 3)),
                              "p": b.p.map(lambda v: fmt(v, 3))}), OUT / "agri_blp",
                caption="BLP of the median Gamma on standardised Z (ln GDP pc, Gini, internet, agriculture share, baseline temperature, sovereignty).",
                label="tab:agri_blp", notes=f"Country-clustered SE. Z standardised over country-years; Gini filled with the sample mean for {n_gini_fill} rows missing it in all years.")
    gt_out = pd.DataFrame({"quintile of predicted rho": gt["group"].astype(int).to_numpy(), "mean Gamma": gt["mean"].map(lambda v: fmt(v, 3)).to_numpy(),
                           "se": gt["se"].map(lambda v: fmt(v, 3)).to_numpy(), "n": gt["n"].map(lambda v: fmt(v, 0)).to_numpy()})
    gt_out.loc[len(gt_out)] = ["top minus bottom", fmt(gd["top_minus_bottom"], 3), fmt(gd["se"], 3), "--"]
    write_table(gt_out, OUT / "agri_gates_clan", caption="GATES: mean median-Gamma by quintile of out-of-fold predicted rho (plug-in m1 - m0).",
                label="tab:agri_gates", notes="Quintile 1 = lowest predicted resilience.")
    ct_out = ct.copy()
    for c_ in ct_out.columns:
        if c_ != "feature":
            ct_out[c_] = ct_out[c_].map(lambda v: fmt(v, 2))
    write_table(ct_out, OUT / "agri_gates_clan_covariates", caption="CLAN: mean of each Z variable by quintile of predicted rho.",
                label="tab:agri_clan", notes="q1 = lowest predicted resilience; top_minus_bottom = q5 - q1.")
    log("hetero", "BLP: " + "; ".join(f"{r.term} {r.coef:.3f} (SE {r.se:.3f}, p {r.p:.3f})" for r in b.itertuples()))
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

    # ---- baseline TWFE with lagged controls
    p2 = p.copy()
    p2["hotm_x_T"] = p2["hotm"] * p2["T_res"]
    ctrl = ["ln_gdppc_l1", "internet_l1", "agri_share_l1", "urban_l1", "ln_pop_l1", "readiness", "ln_ai_pc_lag", "precip_shock", "ln_fert_l1"]
    rb1, n1, c1 = twfe(p2, "dly", ["hotm", "hotm_x_T", "T_res"] + ctrl)
    rb2, n2, c2 = twfe(p2, "agri_growth_v", ["hotm", "hotm_x_T", "T_res"] + ctrl)
    base_rows = []
    for lab, kk in [("hotm", "hotm"), ("hotm x T_res", "hotm_x_T"), ("T_res (main effect)", "T_res"), ("ln GDP pc (lag)", "ln_gdppc_l1"), ("internet (lag)", "internet_l1")]:
        base_rows.append({"term": lab, "dly (cereal yield growth)": fmt_se(rb1[kk][0], rb1[kk][1], 3), "agri VA growth": fmt_se(rb2[kk][0], rb2[kk][1], 3)})
    base_rows.append({"term": "N", "dly (cereal yield growth)": str(n1), "agri VA growth": str(n2)})
    base_rows.append({"term": "countries", "dly (cereal yield growth)": str(c1), "agri VA growth": str(c2)})
    write_table(pd.DataFrame(base_rows), OUT / "agri_baseline",
                caption="Baseline TWFE: country and year fixed effects; controls are one-year lags of ln GDP pc, internet, agriculture share, urban, ln population, ln fertilizer, readiness, ln AI pc (lag), precipitation shock.",
                label="tab:agri_baseline", notes="Country-clustered SE. Gini omitted (missing 25.7%).", align="lcc")
    log("baseline", f"TWFE dly (lagged controls): hotm {rb1['hotm'][0]:.3f} (SE {rb1['hotm'][1]:.3f}); hotm x T_res {rb1['hotm_x_T'][0]:.3f} "
                    f"(SE {rb1['hotm_x_T'][1]:.3f}); T_res {rb1['T_res'][0]:.3f} (SE {rb1['T_res'][1]:.3f}); N {n1}")
    log("baseline", f"TWFE agri VA growth (lagged controls): hotm {rb2['hotm'][0]:.3f} (SE {rb2['hotm'][1]:.3f}); hotm x T_res {rb2['hotm_x_T'][0]:.3f} (SE {rb2['hotm_x_T'][1]:.3f}); N {n2}")

    # ---- policy (main: bottom-first, population-weighted)
    c = country_table(agg, p)
    N = c["N"]
    folds_c = make_folds(c["iso3"], K=K_FOLDS, seed=SEED)
    pos1 = int(np.sum(B_of(c, 1.0) > 0))
    pos2 = int(np.sum(B_of(c, 2.0) > 0))
    log("policy", f"policy sample N = {N} countries (EMDE {int(c['emde'].sum())}); budget floor(0.3 N) = {int(np.floor(K_MAIN * N))}; "
                  f"ASEAN in sample: " + ", ".join(sorted(set(c["iso3"]) & set(ASEAN))))
    log("policy", f"countries with positive benefit B_i: eps=1 {pos1}; eps=2 {pos2}")
    pol = policy_block(c, K_MAIN, folds_c, "bottom", True)
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
    for r in eq.itertuples():
        log("policy", f"eps {r.eps}: equity gain {r.gain:.4f} [{r.gain_lo:.4f}, {r.gain_hi:.4f}]; equity price {r.price:.4f} [{r.price_lo:.4f}, {r.price_hi:.4f}]")
    # (d) differences in held-out value
    diff_rows = []
    for e in EPS_EQ:
        Be = pol["B"][e]
        pe = pol["pi_cf"][e].astype(float)
        cap = pol["comp"]["capable"]
        poor = pol["comp"]["poorest"]
        d1 = ci_of((pe - poor) * Be, c["iso3"])
        d2 = ci_of((cap - poor) * Be, c["iso3"])
        diff_rows.append({"eps": fmt(e, 1), "difference": "EW-DRPL(eps) minus poorest-first", "estimate": f"{d1[0]:.4f}", "95% CI": f"[{d1[1]:.4f}, {d1[2]:.4f}]"})
        diff_rows.append({"eps": fmt(e, 1), "difference": "capable-first minus poorest-first", "estimate": f"{d2[0]:.4f}", "95% CI": f"[{d2[1]:.4f}, {d2[2]:.4f}]"})
        log("differences", f"eps {e}: EW-DRPL minus poorest-first {d1[0]:.4f} [{d1[1]:.4f}, {d1[2]:.4f}]; capable-first minus poorest-first {d2[0]:.4f} [{d2[1]:.4f}, {d2[2]:.4f}]")
    write_table(pd.DataFrame(diff_rows), OUT / "agri_policy_differences", caption="Bootstrap differences in held-out welfare value (budget k = 0.3).",
                label="tab:agri_diff", notes="Country-cluster bootstrap, B = 500; each difference evaluated under the same eps benefit B_i(eps).", align="llll")

    # (f) and treated lists and leaves
    treated_rows = []
    for i in range(N):
        flags = {e: bool(pol["pi_full"][e][i]) for e in EPS}
        if any(flags.values()):
            treated_rows.append({"ISO3": c["iso3"][i], "country": c["name"][i], "region": c["region"][i],
                                 "EMDE": "yes" if c["emde"][i] else "no", "ASEAN": "yes" if c["iso3"][i] in ASEAN else "no",
                                 "agri share": fmt(c["agri"][i], 3), "B(eps=1)": fmt(pol["B"][1.0][i], 4),
                                 **{f"eps {fmt(e, 1)}": "x" if flags[e] else "" for e in EPS}})
    write_table(pd.DataFrame(treated_rows).sort_values("ISO3"), OUT / "agri_treated",
                caption="Countries treated under the full-sample depth-2 tree for each eps (k = 0.3, bottom-first exposure).",
                label="tab:agri_treated", notes="x = treated. ASEAN: VNM, IDN, PHL, THA, MYS, KHM, LAO, MMR.", align="lllllrrcccc")
    leaf_all = []
    for e in [1.0, 2.0]:
        rows, masks = leaf_report(pol["tree"][e], c, pol["B"][e], f"eps={fmt(e, 1)}")
        leaf_all += rows
        log("leaves", f"eps {e} tree: " + " | ".join(f"{r['leaf']} [{r['treated']}] n={r['n countries']}: {r['countries']}" for r in rows))
    pd.DataFrame(leaf_all).to_csv(OUT / "agri_leaves.csv", index=False)
    write_table(pd.DataFrame(leaf_all)[["tree", "leaf", "condition", "treated", "n countries", "countries"]], OUT / "agri_leaves",
                caption="Leaves of the depth-2 policy trees (k = 0.3) with treated status and member countries.",
                label="tab:agri_leaves", notes="LL, LR, RL, RR are the four leaves in the order of the tree search. Treated leaves form the policy.", align="llllrp{9cm}")
    if "VNM" in list(c["iso3"]):
        vi = list(c["iso3"]).index("VNM")
        vnm = {e: bool(pol["pi_full"][e][vi]) for e in EPS}
        log("policy", "Viet Nam treated (full-sample tree): " + ", ".join(f"eps {fmt(e, 1)}: {'yes' if vnm[e] else 'no'}" for e in EPS))
        Zv = c["Z"][vi]
        tree2 = pol["tree"][2.0]
        _, masks2 = leaf_report(tree2, c, pol["B"][2.0], "eps=2")
        leaf_idx = [j for j in range(4) if masks2[j][vi]][0]
        log("leaves", f"Viet Nam eps=2 leaf: {['LL', 'LR', 'RL', 'RR'][leaf_idx]} (treated {tree2.treated_leaves[leaf_idx]}); "
                      f"Z: ln GDP pc {Zv[0]:.3f}, gini {Zv[1]:.2f}, internet {Zv[2]:.3f}, agri share {Zv[3]:.2f}, base temp {Zv[4]:.2f}, sov_z {Zv[5]:.3f}")
        log("leaves", f"Viet Nam B(eps=2) {pol['B'][2.0][vi]:.5f}; B(eps=1) {pol['B'][1.0][vi]:.5f}; agri share {c['agri'][vi]:.3f}")
    else:
        log("leaves", "Viet Nam not in policy sample")
    rules = []
    for e in EPS:
        own = v[(v.method.str.startswith("EW-DRPL")) & (v.eps == e)].iloc[0]
        rules.append({"eps": fmt(e, 1), "rule (full sample, depth 2)": pol["tree"][e].describe(), "treated": int(pol["pi_full"][e].sum()),
                      "held-out value": fmt(own.value, 4)})
    write_table(pd.DataFrame(rules), OUT / "agri_policy_trees", caption="Learned depth-2 tree policies for agricultural-resilience benefits (k = 0.3).",
                label="tab:agri_trees", notes="Rules use Z <= threshold; treated-leaf flags (LL, LR, RL, RR) in the text.", align="lp{8cm}rr")
    for e in EPS:
        log("policy", f"eps {e} rule: {pol['tree'][e].describe()}; treated {int(pol['pi_full'][e].sum())}")
        log("policy", f"eps {e} treated ({int(pol['pi_full'][e].sum())}): " + ", ".join(sorted(c["iso3"][pol["pi_full"][e]])))

    # (e) budget under-use
    B1 = pol["B"][1.0]
    unc1 = float(np.sum(B1[B1 > 0]) / N)
    tv1 = float(np.sum(B1[pol["pi_full"][1.0]]) / N)
    log("policy", f"budget use: m = {pol['m']}; eps=1 tree treats {int(pol['pi_full'][1.0].sum())}; countries with B>0 = {pos1}; "
                  f"value of tree {tv1:.5f} vs value of treating all B>0 countries {unc1:.5f} (same objective, no budget)")
    leaf_sizes = [int(m_.sum()) for m_ in leaf_report(pol["tree"][1.0], c, B1, "x")[1]]
    log("policy", f"eps=1 tree leaf sizes (LL, LR, RL, RR): {leaf_sizes}; treated leaves: {pol['tree'][1.0].treated_leaves}")

    # (c) unweighted objective
    pu_w = policy_block(c, K_MAIN, folds_c, "bottom", False)
    vw = pu_w["values"]
    write_table(pd.DataFrame({"method": vw.method, "eps": vw.eps.map(lambda z: fmt(z, 1)),
                              "value [95% CI]": [f"{r.value:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]" for r in vw.itertuples()]}),
                OUT / "agri_unweighted_values", caption="Unweighted objective (equal country weights, no population scaling; budget floor(0.3 N) countries): held-out values.",
                label="tab:agri_unw", notes="Atkinson weights normalised over countries; B_i without n_i. 95% CIs: country-cluster bootstrap.", align="llr")
    for e in EPS:
        tr = pu_w["tree"][e]
        tl = list(c["iso3"][pu_w["pi_full"][e]])
        log("unweighted", f"eps {e}: tree {tr.describe()}; treated {len(tl)}: " + ", ".join(sorted(tl)))
        own = vw[(vw.method.str.startswith("EW-DRPL")) & (vw.eps == e)].iloc[0]
        log("unweighted", f"eps {e}: held-out value {own.value:.4f} [{own.ci_lo:.4f}, {own.ci_hi:.4f}]")
    tl_unw = {e: set(c["iso3"][pu_w["pi_full"][e]]) for e in EPS}
    for e in EPS:
        log("unweighted", f"eps {e}: Jaccard vs weighted main rule {jacc(tl_unw[e], set(c['iso3'][pol['pi_full'][e]])):.3f}")

    # (b) uniform exposure at eps=2 (and all eps)
    pu = policy_block(c, K_MAIN, folds_c, "uniform", True)
    main2 = set(c["iso3"][pol["pi_full"][2.0]])
    unif2 = set(c["iso3"][pu["pi_full"][2.0]])
    own_u = pu["values"][(pu["values"].method.str.startswith("EW-DRPL")) & (pu["values"].eps == 2.0)].iloc[0]
    log("uniform", f"uniform exposure eps=2: treated {len(unif2)}; Jaccard vs main eps=2 {jacc(unif2, main2):.3f}; "
                   f"held-out value {own_u.value:.4f} [{own_u.ci_lo:.4f}, {own_u.ci_hi:.4f}]; tree {pu['tree'][2.0].describe()}")
    log("uniform", "uniform exposure treated (eps=2): " + ", ".join(sorted(unif2)))

    # budget sensitivity
    ks_rows = []
    for kk in K_SENS:
        pk = policy_block(c, kk, folds_c, "bottom", True)
        for e in EPS:
            own = pk["values"][(pk["values"].method.str.startswith("EW-DRPL")) & (pk["values"].eps == e)].iloc[0]
            ks_rows.append({"k": fmt(kk, 1), "eps": fmt(e, 1), "treated": int(pk["pi_full"][e].sum()), "rule": pk["tree"][e].describe(), "held-out value": fmt(own.value, 4)})
            log("budget", f"k={kk} eps={e}: treated {int(pk['pi_full'][e].sum())}; {pk['tree'][e].describe()}; held-out {own.value:.4f}")
        for r in pk["equity"].itertuples():
            log("budget", f"k={kk} eps={r.eps}: equity gain {r.gain:.4f} [{r.gain_lo:.4f}, {r.gain_hi:.4f}]; price {r.price:.4f} [{r.price_lo:.4f}, {r.price_hi:.4f}]")
    write_table(pd.DataFrame(ks_rows), OUT / "agri_budget_sensitivity", caption="Sensitivity of agricultural policy trees and held-out values to the budget share k.",
                label="tab:agri_budget", notes="Held-out value under the rule's own eps.", align="lllp{7cm}r")

    # ---- sensitivity (a)-(h)
    main1 = set(c["iso3"][pol["pi_full"][1.0]])
    sens = [{"variant": "main (T_res, hotm, dly, bottom exposure)", "rows": str(len(G)), "countries": str(len(np.unique(g))),
             "ATE (SE)": fmt_se(ate, se, 3), "eps=1 treated": str(len(main1)), "Jaccard eps=1": "1.000", "eps=2 treated": str(len(main2)), "Jaccard eps=2": "1.000"}]
    ga, sa = cluster_mean_se(agg["G"]["raw"], g)
    tmp = dict(c)
    tmp["Gam"] = pd.Series(agg["G"]["raw"]).groupby(g).mean().reindex(c["iso3"]).to_numpy()
    a1 = treated_set(tmp, eps=1.0)
    a2 = treated_set(tmp, eps=2.0)
    sens.append({"variant": "(a) raw psi score", "rows": str(len(G)), "countries": str(len(np.unique(g))), "ATE (SE)": fmt_se(ga, sa, 3),
                 "eps=1 treated": str(len(a1)), "Jaccard eps=1": f"{jacc(a1, main1):.3f}", "eps=2 treated": str(len(a2)), "Jaccard eps=2": f"{jacc(a2, main2):.3f}"})
    log("sensitivity", f"(a) raw psi: ATE {ga:.3f} (SE {sa:.3f}); Jaccard eps=1 {jacc(a1, main1):.3f}, eps=2 {jacc(a2, main2):.3f}")
    specs = [
        ("(b) propensity trimming [0.05, 0.95]", frame, None, None, (0.05, 0.95), "dly", "hotm", "hotm_sd_i", "T_res"),
        ("(c) annual shock instead of hotm", frame, None, None, (0.1, 0.9), "dly", "shock", "shock_sd_i", "T_res"),
        ("(d) random forest nuisances", frame, RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=1),
         RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=1), (0.1, 0.9), "dly", "hotm", "hotm_sd_i", "T_res"),
        ("(e) EMDE-only sample", None, None, None, (0.1, 0.9), "dly", "hotm", "hotm_sd_i", "T_res"),
        ("(f) T_stock instead of T_res", frame, None, None, (0.1, 0.9), "dly", "hotm", "hotm_sd_i", "T_stock"),
        ("(g) agricultural VA growth outcome", None, None, None, (0.1, 0.9), "agri_growth_v", "hotm", "hotm_sd_i", "T_res"),
    ]
    variant_ests = {}
    for lab, fr, reg_, clf_, eclip, outc, S_, sb, T_ in specs:
        if lab.startswith("(e)"):
            fr2, rc2 = make_frame(p[p.emde == 1], outcome=outc, S=S_, T=T_, sbar=sb)
        elif lab.startswith("(g)"):
            fr2, rc2 = make_frame(p, outcome=outc, S=S_, T=T_, sbar=sb)
        elif lab.startswith("(c)") or lab.startswith("(f)"):
            fr2, rc2 = make_frame(p, outcome=outc, S=S_, T=T_, sbar=sb)
        else:
            fr2, rc2 = fr, rc
        ests2 = fit_seeds(fr2, xlist(rc2), reg=reg_, clf=clf_, e_clip=eclip)
        agg2 = aggregate(ests2, fr2)
        G2 = agg2["G"]["dr"]
        g2 = agg2["groups"]
        a2_, s2_ = cluster_mean_se(G2, g2)
        th2, se2_ = per_seed(ests2, agg2["units"], "gamma_dr", g2)
        tm2, sa2, lo2, hi2 = split_adj(th2, se2_)
        c2 = country_table(agg2, p, sbar_col=sb)
        ts1 = treated_set(c2, eps=1.0)
        ts2 = treated_set(c2, eps=2.0)
        variant_ests[lab] = (tm2, sa2, lo2, hi2, a2_, s2_)
        sens.append({"variant": lab, "rows": str(len(G2)), "countries": str(c2["N"]), "ATE (SE)": fmt_se(a2_, s2_, 3),
                     "eps=1 treated": str(len(ts1)), "Jaccard eps=1": f"{jacc(ts1, main1):.3f}", "eps=2 treated": str(len(ts2)), "Jaccard eps=2": f"{jacc(ts2, main2):.3f}"})
        log("sensitivity", f"{lab}: rows {len(G2)}, countries {c2['N']}; ATE {a2_:.3f} (SE {s2_:.3f}); split-adjusted {tm2:.3f} (SE {sa2:.3f}); "
                           f"Jaccard eps=1 {jacc(ts1, main1):.3f}, eps=2 {jacc(ts2, main2):.3f} (treated {len(ts1)}, {len(ts2)})")
    sens.append({"variant": "(h) uniform exposure e_id = a_i (policy only)", "rows": "--", "countries": str(N), "ATE (SE)": "--",
                 "eps=1 treated": str(len(treated_set(c, "uniform", 1.0))), "Jaccard eps=1": f"{jacc(treated_set(c, 'uniform', 1.0), main1):.3f}",
                 "eps=2 treated": str(len(unif2)), "Jaccard eps=2": f"{jacc(unif2, main2):.3f}"})
    write_table(pd.DataFrame(sens), OUT / "agri_sensitivity",
                caption="Sensitivity of the agricultural application: ATE of rho and Jaccard overlap of the eps = 1 and eps = 2 treated sets with the main rules.",
                label="tab:agri_sensitivity", notes="Jaccard = |A and B| / |A or B| with the main treated sets (k = 0.3). Median-over-seeds ATE with cluster SE; 20 seeds per row (variant h policy only). Variant (g) outcome is agricultural VA growth.", align="lrrrrrrr")
    log("sensitivity", "Jaccard eps=2 by variant: " + "; ".join(f"{r['variant'][:3]} {r['Jaccard eps=2']}" for r in sens))
    if "(g)" in " ".join(r["variant"] for r in sens):
        tm2g, sa2g, lo2g, hi2g, _, _ = variant_ests["(g) agricultural VA growth outcome"]
        log("sensitivity", f"(g) agri VA growth split-adjusted ATE {tm2g:.3f} (SE {sa2g:.3f}), 95% CI [{lo2g:.3f}, {hi2g:.3f}]")
        log("revision", f"agri VA growth outcome split-adjusted ATE {tm2g:.3f}, SE_adj {sa2g:.3f}, 95% CI [{lo2g:.3f}, {hi2g:.3f}]")

    # ---- revision log block
    log("revision", f"ATE (median over seeds, lagged controls): {ate:.3f} (cluster SE {se:.3f}), split-adjusted {tm_d:.3f} (SE_adj {sa_d:.3f}) CI [{lo_d:.3f}, {hi_d:.3f}]")
    log("revision", f"beta(T=1) split-adjusted {tm1:.3f} (SE_adj {sa1:.3f}) CI [{lo1:.3f}, {hi1:.3f}]; beta(T=0) {tm0:.3f} (SE_adj {sa0:.3f}) CI [{lo0:.3f}, {hi0:.3f}]")
    log("revision", f"mean-over-seeds ATE {ate_mean:.3f} (SE {se_mean:.3f}); winsorised ATE {ate_w:.3f} (SE {se_w:.3f}); two-way SE {se_2w:.3f}; Driscoll-Kraay SE {se_dk:.3f} (lag {L_dk})")
    log("revision", f"policy eps=2 treated {int(pol['pi_full'][2.0].sum())}, eps=1 {int(pol['pi_full'][1.0].sum())} (m = {pol['m']})")
    log("revision", f"policy differences: see differences section; unweighted rule Jaccard eps=1 {jacc(tl_unw[1.0], set(c['iso3'][pol['pi_full'][1.0]])):.3f}")

    fig_policy(OUT / "agri_policy_values.csv", "fig_agri_policy_values")
    B1o = B1
    order = np.argsort(-B1o)[:30]
    trt = pol["pi_full"][1.0]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(np.arange(30), B1o[order], color=[BLACK if trt[i] else LIGHT for i in order], edgecolor=DARK, lw=0.4)
    ax.set_xticks(np.arange(30))
    ax.set_xticklabels(c["iso3"][order], rotation=90, fontsize=6)
    ax.set_ylabel(r"benefit $B_i(\varepsilon=1)$")
    ax.axhline(0, color=DARK, lw=0.6)
    ax.set_title("Top 30 countries by eps=1 benefit (black = treated under eps=1, k=0.3)", fontsize=8)
    fig.tight_layout()
    save_fig(fig, OUT / "fig_agri_ranking")

    # first-stage binned figure
    Dm = pd.get_dummies(p["iso3"].astype(str), dtype=float).to_numpy()
    Yr = pd.get_dummies(p["year"].astype(str), dtype=float).to_numpy()
    M = np.column_stack([Dm, Yr])
    d_ = p.dropna(subset=["dly", "hotm"]).index
    Mm = M[d_]
    res_fs = {}
    for var in ["dly", "hotm"]:
        beta, *_ = np.linalg.lstsq(Mm, p.loc[d_, var].to_numpy(float), rcond=None)
        res_fs[var] = p.loc[d_, var].to_numpy(float) - Mm @ beta
    rh, ry = res_fs["hotm"], res_fs["dly"]
    qs = pd.qcut(rh, 20, labels=False, duplicates="drop")
    bins_df = pd.DataFrame({"h": rh, "y": ry, "q": qs}).groupby("q").agg(h=("h", "mean"), y=("y", "mean"), se=("y", lambda s: s.std(ddof=1) / np.sqrt(len(s))))
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.errorbar(bins_df["h"], bins_df["y"], yerr=1.96 * bins_df["se"], fmt="o", color=BLACK, ms=4, capsize=2, lw=0.8)
    slope = float(np.sum(rh * ry) / np.sum(rh ** 2))
    xx = np.linspace(rh.min(), rh.max(), 50)
    ax.plot(xx, slope * xx, color=MID, ls="--", lw=1)
    ax.axhline(0, color=LIGHT, lw=0.6)
    ax.set_xlabel("hottest-month anomaly, residualised (deg C)")
    ax.set_ylabel("cereal-yield growth, residualised (%)")
    ax.set_title(f"first stage (within-FE slope {slope:.2f})", fontsize=8)
    fig.tight_layout()
    save_fig(fig, OUT / "fig_agri_firststage")

    # ---- summary
    lines = ["# Agricultural yield resilience: application results", "",
             f"Outcome dly (100 x log change in cereal yield), hotm exposure, T_res at t-1, one-year lagged controls, {len(FOLD_SEEDS)} fold seeds "
             f"({FOLD_SEEDS[0]}-{FOLD_SEEDS[-1]}), Gamma = median across seeds. k = {K_MAIN}; B = {N_BOOT}; random draws {N_RANDOM}.", ""]
    for sec, msgs in SUM.items():
        if sec == "revision":
            continue
        lines.append(f"## {sec}")
        lines.append("")
        lines += [f"- {m}" for m in msgs]
        lines.append("")
    lines.append("## revision")
    lines.append("")
    lines += [f"- {m}" for m in SUM.get("revision", [])]
    lines.append("")
    (OUT / "application_summary.md").write_text("\n".join(lines))
    log("runtime", f"total runtime {time.time() - T0:.0f}s")
    print(f"Outputs written to {OUT}")


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
        ax.errorbar(sub["est"], y, xerr=[sub["est"] - sub["lo"], sub["hi"] - sub["est"]], fmt="o", color=BLACK, ms=4, capsize=2, lw=1)
        ax.axvline(0, color=LIGHT, lw=0.8)
        ax.set_title(rf"$\varepsilon$ = {e:g}")
        ax.set_xlabel("welfare value")
    axes[0].set_yticks(list(ypos.values()))
    axes[0].set_yticklabels(methods, fontsize=6.5)
    axes[0].invert_yaxis()
    fig.tight_layout()
    save_fig(fig, OUT / stem)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposure-shares", type=float, metavar="BUDGET",
                        help="write descriptive allocation shares for a budget in (0, 1] as CSV")
    parser.add_argument("--equal-country", action="store_true",
                        help="weight each country equally in the exposure comparison")
    args = parser.parse_args()
    if args.equal_country and args.exposure_shares is None:
        parser.error("--equal-country requires --exposure-shares")
    if args.exposure_shares is not None:
        if not 0 < args.exposure_shares <= 1:
            parser.error("--exposure-shares requires a budget in (0, 1]")
        panel = load_panel()
        frame, _ = make_frame(panel)
        shares = exposure_shares(policy_country_frame(frame, panel), args.exposure_shares,
                                 population_weighted=not args.equal_country)
        shares.to_csv(sys.stdout, index=False, float_format="%.6f")
    else:
        main()
