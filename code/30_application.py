"""Empirical application of EW-DRPL: decile layer (main), welfare and policy learning,
country-year layer, classical baselines, cost module (DML), sensitivity and diagnostics.

Run:   python code/30_application.py
       APP_SMOKE=1 python code/30_application.py   (one fold seed, few bootstraps, scratch output)

Treatment: T_stock_it = 1 if the climate-AI knowledge stock per million population at t-1 is
above the cross-country median (60th percentile in the robustness variant). The stock is
K_it = sum_{tau<=t} climai_works_{i,tau} (1 - delta)^(t-tau), delta = 0.15, from 1998, so T is
first available in 1999. Spells use T at y0. Country-year rows use T at t-1 (the value at the
start of the growth year).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from ewdrpl import EWDRPL  # noqa: E402
from ewdrpl.inference import blp, clan, cluster_bootstrap, cluster_mean_se, gates  # noqa: E402
from ewdrpl.nuisance import crossfit_propensity, default_classifier, make_folds  # noqa: E402
from ewdrpl.policy import crossfit_policy, search_tree, topk_policy  # noqa: E402
from ewdrpl.welfare import atkinson_weights, benefit as benefit_from_gamma  # noqa: E402
from application_utils import (  # noqa: E402
    BLACK, DARK, LIGHT, MID, fig_style, fmt, fmt_se, plr_crossfit, plr_estimate, save_fig, write_table,
)

import matplotlib.pyplot as plt  # noqa: E402

PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
SMOKE = os.environ.get("APP_SMOKE") == "1"
OUT = (Path("/tmp/claude-0/-home-user-Artificial-Intelligence-Climate-Change-and-Inclusive-Socio-Economic-Development/"
            "fedb9d6e-2447-5a7d-8860-2639400da4da/scratchpad/app_smoke") if SMOKE
       else ROOT / "results" / "application")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
FOLD_SEEDS = [42] if SMOKE else [42, 43, 44, 45, 46]
DELTA = 0.15
K_MAIN = 0.3
K_SENS = [0.2, 0.5]
EPS = [0.0, 0.5, 1.0, 2.0]
EPS_EQ = [1.0, 2.0]
N_BOOT = 20 if SMOKE else 500
N_RANDOM = 50 if SMOKE else 1000
ASEAN = ["VNM", "IDN", "PHL", "THA", "MYS", "KHM", "LAO", "MMR"]
Z_NAMES = ["ln_gdppc", "gini", "internet", "agri_share", "base_temp", "sov_z"]
X_CORE = ["ln_gdppc", "gini", "internet", "agri_share", "urban", "ln_pop", "readiness",
          "ln_ai_pc_lag", "base_temp", "shock_sd_i"]
X_SPELL = X_CORE + ["y0", "gap"]     # decile-layer X (region dummies appended)
X_CY = X_CORE + ["year"]             # country-year X (region dummies appended)

T0 = time.time()
SUM: dict[str, list[str]] = {}
RES: dict = {}


def log(section: str, msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)
    SUM.setdefault(section, []).append(msg)


# ============================================================ 1. treatment
def build_treatment() -> pd.DataFrame:
    cy = pd.read_csv(PROC / "country_year_v2.csv")
    n_rows = len(cy)
    # Namibia: ISO2 "NA" is read as missing by 11_build_v2.py, so its OpenAlex columns are empty.
    # Raw counts are read here without that parsing and used only to fill climai_works.
    raw = pd.read_csv(RAW / "openalex_v2_counts.csv", keep_default_na=False, na_values=[""])
    iso_map = cy.dropna(subset=["iso2"]).drop_duplicates("iso2").set_index("iso2")["iso3"]
    raw["iso3"] = raw["iso2"].map(iso_map)
    raw.loc[raw["iso2"] == "NA", "iso3"] = "NAM"
    raw = raw.dropna(subset=["iso3"]).drop_duplicates(["iso3", "year"])
    chk = cy[["iso3", "year", "climai_works"]].merge(raw[["iso3", "year", "climai_pubs"]],
                                                     on=["iso3", "year"], how="left")
    other = chk[chk.iso3 != "NAM"]
    mism = int(((other.climai_works - other.climai_pubs).abs() > 1e-9).sum())
    na_before = int(cy.loc[cy.year >= 1998, "climai_works"].isna().sum())
    cy = cy.merge(raw[["iso3", "year", "climai_pubs"]], on=["iso3", "year"], how="left")
    assert len(cy) == n_rows
    cy["climai_works"] = cy["climai_works"].fillna(cy["climai_pubs"])
    na_after = int(cy.loc[cy.year >= 1998, "climai_works"].isna().sum())
    log("treatment", f"climai_works NaN (1998+) before/after recovering Namibia from raw counts: "
                     f"{na_before}/{na_after}; mismatches with raw counts for other countries: {mism}")

    cy["sov_z"] = cy["sovereignty"].fillna(-0.1)   # missing = fewer than 5 climate-AI papers
    cy = cy.sort_values(["iso3", "year"]).reset_index(drop=True)
    span = cy.groupby("iso3").year.agg(["min", "max", "size"])
    assert ((span["max"] - span["min"] + 1) == span["size"]).all()
    cy["K"] = np.nan
    for _, g in cy.groupby("iso3", sort=False):
        K = 0.0
        for i in g.index[g.year >= 1998]:
            K = K * (1 - DELTA) + cy.at[i, "climai_works"]
            cy.at[i, "K"] = K
    cy["kpc"] = cy["K"] / (cy["pop"] / 1e6)
    wide = cy.pivot(index="year", columns="iso3", values="kpc")
    lag = wide.shift(1)                          # row t holds k_pc at t-1
    thr50 = lag.median(axis=1)
    thr60 = lag.quantile(0.6, axis=1)
    T50 = lag.gt(thr50, axis=0).astype(float).where(lag.notna())
    T60 = lag.gt(thr60, axis=0).astype(float).where(lag.notna())

    def stk(w, name):
        return w.stack().rename(name).reset_index()

    cy = cy.merge(stk(lag, "kpc_lag"), on=["year", "iso3"], how="left")
    cy = cy.merge(stk(T50, "T_stock"), on=["year", "iso3"], how="left")
    cy = cy.merge(stk(T60, "T_stock60"), on=["year", "iso3"], how="left")
    cy["ln_kpc_lag"] = np.log1p(cy["kpc_lag"])
    cy["T_prev"] = cy.groupby("iso3")["T_stock"].shift(1)
    n_med = lag.notna().sum(axis=1)
    first = int(cy.dropna(subset=["T_stock"]).year.min())
    log("treatment", f"T_stock first available in {first} (median over {int(n_med.loc[first])} countries); "
                     f"in 2023 the median uses {int(n_med.loc[2023])} countries (of 159)")
    log("treatment", f"threshold in 2010: median of k_pc(t-1) = {thr50.loc[2010]:.3f}, "
                     f"60th percentile = {thr60.loc[2010]:.3f} climate-AI works per million")
    tv = cy.dropna(subset=["T_stock"])
    log("treatment", f"share T_stock = 1 across country-years (1999-2023): {tv.T_stock.mean():.3f} (n = {len(tv)}); "
                     f"share T_stock60 = 1: {cy.dropna(subset=['T_stock60']).T_stock60.mean():.3f}")
    return cy


# ============================================================ 2. spell frame
def build_spells(cy: pd.DataFrame) -> pd.DataFrame:
    sp = pd.read_csv(PROC / "spells_deciles_v2.csv")
    keyc = cy[["iso3", "year", "T_stock", "T_stock60", "ln_kpc_lag"]].rename(columns={"year": "y0"})
    sp = sp.merge(keyc, on=["iso3", "y0"], how="left")
    sp["sov_z"] = sp["sovereignty"].fillna(-0.1)     # missing = fewer than 5 climate-AI papers
    latest = int(cy.year.max())
    sp["pop_latest"] = sp["iso3"].map(cy[cy.year == latest].set_index("iso3")["pop"])
    sp["uid"] = sp["spell_id"]
    return sp


def region_dummies(df: pd.DataFrame, prefix: str = "reg") -> tuple[pd.DataFrame, list[str]]:
    regs = sorted(df["region"].dropna().unique())
    out = pd.DataFrame(index=df.index)
    cols = []
    for j, r in enumerate(regs[1:], start=1):   # first region (alphabetical) is the base
        out[f"{prefix}{j}"] = (df["region"] == r).astype(float)
        cols.append(f"{prefix}{j}")
    return out, cols


def decile_sample(sp: pd.DataFrame, tcol: str = "T_stock", winsor: bool = True) -> pd.DataFrame:
    req = ["Y", "S_spell", tcol, "y_d1", "region"] + X_SPELL
    s = sp.dropna(subset=req).copy()
    rd, rc = region_dummies(s)
    s = pd.concat([s, rd], axis=1)
    s["T"] = s[tcol].astype(float)
    s["s_bar"] = s["shock_sd_i"]          # exposure column (shock_sd_i is also a covariate)
    s["Yw"] = s["Y"]
    if winsor:
        s["Yw"] = s.groupby("d")["Y"].transform(lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
    s = s.sort_values(["spell_id", "d"]).reset_index(drop=True)
    s.attrs["reg_cols"] = rc
    return s


# ============================================================ 3. EW-DRPL fits across fold seeds
def _fit_one(spl, X, outcome, seed, reg, clf, e_clip):
    return EWDRPL(K=5, seed=seed, regressor=reg, classifier=clf, e_clip=e_clip).fit(
        spl, unit="uid", group="iso3", decile="d", outcome=outcome, shock="S_spell", treat="T",
        covariates=X, income="y_d1", exposure="s_bar")


def fit_seeds(spl, X, outcome="Yw", reg=None, clf=None, e_clip=(0.05, 0.95), n_jobs=4):
    return Parallel(n_jobs=n_jobs)(
        delayed(_fit_one)(spl, X, outcome, s, reg, clf, e_clip) for s in FOLD_SEEDS)


def aggregate(ests, spl) -> dict:
    """Average the scores over the fold-seed repetitions."""
    base = ests[0]
    units = np.sort(base.scores_["unit"].unique())
    dl = base.deciles_

    def piv(est, col):
        return est.scores_.pivot(index="unit", columns="d", values=col).loc[units, dl].to_numpy(float)

    G = {}
    for col, key in [("gamma_dr", "dr"), ("gamma_raw", "raw"), ("psi_star", "psi"),
                     ("rho_plugin", "rho")]:
        G[key] = np.mean([piv(e, col) for e in ests], axis=0)
    seed_ate = [float(piv(e, "gamma_dr").mean()) for e in ests]
    meta = spl.drop_duplicates("uid").set_index("uid").loc[units]
    inc = spl.drop_duplicates(["uid", "d"]).pivot(index="uid", columns="d", values="y_d1") \
        .loc[units, dl].to_numpy(float)
    e_hat = base.scores_.groupby("unit")["e_hat"].first().loc[units].to_numpy(float)
    return {"units": units, "dl": dl, "G": G, "meta": meta, "inc": inc,
            "groups": meta["iso3"].to_numpy(), "e_hat": e_hat, "seed_ate": seed_ate,
            "fold_country": base.fold_, "countries": base.countries_}


# ============================================================ 4. decile-layer outputs
def incidence(agg):
    G = agg["G"]["dr"]
    g = agg["groups"]
    t1 = agg["meta"]["T"].to_numpy() == 1
    rows = []
    for j, d in enumerate(agg["dl"]):
        m, se = cluster_mean_se(G[:, j], g)
        m1, s1 = cluster_mean_se(agg["G"]["psi"][t1, j], g[t1])
        m0, s0 = cluster_mean_se(agg["G"]["psi"][~t1, j], g[~t1])
        rows.append({"decile": int(d), "gamma": m, "se": se, "ci_lo": m - 1.96 * se,
                     "ci_hi": m + 1.96 * se, "psi_T1": m1, "se_psi_T1": s1, "psi_T0": m0, "se_psi_T0": s0})
    ma, sa = cluster_mean_se(G.mean(axis=1), g)
    rows.append({"decile": "average", "gamma": ma, "se": sa, "ci_lo": ma - 1.96 * sa,
                 "ci_hi": ma + 1.96 * sa, "psi_T1": np.nan, "se_psi_T1": np.nan,
                 "psi_T0": np.nan, "se_psi_T0": np.nan})
    info = {"n_spells": len(G), "n_countries": len(np.unique(g)), "n_T1": int(t1.sum()),
            "n_T0": int((~t1).sum()), "seed_ate_min": min(agg["seed_ate"]),
            "seed_ate_max": max(agg["seed_ate"])}
    return pd.DataFrame(rows), info


def heterogeneity(agg):
    gbar = agg["G"]["dr"].mean(axis=1)
    rho = agg["G"]["rho"].mean(axis=1)
    g = agg["groups"]
    Z = agg["meta"][Z_NAMES].to_numpy(float)
    Zs = (Z - Z.mean(0)) / Z.std(0, ddof=1)
    return blp(gbar, Zs, g, Z_NAMES), *gates(rho, gbar, g, 5), clan(Z, rho, g, Z_NAMES, 5)


# ============================================================ 5. welfare and policy
def country_frame(agg, which: str = "dr") -> dict:
    meta = agg["meta"].reset_index()
    g = agg["groups"]
    dfG = pd.DataFrame(agg["G"][which])
    dfG["iso3"] = g
    Gc = dfG.groupby("iso3").mean()
    dfP = pd.DataFrame(agg["G"]["rho"])
    dfP["iso3"] = g
    Pc = dfP.groupby("iso3").mean()
    countries = Gc.index.to_numpy()
    last = meta.sort_values(["y0", "y1"]).groupby("iso3").tail(1).set_index("iso3").loc[countries]
    inc = pd.DataFrame(agg["inc"], index=agg["units"]).loc[last["uid"].to_numpy()].to_numpy(float)
    pop = last["pop_latest"].to_numpy(float)
    return {"iso3": countries, "name": last["country"].to_numpy(), "emde": last["emde"].to_numpy(int),
            "region": last["region"].to_numpy(), "Gam": Gc.loc[countries].to_numpy(),
            "Plug": Pc.loc[countries].to_numpy(), "Inc": inc, "Z": last[Z_NAMES].to_numpy(float),
            "s_bar": last["shock_sd_i"].to_numpy(float), "w": pop / pop.sum(), "pop": pop,
            "ln_kpc": last["ln_kpc_lag"].to_numpy(float), "gdppc": last["ln_gdppc"].to_numpy(float)}


def B_of(ctry, eps, mat):
    """B_i(eps) = s_bar_i n_i mean_d(omega_id(eps) mat_id), n_i = N w_i (population scale)."""
    om = atkinson_weights(ctry["Inc"], eps, ctry["w"])
    return benefit_from_gamma(mat, om, ctry["s_bar"], ctry["w"])


def ci_of(contrib, clusters):
    r = cluster_bootstrap(contrib, clusters, n_boot=N_BOOT, seed=SEED)
    return r["estimate"], r["ci"][0], r["ci"][1]


def policy_block(ctry, k: float, folds: np.ndarray) -> dict:
    Z = ctry["Z"]
    iso = ctry["iso3"]
    N = len(iso)
    m = int(np.floor(k * N + 1e-9))
    B = {e: B_of(ctry, e, ctry["Gam"]) for e in EPS}
    Bp = {e: B_of(ctry, e, ctry["Plug"]) for e in EPS}
    tree = {e: search_tree(Z, B[e], k, depth=2, feature_names=Z_NAMES) for e in EPS}
    pi_full = {e: tree[e].predict(Z) for e in EPS}
    pi_cf = {e: crossfit_policy(Z, B[e], k, folds)[0] for e in EPS}
    pi_cfp = {e: crossfit_policy(Z, Bp[e], k, folds)[0] for e in EPS}
    capable = topk_policy(ctry["ln_kpc"], k).astype(float)
    poorest = topk_policy(-ctry["gdppc"], k).astype(float)
    rng = np.random.default_rng(SEED)
    idx = np.argsort(rng.random((N_RANDOM, N)), axis=1)[:, :m]
    p_rand = np.bincount(idx.ravel(), minlength=N) / N_RANDOM   # expected inclusion per country
    methods = {
        "EW-DRPL (own-eps rule, held out)": lambda e: pi_cf[e].astype(float),
        "Utilitarian tree (eps=0, held out)": lambda e: pi_cf[0.0].astype(float),
        "Plug-in tree (held out)": lambda e: pi_cfp[e].astype(float),
        "Capable-first (top 30% by ln k_pc)": lambda e: capable,
        "Poorest-first (bottom 30% by ln GDP pc)": lambda e: poorest,
        "Random 30% (expected over draws)": lambda e: p_rand,
    }
    values = []
    for name, pif in methods.items():
        for e in EPS:
            est, lo, hi = ci_of(pif(e) * B[e], iso)
            values.append({"method": name, "eps": e, "value": est, "ci_lo": lo, "ci_hi": hi})
    matrix = {}
    for e in EPS:
        for e2 in EPS:
            matrix[(e, e2)] = ci_of(pi_cf[e].astype(float) * B[e2], iso)
    equity = []
    for e in EPS_EQ:
        g_ = ci_of((pi_cf[e].astype(float) - pi_cf[0.0].astype(float)) * B[e], iso)
        p_ = ci_of((pi_cf[0.0].astype(float) - pi_cf[e].astype(float)) * B[0.0], iso)
        equity.append({"eps": e, "gain": g_[0], "gain_lo": g_[1], "gain_hi": g_[2],
                       "price": p_[0], "price_lo": p_[1], "price_hi": p_[2]})
    return {"k": k, "N": N, "m": m, "B": B, "tree": tree, "pi_full": pi_full, "pi_cf": pi_cf,
            "values": pd.DataFrame(values), "matrix": matrix, "equity": pd.DataFrame(equity)}


# ============================================================ 6. country-year layer and baseline
def layer1(cy: pd.DataFrame, outcome: str) -> dict:
    c = cy.copy()
    rd, rc = region_dummies(c)
    c = pd.concat([c, rd], axis=1)
    c = c.dropna(subset=[outcome, "shock", "T_prev", "region"] + X_CORE).copy()
    c["uid"] = c["iso3"] + "_" + c["year"].astype(str)
    c["d"] = 1
    c["Y"] = c[outcome]
    c["S_spell"] = c["shock"]
    c["T"] = c["T_prev"].astype(float)
    c["y_d1"] = 1.0
    c["s_bar"] = c["shock_sd_i"]
    ests = fit_seeds(c, X_CY + rc, outcome="Y", n_jobs=4)
    agg = aggregate(ests, c)
    G = agg["G"]["dr"][:, 0]
    g = agg["groups"]
    ate, se = cluster_mean_se(G, g)
    Z = agg["meta"][Z_NAMES].to_numpy(float)
    Zs = (Z - Z.mean(0)) / Z.std(0, ddof=1)
    b = blp(G, Zs, g, Z_NAMES)
    return {"n": len(G), "countries": len(np.unique(g)), "ate": ate, "se": se, "blp": b,
            "frame": c, "seed_ate": agg["seed_ate"]}


def twfe(df: pd.DataFrame, y: str):
    d = df.copy()
    d["shock_x_T"] = d["shock"] * d["T_prev"]
    regs = ["shock", "T_prev", "shock_x_T", "ln_gdppc", "gini", "internet", "agri_share",
            "urban", "ln_pop", "readiness", "ln_ai_pc_lag"]
    FE = pd.concat([pd.get_dummies(d["iso3"], drop_first=True, dtype=float),
                    pd.get_dummies(d["year"].astype(str), drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(pd.concat([d[regs].astype(float), FE], axis=1), has_constant="add")
    res = sm.OLS(d[y].astype(float).to_numpy(), X.to_numpy(float)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["iso3"])[0]})
    names = list(X.columns)
    out = {r: (float(res.params[names.index(r)]), float(res.bse[names.index(r)]),
               float(res.pvalues[names.index(r)])) for r in ["shock", "T_prev", "shock_x_T"]}
    return out, int(res.nobs), int(d["iso3"].nunique())


def pooled_decile(spl: pd.DataFrame, rc: list[str]):
    d = spl.copy()
    d["S_x_T"] = d["S_spell"] * d["T"]
    dd = pd.get_dummies(d["d"].astype(str), prefix="dec", drop_first=True, dtype=float)
    regs = ["S_spell", "S_x_T", "T"] + X_SPELL + rc
    X = sm.add_constant(pd.concat([d[regs].astype(float), dd], axis=1), has_constant="add")
    res = sm.OLS(d["Yw"].astype(float).to_numpy(), X.to_numpy(float)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["iso3"])[0]})
    names = list(X.columns)
    out = {r: (float(res.params[names.index(r)]), float(res.bse[names.index(r)]),
               float(res.pvalues[names.index(r)])) for r in ["S_spell", "S_x_T", "T"]}
    return out, int(res.nobs), int(d["iso3"].nunique())


# ============================================================ 7. cost module (DML)
def cost_module(cy: pd.DataFrame) -> dict:
    c = cy[(cy.year >= 2001) & (cy.year <= 2022)].copy()
    ctrl = ["ln_gdppc", "urban", "ln_pop", "internet", "agri_share", "ren_share", "base_temp", "year"]
    c = c.dropna(subset=["ln_elec_pc", "ln_co2pc", "ln_kpc_lag", "ln_ai_pc_lag"] + ctrl).copy()
    means = c.groupby("iso3")[ctrl].transform("mean").add_prefix("mean_")   # Mundlak device
    c = pd.concat([c, means], axis=1)
    Xm = c[ctrl + list(means.columns)].to_numpy(float)
    groups = c["iso3"].to_numpy()
    folds = make_folds(groups, K=5, seed=SEED)
    reg = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, min_samples_leaf=10, random_state=SEED)
    ren = c["ren_share"].to_numpy(float)
    ren_s = (ren - ren.mean()) / ren.std(ddof=1)
    hi = (ren >= np.median(ren)).astype(float)
    out = {}
    for y in ["ln_elec_pc", "ln_co2pc"]:
        for t in ["ln_kpc_lag", "ln_ai_pc_lag"]:
            ry, rd = plr_crossfit(c[y].to_numpy(float), c[t].to_numpy(float), Xm, folds, reg)
            th, se, psi = plr_estimate(ry, rd, groups)
            b = blp(psi, ren_s[:, None], groups, ["ren_share_std"])
            bh = blp(psi, hi[:, None], groups, ["high_ren"])
            out[(y, t)] = {
                "theta": th, "se": se, "p": float(2 * (1 - norm.cdf(abs(th / se)))),
                "N": len(c), "G": int(len(np.unique(groups))),
                "blp_coef": float(b.loc[1, "coef"]), "blp_se": float(b.loc[1, "se"]),
                "psi_low": float(bh.loc[0, "coef"]), "psi_low_se": float(bh.loc[0, "se"]),
                "diff": float(bh.loc[1, "coef"]), "diff_se": float(bh.loc[1, "se"]),
                "psi_high": float(bh.loc[0, "coef"] + bh.loc[1, "coef"]),
            }
    return out


# ============================================================ 8. sensitivity
def jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else np.nan


def run_variant(spl, X, outcome="Yw", reg=None, clf=None, e_clip=(0.05, 0.95)):
    ests = fit_seeds(spl, X, outcome=outcome, reg=reg, clf=clf, e_clip=e_clip)
    agg = aggregate(ests, spl)
    tab, info = incidence(agg)
    cf = country_frame(agg, "dr")
    tr = search_tree(cf["Z"], B_of(cf, 1.0, cf["Gam"]), K_MAIN, depth=2, feature_names=Z_NAMES)
    treated = set(cf["iso3"][tr.predict(cf["Z"])])
    return {"avg": tab[tab.decile == "average"].iloc[0], "treated": treated, "info": info,
            "cf": cf, "tree": tr}


# ============================================================ MAIN
def main() -> None:
    fig_style()
    cy = build_treatment()
    sp = build_spells(cy)

    spl = decile_sample(sp, "T_stock", winsor=True)
    rc = spl.attrs["reg_cols"]
    X = X_SPELL + rc
    md = spl.drop_duplicates("uid")
    n_file = sp["spell_id"].nunique()
    log("sample", f"spells in spells_deciles_v2.csv: {n_file}; analysis sample (non-missing Y, S_spell, T_stock "
                  f"and X, incl. y0 >= 1999): {md.uid.nunique()} spells, {md.iso3.nunique()} countries, "
                  f"{len(spl)} spell x decile rows; EMDE countries {md[md.emde == 1].iso3.nunique()}; "
                  f"share of spells with T = 1: {md['T'].mean():.3f}")

    t = time.time()
    ests = fit_seeds(spl, X, outcome="Yw")
    agg = aggregate(ests, spl)
    RES["agg"] = agg
    log("fit", f"EW-DRPL main fit over fold seeds {FOLD_SEEDS} in {time.time() - t:.0f}s; ATE over deciles by seed: "
               + ", ".join(f"{v:.3f}" for v in agg["seed_ate"]))

    # ---- layer 2: incidence
    inc_tab, info = incidence(agg)
    RES["inc"], RES["inc_info"] = inc_tab, info
    avg = inc_tab[inc_tab.decile == "average"].iloc[0]
    write_table(inc_tab.round(6), OUT / "decile_incidence",
                caption="Resilience incidence curve (mean Gamma by decile, country-clustered SE) and damage profile psi* by T.",
                label="tab:incidence",
                notes=f"Spells {info['n_spells']}, countries {info['n_countries']}, T=1 spells {info['n_T1']}, "
                      f"T=0 spells {info['n_T0']}. Gamma averaged over fold seeds. 'average' = mean over deciles.")
    log("incidence", f"sample: {info['n_spells']} spells, {info['n_countries']} countries "
                     f"(T=1 spells {info['n_T1']}, T=0 spells {info['n_T0']})")
    for r in inc_tab.itertuples():
        if r.decile != "average":
            log("incidence", f"Gamma decile {r.decile}: {r.gamma:.3f} (SE {r.se:.3f}); psi* T=1 {r.psi_T1:.3f}, T=0 {r.psi_T0:.3f}")
    log("incidence", f"average over deciles: {avg.gamma:.3f} (SE {avg.se:.3f}); seed range "
                     f"{info['seed_ate_min']:.3f} to {info['seed_ate_max']:.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    dd = inc_tab[inc_tab.decile != "average"].copy()
    x = dd.decile.astype(int).to_numpy()
    ax[0].errorbar(x, dd.gamma, yerr=1.96 * dd.se, color=BLACK, marker="o", ms=4, capsize=2, lw=1)
    ax[0].axhline(avg.gamma, color=MID, ls="--", lw=1)
    ax[0].text(10.3, avg.gamma, "avg", color=DARK, va="center", fontsize=7)
    ax[0].axhline(0, color=LIGHT, lw=0.6)
    ax[0].set_xlabel("income decile (1 = poorest)")
    ax[0].set_ylabel(r"resilience effect $\bar\Gamma_d$")
    ax[0].set_title("(a) Resilience incidence curve")
    ax[0].set_xticks(x)
    ax[1].errorbar(x - 0.1, dd.psi_T1, yerr=1.96 * dd.se_psi_T1, color=BLACK, marker="s", ms=4,
                   capsize=2, lw=1, label="T = 1")
    ax[1].errorbar(x + 0.1, dd.psi_T0, yerr=1.96 * dd.se_psi_T0, color=MID, marker="^", ms=4,
                   capsize=2, lw=1, ls="--", label="T = 0")
    ax[1].axhline(0, color=LIGHT, lw=0.6)
    ax[1].set_xlabel("income decile (1 = poorest)")
    ax[1].set_ylabel(r"damage profile $\bar\psi^*_d$")
    ax[1].set_title("(b) Damage profile by capability status")
    ax[1].set_xticks(x)
    ax[1].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save_fig(fig, OUT / "fig_incidence")

    # ---- BLP / GATES / CLAN
    b, gt, gd, ct = heterogeneity(agg)
    write_table(pd.DataFrame({"term": b.term, "coef": b.coef.map(lambda v: fmt(v, 3)),
                              "se": b.se.map(lambda v: fmt(v, 3)), "p": b.p.map(lambda v: fmt(v, 3))}),
                OUT / "decile_blp", caption="BLP of the decile-averaged Gamma on standardised Z.",
                label="tab:blp", notes="Z: ln GDP pc, Gini, internet, agriculture share, baseline temperature, "
                                       "sovereignty (missing set to -0.1), standardised over spells. Country-clustered SE.")
    gt_out = pd.DataFrame({"quintile of predicted rho": gt["group"].astype(int).to_numpy(),
                           "mean Gamma": gt["mean"].map(lambda v: fmt(v, 3)).to_numpy(),
                           "se": gt["se"].map(lambda v: fmt(v, 3)).to_numpy(),
                           "n spells": gt["n"].map(lambda v: fmt(v, 0)).to_numpy()})
    gt_out.loc[len(gt_out)] = ["top minus bottom", fmt(gd["top_minus_bottom"], 3), fmt(gd["se"], 3), "--"]
    write_table(gt_out, OUT / "decile_gates", caption="GATES: mean decile-averaged Gamma by quintile of out-of-fold predicted rho.",
                label="tab:gates", notes="Quintile 1 = lowest predicted resilience effect. Predicted rho = plug-in m1 - m0 averaged over deciles.")
    ct_out = ct.copy()
    for c_ in ct_out.columns:
        if c_ != "feature":
            ct_out[c_] = ct_out[c_].map(lambda v: fmt(v, 2))
    write_table(ct_out, OUT / "decile_clan", caption="CLAN: mean of Z by quintile of predicted rho.",
                label="tab:clan", notes="q1 = lowest predicted resilience, q5 = highest. top_minus_bottom = q5 - q1; se country-clustered.")
    RES["blp"], RES["gates"], RES["gates_diff"], RES["clan"] = b, gt, gd, ct
    log("hetero", "BLP on standardised Z: " + "; ".join(f"{r.term} {r.coef:.3f} (SE {r.se:.3f})" for r in b.itertuples()))
    log("hetero", f"GATES top minus bottom quintile: {gd['top_minus_bottom']:.3f} (SE {gd['se']:.3f})")
    log("hetero", "GATES quintile means: " + ", ".join(f"q{int(r.group)} {r.mean:.3f}" for r in gt.itertuples()))
    log("hetero", "CLAN top minus bottom: " + "; ".join(f"{r.feature} {r.top_minus_bottom:.2f}" for r in ct.itertuples()))

    # ---- welfare and policy
    cf = country_frame(agg, "dr")
    assert np.array_equal(agg["countries"], cf["iso3"])
    N = len(cf["iso3"])
    folds = agg["fold_country"]
    log("policy", f"welfare/policy country sample N = {N} (EMDE {int(cf['emde'].sum())}); budget floor(0.3 N) = {int(np.floor(K_MAIN * N))}")
    pol = policy_block(cf, K_MAIN, folds)
    RES["pol"] = pol
    v = pol["values"]
    write_table(pd.DataFrame({"method": v.method, "eps": v.eps.map(lambda z: fmt(z, 1)),
                              "value [95% CI]": [f"{r.value:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]" for r in v.itertuples()]}),
                OUT / "policy_values", caption="Welfare value (1/N) sum_i pi_i B_i(eps) by method and eps (budget k = 0.3).",
                label="tab:policy_values", notes="Held-out values are cross-fitted over country folds. 95% CIs: country-cluster bootstrap (B = 500).",
                align="llr")
    mat_rows = []
    for e in EPS:
        row = {"rule eps": fmt(e, 1)}
        for e2 in EPS:
            m_, lo, hi = pol["matrix"][(e, e2)]
            row[f"eval eps={fmt(e2, 1)}"] = f"{m_:.4f} [{lo:.4f}, {hi:.4f}]"
        mat_rows.append(row)
    write_table(pd.DataFrame(mat_rows), OUT / "policy_matrix",
                caption="Cross-fitted held-out value of the eps rule (rows) evaluated under eps' (columns).",
                label="tab:policy_matrix", notes="Budget k = 0.3. Brackets: 95% country-cluster bootstrap CI.")
    eq = pol["equity"]
    write_table(pd.DataFrame({"eps": eq.eps.map(lambda z: fmt(z, 1)),
                              "equity gain W_eps(pi_eps) - W_eps(pi_0)": [f"{r.gain:.4f} [{r.gain_lo:.4f}, {r.gain_hi:.4f}]" for r in eq.itertuples()],
                              "equity price W_0(pi_0) - W_0(pi_eps)": [f"{r.price:.4f} [{r.price_lo:.4f}, {r.price_hi:.4f}]" for r in eq.itertuples()]}),
                OUT / "equity", caption="Equity gain and equity price of the eps-rule against the utilitarian rule (k = 0.3).",
                label="tab:equity", notes="Cross-fitted held-out values; 95% country-cluster bootstrap CIs in brackets.", align="lll")
    log("policy", "held-out policy matrix (rows: rule eps; columns: evaluation eps " + ", ".join(fmt(e, 1) for e in EPS) + "): "
        + " | ".join(f"eps {fmt(e, 1)}: " + ", ".join(f"{pol['matrix'][(e, e2)][0]:.4f}" for e2 in EPS) for e in EPS))
    for r in eq.itertuples():
        log("policy", f"eps {r.eps}: equity gain {r.gain:.4f} [{r.gain_lo:.4f}, {r.gain_hi:.4f}]; equity price {r.price:.4f} [{r.price_lo:.4f}, {r.price_hi:.4f}]")
    for e in EPS:
        treated = list(cf["iso3"][pol["pi_full"][e]])
        log("policy", f"eps {e} rule (full sample): {pol['tree'][e].describe()}")
        log("policy", f"eps {e} treated ({len(treated)}): " + ", ".join(sorted(treated)))
    rows = []
    for i in range(N):
        flags = {e: bool(pol["pi_full"][e][i]) for e in EPS}
        if any(flags.values()):
            rows.append({"ISO3": cf["iso3"][i], "country": cf["name"][i], "region": cf["region"][i],
                         "EMDE": "yes" if cf["emde"][i] else "no", "ASEAN": "yes" if cf["iso3"][i] in ASEAN else "no",
                         **{f"eps {fmt(e, 1)}": "x" if flags[e] else "" for e in EPS}})
    write_table(pd.DataFrame(rows).sort_values("ISO3"), OUT / "treated_countries",
                caption="Countries treated under the full-sample depth-2 tree for each eps (k = 0.3).",
                label="tab:treated", notes="x = treated. Only countries treated under at least one eps are listed. "
                                           "ASEAN: VNM, IDN, PHL, THA, MYS, KHM, LAO, MMR.", align="llllcccc")
    iso_list = list(cf["iso3"])
    if "VNM" in iso_list:
        vnm = {e: bool(pol["pi_full"][e][iso_list.index("VNM")]) for e in EPS}
        log("policy", "Viet Nam treated (full-sample tree): " + ", ".join(f"eps {fmt(e, 1)}: {'yes' if vnm[e] else 'no'}" for e in EPS))
        RES["vnm"] = vnm
    rules = []
    for e in EPS:
        own = v[(v.method.str.startswith("EW-DRPL")) & (v.eps == e)].iloc[0]
        rules.append({"eps": fmt(e, 1), "rule (full sample, depth 2)": pol["tree"][e].describe(),
                      "treated": int(pol["pi_full"][e].sum()), "held-out value": fmt(own.value, 4)})
    write_table(pd.DataFrame(rules), OUT / "policy_trees", caption="Learned depth-2 tree policies (k = 0.3).",
                label="tab:trees", notes="Rules use Z <= threshold (raw units); treated-leaf flags are (LL, LR, RL, RR) in the rule text.",
                align="lp{8cm}rr")

    # ---- budget sensitivity
    ks_rows = []
    for kk in K_SENS:
        pk = policy_block(cf, kk, folds)
        RES[f"pol_k{kk}"] = pk
        vk = pk["values"]
        for e in EPS:
            own = vk[(vk.method.str.startswith("EW-DRPL")) & (vk.eps == e)].iloc[0]
            ks_rows.append({"k": fmt(kk, 1), "eps": fmt(e, 1), "treated": int(pk["pi_full"][e].sum()),
                            "rule (full sample)": pk["tree"][e].describe(), "held-out value": fmt(own.value, 4)})
            log("budget", f"k={kk} eps={e}: treated {int(pk['pi_full'][e].sum())}; rule {pk['tree'][e].describe()}; "
                          f"held-out value {own.value:.4f}")
        for r in pk["equity"].itertuples():
            log("budget", f"k={kk} eps={r.eps}: equity gain {r.gain:.4f} [{r.gain_lo:.4f}, {r.gain_hi:.4f}]; "
                          f"equity price {r.price:.4f} [{r.price_lo:.4f}, {r.price_hi:.4f}]")
    write_table(pd.DataFrame(ks_rows), OUT / "policy_budget_sensitivity",
                caption="Sensitivity of tree rules and held-out values to the budget share k.",
                label="tab:budget", notes="Held-out value under the rule's own eps.", align="lllp{7cm}r")

    # ---- layer 1: country-year ATE and BLP
    L1 = {}
    for outc in ["gdppc_growth", "agri_growth"]:
        t = time.time()
        L1[outc] = layer1(cy, outc)
        L1[outc]["twfe"], L1[outc]["twfe_n"], L1[outc]["twfe_g"] = twfe(L1[outc]["frame"], outc)
        log("layer1", f"{outc}: N = {L1[outc]['n']} country-years, {L1[outc]['countries']} countries; ATE of Gamma "
                      f"{L1[outc]['ate']:.3f} (SE {L1[outc]['se']:.3f}); seed ATE range "
                      f"{min(L1[outc]['seed_ate']):.3f} to {max(L1[outc]['seed_ate']):.3f}; {time.time() - t:.0f}s")
        log("layer1", f"{outc} BLP: " + "; ".join(f"{r.term} {r.coef:.3f} (SE {r.se:.3f})" for r in L1[outc]["blp"].itertuples()))
    RES["L1"] = L1
    write_table(pd.DataFrame({"outcome": list(L1), "N country-years": [L1[o]["n"] for o in L1],
                              "countries": [L1[o]["countries"] for o in L1],
                              "ATE of Gamma": [fmt(L1[o]["ate"], 3) for o in L1],
                              "SE": [fmt(L1[o]["se"], 3) for o in L1]}),
                OUT / "layer1_ate", caption="Country-year layer: ATE of the resilience score (D = 1).",
                label="tab:layer1", notes="Treatment T_stock at t-1; shock = de-trended temperature anomaly. Country-clustered SE.",
                align="lrrrr")
    write_table(pd.DataFrame([{"outcome": o, "term": r.term, "coef": fmt(r.coef, 3), "se": fmt(r.se, 3), "p": fmt(r.p, 3)}
                              for o in L1 for r in L1[o]["blp"].itertuples()]),
                OUT / "layer1_blp", caption="Country-year layer: BLP of Gamma on standardised Z.",
                label="tab:layer1blp", notes="Country-clustered SE.", align="llrrr")

    # ---- classical baseline vs decile analogue
    dec_out, n_dec, g_dec = pooled_decile(spl, rc)
    RES["dec_pooled"] = dec_out
    rowmap = [("shock S", "shock", "S_spell"), ("shock x T", "shock_x_T", "S_x_T"), ("T (main effect)", "T_prev", "T")]
    rows = []
    for lab, kt, kd in rowmap:
        r = {"term": lab}
        for o in ["gdppc_growth", "agri_growth"]:
            vv = L1[o]["twfe"][kt]
            r[f"TWFE {o}"] = fmt_se(vv[0], vv[1], 3)
        vv = dec_out[kd]
        r["Decile pooled (income growth)"] = fmt_se(vv[0], vv[1], 3)
        rows.append(r)
    rows.append({"term": "N", "TWFE gdppc_growth": str(L1["gdppc_growth"]["twfe_n"]),
                 "TWFE agri_growth": str(L1["agri_growth"]["twfe_n"]), "Decile pooled (income growth)": str(n_dec)})
    rows.append({"term": "countries", "TWFE gdppc_growth": str(L1["gdppc_growth"]["twfe_g"]),
                 "TWFE agri_growth": str(L1["agri_growth"]["twfe_g"]), "Decile pooled (income growth)": str(g_dec)})
    write_table(pd.DataFrame(rows), OUT / "baseline_comparison",
                caption="Classical baseline: TWFE interaction models (country-year) and the pooled decile analogue.",
                label="tab:baseline",
                notes="TWFE: country and year fixed effects; controls ln GDP pc, Gini, internet, agriculture share, urban share, "
                      "ln population, readiness, ln AI pc (lag). Decile pooled: decile fixed effects and the decile-layer X "
                      "(region dummies included). T main effect included in both. SE clustered by country. Outcome scales differ "
                      "(annual growth in % vs. decile income growth in % per year).",
                align="lcccc")
    for lab, kt in [("TWFE gdppc", "gdppc_growth"), ("TWFE agri", "agri_growth")]:
        vv = L1[kt]["twfe"]
        log("baseline", f"{lab}: shock {vv['shock'][0]:.3f} (SE {vv['shock'][1]:.3f}); shock x T {vv['shock_x_T'][0]:.3f} "
                        f"(SE {vv['shock_x_T'][1]:.3f}); N {L1[kt]['twfe_n']}")
    log("baseline", f"decile pooled: S {dec_out['S_spell'][0]:.3f} (SE {dec_out['S_spell'][1]:.3f}); "
                    f"S x T {dec_out['S_x_T'][0]:.3f} (SE {dec_out['S_x_T'][1]:.3f}); N {n_dec}, countries {g_dec}")

    # ---- cost module
    cost = cost_module(cy)
    RES["cost"] = cost
    cc = [("ln_elec_pc", "ln_kpc_lag"), ("ln_co2pc", "ln_kpc_lag"), ("ln_elec_pc", "ln_ai_pc_lag"), ("ln_co2pc", "ln_ai_pc_lag")]
    heads = ["elec on K-stock", "CO2 on K-stock", "elec on AI pc", "CO2 on AI pc"]
    cost_rows = [
        {"": "treatment coefficient", **{h: fmt_se(cost[k]["theta"], cost[k]["se"], 3) for h, k in zip(heads, cc)}},
        {"": "p-value", **{h: fmt(cost[k]["p"], 3) for h, k in zip(heads, cc)}},
        {"": "BLP slope on std. renewable share", **{h: fmt_se(cost[k]["blp_coef"], cost[k]["blp_se"], 3) for h, k in zip(heads, cc)}},
        {"": "orthogonal score, low renewable share", **{h: fmt_se(cost[k]["psi_low"], cost[k]["psi_low_se"], 3) for h, k in zip(heads, cc)}},
        {"": "difference, high minus low renewable share", **{h: fmt_se(cost[k]["diff"], cost[k]["diff_se"], 3) for h, k in zip(heads, cc)}},
        {"": "country-years", **{h: str(cost[k]["N"]) for h, k in zip(heads, cc)}},
        {"": "countries", **{h: str(cost[k]["G"]) for h, k in zip(heads, cc)}},
    ]
    write_table(pd.DataFrame(cost_rows), OUT / "cost",
                caption="Cost module: DML partially linear estimates of electricity and CO2 per capita on climate-AI capability.",
                label="tab:cost",
                notes="Columns: outcome on treatment (K-stock = ln(1 + k_pc lagged); AI pc = ln(1 + ai_pc lagged)). Cross-fitted "
                      "gradient boosting, 5 country folds (seed 42), Mundlak country means of all controls. SE clustered by country. "
                      "Renewable share split at its median.", align="lcccc")
    for k, v_ in cost.items():
        log("cost", f"{k[0]} on {k[1]}: theta {v_['theta']:.4f} (SE {v_['se']:.4f}, p {v_['p']:.3f}); N {v_['N']}, countries {v_['G']}; "
                    f"BLP slope on std. renewable share {v_['blp_coef']:.4f} (SE {v_['blp_se']:.4f}); "
                    f"high-minus-low renewable difference {v_['diff']:.4f} (SE {v_['diff_se']:.4f})")

    # ---- sensitivity of the decile layer
    t = time.time()
    main_treated = set(cf["iso3"][pol["pi_full"][1.0]])
    sens_rows = [{"variant": "main", "spells": info["n_spells"], "countries": info["n_countries"],
                  "avg Gamma (SE)": fmt_se(avg.gamma, avg.se, 3), "eps=1 treated": len(main_treated),
                  "overlap with main": "1.000", "Jaccard": "1.000"}]
    cf_raw = country_frame(agg, "raw")
    tr_raw = search_tree(cf_raw["Z"], B_of(cf_raw, 1.0, cf_raw["Gam"]), K_MAIN, depth=2, feature_names=Z_NAMES)
    tset = set(cf_raw["iso3"][tr_raw.predict(cf_raw["Z"])])
    m_raw, s_raw = cluster_mean_se(agg["G"]["raw"].mean(axis=1), agg["groups"])
    sens_rows.append({"variant": "(a) raw psi score (non-orthogonal)", "spells": info["n_spells"], "countries": info["n_countries"],
                      "avg Gamma (SE)": fmt_se(m_raw, s_raw, 3), "eps=1 treated": len(tset),
                      "overlap with main": f"{len(tset & main_treated) / len(main_treated):.3f}", "Jaccard": f"{jacc(tset, main_treated):.3f}"})
    log("sensitivity", f"(a) raw psi score: average Gamma {m_raw:.3f} (SE {s_raw:.3f}); eps=1 treated {len(tset)}; "
                       f"overlap {len(tset & main_treated)}/{len(main_treated)}")
    specs = [
        ("(b) propensity clipped to [0.1, 0.9]", spl, X, "Yw", None, None, (0.1, 0.9)),
        ("(c) T at 60th percentile", decile_sample(sp, "T_stock60", winsor=True), X, "Yw", None, None, (0.05, 0.95)),
        ("(d) random forest nuisances", spl, X, "Yw",
         RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=1),
         RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=1), (0.05, 0.95)),
        ("(e) EMDE-only sample", spl[spl.emde == 1].copy(), X, "Yw", None, None, (0.05, 0.95)),
        ("(f) unwinsorised Y", spl, X, "Y", None, None, (0.05, 0.95)),
    ]
    for lab, sdf, Xv, outc, reg_, clf_, eclip in specs:
        vv = run_variant(sdf, Xv, outcome=outc, reg=reg_, clf=clf_, e_clip=eclip)
        ts = vv["treated"]
        sens_rows.append({"variant": lab, "spells": vv["info"]["n_spells"], "countries": vv["info"]["n_countries"],
                          "avg Gamma (SE)": fmt_se(vv["avg"].gamma, vv["avg"].se, 3), "eps=1 treated": len(ts),
                          "overlap with main": f"{len(ts & main_treated) / len(main_treated):.3f}",
                          "Jaccard": f"{jacc(ts, main_treated):.3f}"})
        log("sensitivity", f"{lab}: spells {vv['info']['n_spells']}, countries {vv['info']['n_countries']}; average Gamma "
                           f"{vv['avg'].gamma:.3f} (SE {vv['avg'].se:.3f}); eps=1 treated {len(ts)}; overlap "
                           f"{len(ts & main_treated)}/{len(main_treated)}, Jaccard {jacc(ts, main_treated):.3f}")
    sens = pd.DataFrame(sens_rows)
    write_table(sens, OUT / "sensitivity",
                caption="Sensitivity of the decile layer: average Gamma over deciles and overlap of the eps = 1 treated set with the main rule.",
                label="tab:sensitivity",
                notes="Overlap = share of the main eps=1 treated countries also treated under the variant (full-sample depth-2 tree, k = 0.3). "
                      "Jaccard = |A and B| / |A or B|. Variant (a) reuses the main fit. Variant (b) clips rather than drops propensities.",
                align="lrrrrrr")
    RES["sens"] = sens
    log("sensitivity", f"block time {time.time() - t:.0f}s")

    # ---- diagnostics
    meta = agg["meta"]
    Xmat = meta[X].astype(float).to_numpy()
    fold_spell = pd.Series(agg["fold_country"], index=agg["countries"]).loc[agg["groups"]].to_numpy()
    Tv = meta["T"].to_numpy()
    e_raw = crossfit_propensity(Xmat, Tv, fold_spell, default_classifier(), clip=(1e-6, 1 - 1e-6))
    e_cl = agg["e_hat"]
    trim05 = float(np.mean((e_raw < 0.05) | (e_raw > 0.95)))
    trim01 = float(np.mean((e_raw < 0.10) | (e_raw > 0.90)))
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    bins = np.linspace(0, 1, 26)
    ax.hist(e_raw[Tv == 0], bins=bins, color=LIGHT, edgecolor=DARK, lw=0.5, label="T = 0")
    ax.hist(e_raw[Tv == 1], bins=bins, histtype="step", color=BLACK, lw=1.4, ls="--", label="T = 1")
    ax.axvline(0.05, color=MID, ls=":", lw=1)
    ax.axvline(0.95, color=MID, ls=":", lw=1)
    ax.set_xlabel(r"cross-fitted propensity $\hat e(X)$")
    ax.set_ylabel("spells")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, OUT / "fig_overlap")
    w1 = Tv / e_cl
    w0 = (1 - Tv) / (1 - e_cl)
    bal = []
    for x_ in X:
        v_ = meta[x_].astype(float).to_numpy()
        sd = v_.std(ddof=1)
        smd0 = (v_[Tv == 1].mean() - v_[Tv == 0].mean()) / sd
        smd1 = (np.sum(w1 * v_) / np.sum(w1) - np.sum(w0 * v_) / np.sum(w0)) / sd
        bal.append({"covariate": x_, "SMD before weighting": fmt(smd0, 3), "SMD after IPW": fmt(smd1, 3)})
    write_table(pd.DataFrame(bal), OUT / "balance",
                caption="Covariate balance before and after inverse-propensity weighting (ATE weights, clipped propensity).",
                label="tab:balance", notes="Standardised mean differences (T=1 minus T=0), denominators are the unweighted SD of each covariate.",
                align="lrr")
    dsc = []
    spells_u = spl.drop_duplicates("uid")
    for lab, src, col in [("Y (income growth, winsorised, spell x decile)", spl, "Yw"),
                          ("S (spell mean shock, deg C)", spells_u, "S_spell")] + \
                         [(x_, spells_u, x_) for x_ in X_SPELL]:
        vv = src[col].astype(float)
        tt = src["T"].astype(float)
        dsc.append({"variable": lab, "mean T=0": fmt(vv[tt == 0].mean(), 3), "sd T=0": fmt(vv[tt == 0].std(), 3),
                    "mean T=1": fmt(vv[tt == 1].mean(), 3), "sd T=1": fmt(vv[tt == 1].std(), 3)})
    write_table(pd.DataFrame(dsc), OUT / "descriptives",
                caption="Descriptive statistics by climate-AI capability status (decile-layer analysis sample).",
                label="tab:desc", notes=f"Spell-level variables use {len(spells_u)} spells; Y is pooled over spell x decile rows.",
                align="lrrrr")
    log("diagnostics", f"raw propensity outside [0.05, 0.95]: {trim05:.3f} of spells; outside [0.10, 0.90]: {trim01:.3f}; "
                       f"mean e: T=1 {e_raw[Tv == 1].mean():.3f}, T=0 {e_raw[Tv == 0].mean():.3f}")
    log("diagnostics", "max |SMD| before weighting: " + f"{max(abs(float(r['SMD before weighting'])) for r in bal):.3f}"
        + "; after IPW: " + f"{max(abs(float(r['SMD after IPW'])) for r in bal):.3f}")

    write_summary()
    log("runtime", f"total runtime {time.time() - T0:.0f}s")
    print(f"Outputs written to {OUT}")


def write_summary() -> None:
    lines = ["# Application results (generated by code/30_application.py)", "",
             f"Seed {SEED}; fold seeds {FOLD_SEEDS}; delta = {DELTA}; main budget k = {K_MAIN}; "
             f"bootstrap draws B = {N_BOOT}; random-policy draws {N_RANDOM}. All numbers below come from the run that wrote this file.", ""]
    for sec, msgs in SUM.items():
        lines.append(f"## {sec}")
        lines.append("")
        lines += [f"- {m}" for m in msgs]
        lines.append("")
    (OUT / "application_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
