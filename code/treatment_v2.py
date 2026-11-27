"""Shared treatment construction for the v2 application (climate-AI capability).

add_treatments(cy) adds to the country-year frame:
  K, kpc            climate-AI knowledge stock K_it = sum_{tau<=t} climai_works (1-delta)^(t-tau), per million pop
  kpc_lag, ln_kpc_lag   k_pc at t-1 and log(1 + k_pc_{t-1})
  T_stock           1 if k_pc_{t-1} > cross-country median (old, no income adjustment)
  ln_gdppc_lag      ln GDP pc at t-1
  r_inc             residual of ln(1+k_pc_{t-1}) on a cubic in ln GDP pc_{t-1} with year FE (pooled)
  T_res             1 if r_inc > 0 (main treatment), T_resSD = 1 if r_inc > 0.25 SD(r)
  T_prev, T_stock_prev, r_prev   values at t-1 (the treatment used for growth in year t)
  sov_z             sovereignty with missing set to -0.1
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DELTA = 0.15


def add_treatments(cy: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cy = cy.sort_values(["iso3", "year"]).reset_index(drop=True)
    span = cy.groupby("iso3").year.agg(["min", "max", "size"])
    assert ((span["max"] - span["min"] + 1) == span["size"]).all(), "country-year panel must be complete"
    cy["K"] = np.nan
    for _, g in cy.groupby("iso3", sort=False):
        K = 0.0
        for i in g.index[g.year >= 1998]:
            K = K * (1 - DELTA) + cy.at[i, "climai_works"]
            cy.at[i, "K"] = K
    cy["kpc"] = cy["K"] / (cy["pop"] / 1e6)
    wide = cy.pivot(index="year", columns="iso3", values="kpc")
    lag = wide.shift(1)
    thr50 = lag.median(axis=1)
    T50 = lag.gt(thr50, axis=0).astype(float).where(lag.notna())

    def stk(w, name):
        return w.stack().rename(name).reset_index()

    cy = cy.merge(stk(lag, "kpc_lag"), on=["year", "iso3"], how="left")
    cy = cy.merge(stk(T50, "T_stock"), on=["year", "iso3"], how="left")
    cy["ln_kpc_lag"] = np.log1p(cy["kpc_lag"])
    cy["ln_gdppc_lag"] = cy.groupby("iso3")["ln_gdppc"].shift(1)
    cy["sov_z"] = cy["sovereignty"].fillna(-0.1)

    fit = cy.dropna(subset=["ln_kpc_lag", "ln_gdppc_lag"])
    x = fit["ln_gdppc_lag"].to_numpy(float)
    xc = (x - x.mean()) / x.std()
    yrs = pd.get_dummies(fit["year"].astype(str), dtype=float).to_numpy()
    M = np.column_stack([xc, xc ** 2, xc ** 3, yrs])
    yv = fit["ln_kpc_lag"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(M, yv, rcond=None)
    resid = yv - M @ beta
    r2 = 1 - np.sum(resid ** 2) / np.sum((yv - yv.mean()) ** 2)
    rcol = pd.Series(np.nan, index=cy.index)
    rcol.loc[fit.index] = resid
    cy["r_inc"] = rcol
    sd_r = float(np.std(resid, ddof=1))
    cy["T_res"] = (cy["r_inc"] > 0).astype(float).where(cy["r_inc"].notna())
    cy["T_resSD"] = (cy["r_inc"] > 0.25 * sd_r).astype(float).where(cy["r_inc"].notna())
    cy["T_prev"] = cy.groupby("iso3")["T_res"].shift(1)
    cy["T_stock_prev"] = cy.groupby("iso3")["T_stock"].shift(1)
    cy["r_prev"] = cy.groupby("iso3")["r_inc"].shift(1)
    info = {"r2": r2, "sd_r": sd_r, "n_fit": len(fit)}
    return cy, info
