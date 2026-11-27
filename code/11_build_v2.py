"""
Step 2 (design v2): build the analysis files from data/raw.

Outputs
-------
* data/processed/country_year_v2.csv   country-year panel, 1998-2023, countries with pop >= 1 million in 2010
* data/processed/spells_deciles_v2.csv survey spells x deciles (long format, one row per spell and decile)

Inputs are written by code/01_download.py and code/10_download_v2.py. Run
`python code/10_download_v2.py && python code/11_build_v2.py`.

Definitions follow docs/research_design_v2.md (sections 2, 5, 7). Choices the design leaves open
are listed in data/processed/codebook_v2.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OWID_V2 = RAW / "owid_v2"
PROC = ROOT / "data" / "processed"

YEARS = list(range(1998, 2024))
PERIODS = [("1998-2007", 1998, 2007), ("2008-2015", 2008, 2015), ("2016-2023", 2016, 2023)]
TREND_WIN = (1980, 2023)      # trend fit window for the temperature shock
BASE_WIN = (1980, 2000)       # baseline temperature window
SD_WIN = (2000, 2023)         # window for shock_sd_i
POP_MIN = 1_000_000
POP_YEAR = 2010
MIN_CLIMAI_PERIOD = 5         # sovereignty requires at least this many climate-AI works in the period
MAX_X_LOOKBACK = 3            # spells: covariates at y0, else nearest earlier non-missing within 3 years
GAP_RANGE = (1, 10)
DECILES = [f"decile{d}" for d in range(1, 11)]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_wb():
    meta = json.loads((RAW / "wb_countries.json").read_text())[1]
    rows = []
    for c in meta:
        region = c["region"]["value"].strip()
        iso3 = c["id"] or ""
        if region == "Aggregates" or len(iso3) != 3:
            continue
        rows.append({"iso3": iso3, "iso2": c["iso2Code"], "country": c["name"],
                     "region": region, "income_group": c["incomeLevel"]["value"].strip()})
    return pd.DataFrame(rows)


def read_wdi(code):
    payload = json.loads((RAW / "wdi" / f"{code}.json").read_text())
    rows = [(r["countryiso3code"], int(r["date"]), r["value"])
            for r in (payload[1] or []) if r.get("countryiso3code")]
    df = pd.DataFrame(rows, columns=["iso3", "year", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.drop_duplicates(["iso3", "year"])


def load_temperature():
    """Annual mean surface temperature (deg C) from OWID monthly data (12 months required).
    Fallback: OWID annual temperature anomalies (not an absolute level)."""
    monthly = OWID_V2 / "average-monthly-surface-temperature.csv"
    if monthly.exists():
        t = pd.read_csv(monthly).dropna(subset=["Code"])
        t["year"] = t["Month"].str[:4].astype(int)
        g = t.groupby(["Code", "year"])["Monthly average"].agg(["mean", "count"]).reset_index()
        g = g[g["count"] == 12]
        print("[temp] source: monthly OWID grapher, annual mean of 12 months")
        return g.rename(columns={"Code": "iso3", "mean": "value"})[["iso3", "year", "value"]]
    anom = OWID_V2 / "annual-temperature-anomalies.csv"
    t = pd.read_csv(anom).dropna(subset=["Code"])
    print("[temp] WARNING: fallback to annual anomalies; base_temp is an anomaly, not a level")
    return t.rename(columns={"Code": "iso3", "Year": "year", "Temperature anomaly": "value"})[
        ["iso3", "year", "value"]]


def load_precip():
    path = OWID_V2 / "average-precipitation-per-year.csv"
    if not path.exists():
        print("[precip] not available; precip columns left empty")
        return None
    p = pd.read_csv(path).dropna(subset=["Code"])
    return p.rename(columns={"Code": "iso3", "Year": "year", "Annual precipitation": "value"})[
        ["iso3", "year", "value"]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detrended_shock(long_df):
    """Deviation from a country-specific linear trend fitted on TREND_WIN.
    Returns long frame (iso3, year, shock) and the trend slope per country."""
    out, slopes = [], {}
    for iso3, g in long_df.groupby("iso3"):
        s = g.dropna(subset=["value"]).set_index("year")["value"].sort_index()
        win = s[(s.index >= TREND_WIN[0]) & (s.index <= TREND_WIN[1])]
        if len(win) < 30:
            continue
        b, a = np.polyfit(win.index.values.astype(float), win.values, 1)
        out.append(pd.DataFrame({"iso3": iso3, "year": win.index.values,
                                 "shock": win.values - (a + b * win.index.values.astype(float))}))
        slopes[iso3] = b
    return pd.concat(out, ignore_index=True), slopes


def attach(panel, long_df, col):
    m = long_df[["iso3", "year", "value"]].drop_duplicates(["iso3", "year"])
    return panel.merge(m.rename(columns={"value": col}), on=["iso3", "year"], how="left")


def window_stat(long_df, lo, hi, how, min_obs):
    w = long_df[(long_df.year >= lo) & (long_df.year <= hi)].dropna(subset=["value"])
    agg = w.groupby("iso3")["value"].agg(["mean", "std", "count"])
    agg = agg[agg["count"] >= min_obs]
    return agg["mean"] if how == "mean" else agg["std"]


def asof_lookup(pairs, panel, cols):
    """Value of each col at (iso3, y0); if missing, the nearest earlier non-missing value
    within MAX_X_LOOKBACK years."""
    out = pairs[["iso3", "y0"]].reset_index(drop=True).copy()
    for c in cols:
        out[c] = np.nan
    base = panel[["iso3", "year"] + cols]
    for k in range(MAX_X_LOOKBACK + 1):
        tmp = base.copy()
        tmp["y0"] = tmp["year"] + k
        tmp = tmp.drop(columns="year").drop_duplicates(["iso3", "y0"])
        m = out[["iso3", "y0"]].merge(tmp, on=["iso3", "y0"], how="left")
        for c in cols:
            out[c] = out[c].fillna(m[c])
    return out[cols]


# ---------------------------------------------------------------------------
# Country-year panel
# ---------------------------------------------------------------------------
def build_country_year(wb):
    pop = read_wdi("SP.POP.TOTL")
    pop_ok = pop[(pop.year == POP_YEAR) & (pop.value >= POP_MIN)].iso3
    uni = wb[wb.iso3.isin(pop_ok)].sort_values("iso3").reset_index(drop=True)
    print(f"[universe] {len(uni)} countries with population >= 1 million in {POP_YEAR}")

    panel = pd.MultiIndex.from_product([uni.iso3.tolist(), YEARS], names=["iso3", "year"]).to_frame(index=False)
    panel = panel.merge(uni, on="iso3", how="left")

    # Temperature and shock (trend fitted on the full 1980-2023 temperature series)
    temp_long = load_temperature()
    shock_long, slopes = detrended_shock(temp_long[temp_long.iso3.isin(uni.iso3)])
    temp_u = temp_long[temp_long.iso3.isin(uni.iso3)]
    panel = attach(panel, temp_u, "temp")
    shock_long = shock_long.rename(columns={"shock": "value"})
    panel = attach(panel, shock_long, "shock")
    panel["shock_hot"] = np.maximum(panel["shock"], 0)
    sd = window_stat(shock_long.rename(columns={"value": "value"}), *SD_WIN, how="sd", min_obs=20)
    base = window_stat(temp_u, *BASE_WIN, how="mean", min_obs=15)
    panel["shock_sd_i"] = panel.iso3.map(sd)
    panel["base_temp"] = panel.iso3.map(base)

    precip = load_precip()
    if precip is not None:
        precip = precip[precip.iso3.isin(uni.iso3)]
        panel = attach(panel, precip, "precip_mm")
        pshock, _ = detrended_shock(precip)
        panel = attach(panel, pshock.rename(columns={"shock": "value"}), "precip_shock")

    # WDI
    wdi_map = {
        "gdppc_growth": "NY.GDP.PCAP.KD.ZG",
        "agri_growth": "NV.AGR.TOTL.KD.ZG",
        "internet": "IT.NET.USER.ZS",
        "agri_share": "NV.AGR.TOTL.ZS",
        "urban": "SP.URB.TOTL.IN.ZS",
    }
    for col, code in wdi_map.items():
        panel = attach(panel, read_wdi(code), col)
    gdppc_pp = read_wdi("NY.GDP.PCAP.PP.KD")
    gdppc_pp["value"] = np.log(gdppc_pp["value"].where(gdppc_pp["value"] > 0))
    panel = attach(panel, gdppc_pp, "ln_gdppc")
    panel = attach(panel, pop, "pop")
    panel["ln_pop"] = np.log(panel["pop"].where(panel["pop"] > 0))

    # Gini: linear interpolation inside the observed range only (no extrapolation)
    gini_long = read_wdi("SI.POV.GINI")
    wide = gini_long.pivot(index="iso3", columns="year", values="value")
    wide = wide.interpolate(axis=1, limit_area="inside")
    gini_i = wide.stack().rename("value").reset_index()
    panel = attach(panel, gini_i, "gini")

    # ND-GAIN readiness (x100); the design uses the lagged value as the governance proxy
    rd = pd.read_csv(RAW / "ndgain" / "readiness" / "readiness.csv")
    ycols = [c for c in rd.columns if c.isdigit()]
    rlong = rd.melt(id_vars=["ISO3"], value_vars=ycols, var_name="year", value_name="value")
    rlong = rlong.rename(columns={"ISO3": "iso3"})
    rlong["year"] = rlong["year"].astype(int)
    rlong["value"] = rlong["value"] * 100
    panel = attach(panel, rlong, "readiness_t0")
    lagged = rlong.copy()
    lagged["year"] = lagged["year"] + 1           # value of t-1 keyed at year t
    panel = attach(panel, lagged, "readiness")

    # OpenAlex counts (iso2 -> iso3)
    iso2_to_3 = dict(zip(wb.iso2, wb.iso3))
    cnt = pd.read_csv(RAW / "openalex_v2_counts.csv", keep_default_na=False, na_values=[""])  # keep ISO2 "NA" (Namibia)
    cnt["iso3"] = cnt["iso2"].map(iso2_to_3)
    cnt = cnt.dropna(subset=["iso3"])
    for src, dst in [("all_pubs", "all_works"), ("ai_pubs", "ai_works"), ("climai_pubs", "climai_works")]:
        panel = attach(panel, cnt.rename(columns={src: "value"}), dst)
    panel["ai_pc"] = panel["ai_works"] / panel["pop"] * 1e6
    panel["climai_pc"] = panel["climai_works"] / panel["pop"] * 1e6
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)

    # Climate-AI capability: 3-year rolling sum per million, lagged one year, median split
    panel["climai_pc_3y"] = panel.groupby("iso3")["climai_pc"].transform(
        lambda s: s.rolling(3, min_periods=3).sum())
    panel["lag_climai_3y"] = panel.groupby("iso3")["climai_pc_3y"].shift(1)
    med = panel.groupby("year")["lag_climai_3y"].transform("median")
    panel["T_climai"] = np.where(panel["lag_climai_3y"].notna(),
                                 (panel["lag_climai_3y"] > med).astype(float), np.nan)
    panel["ln_ai_pc_lag"] = np.log1p(panel.groupby("iso3")["ai_pc"].shift(1))

    # Knowledge sovereignty: domestic-only share of climate-AI works, by period
    dom = pd.read_csv(RAW / "openalex_v2_domestic.csv", keep_default_na=False, na_values=[""])
    dom["iso3"] = dom["iso2"].map(iso2_to_3)
    dom = dom.dropna(subset=["iso3"])
    period_of_year = {y: p for p, a, b in PERIODS for y in range(a, b + 1)}
    panel["period"] = panel["year"].map(period_of_year)
    panel = panel.merge(dom[["iso3", "period", "climai_domestic", "climai_total"]],
                        on=["iso3", "period"], how="left")
    panel = panel.rename(columns={"climai_total": "climai_period_n"})
    ok = panel["climai_period_n"] >= MIN_CLIMAI_PERIOD
    panel["sovereignty"] = np.where(ok & panel["climai_domestic"].notna(),
                                    panel["climai_domestic"] / panel["climai_period_n"], np.nan)

    # Energy and CO2 (OWID)
    en = pd.read_csv(RAW / "owid" / "owid-energy-data.csv",
                     usecols=["iso_code", "year", "renewables_share_elec", "per_capita_electricity"])
    en = en.rename(columns={"iso_code": "iso3"}).dropna(subset=["iso3"])
    en["ren_share"] = en["renewables_share_elec"]
    en["ln_elec_pc"] = np.log(en["per_capita_electricity"].where(en["per_capita_electricity"] > 0))
    panel = attach(panel, en[["iso3", "year", "ren_share"]].rename(columns={"ren_share": "value"}), "ren_share")
    panel = attach(panel, en[["iso3", "year", "ln_elec_pc"]].rename(columns={"ln_elec_pc": "value"}),
                   "ln_elec_pc")
    co2 = pd.read_csv(RAW / "owid" / "owid-co2-data.csv", usecols=["iso_code", "year", "co2_per_capita"])
    co2 = co2.rename(columns={"iso_code": "iso3"}).dropna(subset=["iso3"])
    co2["value"] = np.log(co2["co2_per_capita"].where(co2["co2_per_capita"] > 0))
    panel = attach(panel, co2[["iso3", "year", "value"]], "ln_co2pc")

    panel["emde"] = (panel["income_group"] != "High income").astype(int)
    panel["slope_temp_trend"] = panel.iso3.map(slopes)

    panel = panel[panel.year.isin(YEARS)].copy()
    cols = ["iso3", "iso2", "country", "region", "income_group", "emde", "year",
            "temp", "shock", "shock_hot", "shock_sd_i", "base_temp", "slope_temp_trend",
            "precip_mm", "precip_shock",
            "gdppc_growth", "agri_growth", "ln_gdppc", "gini", "internet", "agri_share", "urban",
            "pop", "ln_pop", "readiness", "readiness_t0",
            "all_works", "ai_works", "climai_works", "ai_pc", "climai_pc", "climai_pc_3y",
            "T_climai", "ln_ai_pc_lag", "period", "climai_domestic", "climai_period_n", "sovereignty",
            "ren_share", "ln_elec_pc", "ln_co2pc"]
    for c in cols:
        if c not in panel.columns:
            panel[c] = np.nan
    panel = panel[cols].sort_values(["iso3", "year"]).reset_index(drop=True)
    out = PROC / "country_year_v2.csv"
    panel.to_csv(out, index=False, float_format="%.6g")
    print(f"[build] wrote {out} ({len(panel)} rows, {panel.iso3.nunique()} countries)")
    return panel, shock_long, uni


# ---------------------------------------------------------------------------
# Survey spells and decile outcomes
# ---------------------------------------------------------------------------
def build_spells(panel, shock_long, uni):
    pip = pd.read_csv(RAW / "pip_surveys.csv")
    pip = pip.rename(columns={"country_code": "iso3", "reporting_year": "year"})
    n_raw = len(pip)
    pip = pip[pip.iso3.isin(uni.iso3)]
    print(f"[pip] {len(pip)} survey rows in universe (of {n_raw} total)")

    # Reporting level: national if available, else the only level present; else skip country
    keep, skipped_level = [], []
    for iso3, g in pip.groupby("iso3"):
        levels = set(g.reporting_level)
        if "national" in levels:
            keep.append(g[g.reporting_level == "national"])
        elif len(levels) == 1:
            keep.append(g)
        else:
            skipped_level.append(iso3)
    pip = pd.concat(keep, ignore_index=True)
    print(f"[pip] countries skipped (no national series and several levels): {skipped_level}")

    # One welfare type per country: the one with most surveys (ties: alphabetical)
    cnt = pip.groupby(["iso3", "welfare_type"]).size().reset_index(name="n")
    cnt = cnt.sort_values(["iso3", "n", "welfare_type"], ascending=[True, False, True])
    best = cnt.drop_duplicates("iso3")[["iso3", "welfare_type"]]
    pip = pip.merge(best, on=["iso3", "welfare_type"])

    dup = pip.duplicated(["iso3", "year"]).sum()
    pip = pip.drop_duplicates(["iso3", "year"], keep="first")
    print(f"[pip] duplicate country-year survey rows dropped: {dup}")

    s = pip.sort_values(["iso3", "year"]).reset_index(drop=True)
    s["country"] = s["iso3"].map(dict(zip(uni.iso3, uni.country)))
    cols = ["year", "survey_comparability", "mean"] + DECILES
    nxt = s.groupby("iso3")[cols].shift(-1)
    pr = pd.concat([s[["iso3", "country", "reporting_level", "welfare_type"]],
                    s[cols].add_suffix("_0"), nxt.add_suffix("_1")], axis=1)
    pr = pr.dropna(subset=["year_1"])
    pr["y0"] = pr["year_0"].astype(int)
    pr["y1"] = pr["year_1"].astype(int)
    pr["gap"] = pr["y1"] - pr["y0"]
    n_pairs = len(pr)
    pr = pr[(pr.gap >= GAP_RANGE[0]) & (pr.gap <= GAP_RANGE[1])]
    pr = pr[pr.survey_comparability_0 == pr.survey_comparability_1]
    n_comp = len(pr)
    dec_cols_0 = [f"{d}_0" for d in DECILES]
    dec_cols_1 = [f"{d}_1" for d in DECILES]
    valid = (pr[["mean_0", "mean_1"]] > 0).all(axis=1) & (pr[dec_cols_0 + dec_cols_1] > 0).all(axis=1)
    pr = pr[valid]
    sums = pr[dec_cols_0].sum(axis=1)
    print(f"[spells] consecutive pairs: {n_pairs}; gap 1-10 and same comparability: {n_comp}; "
          f"valid positive decile means: {len(pr)}; decile shares sum deviating >0.01 from 1: "
          f"{int(((sums - 1).abs() > 0.01).sum())}")
    pr = pr.reset_index(drop=True)

    # Annual shock path per country (1980-2023) for the spell averages
    shock_by = {iso3: g.set_index("year")["value"] for iso3, g in shock_long.groupby("iso3")}
    S = []
    for iso3, y0, y1 in zip(pr.iso3, pr.y0, pr.y1):
        sh = shock_by.get(iso3)
        yrs = list(range(y0 + 1, y1 + 1))
        if sh is None or not all(y in sh.index and pd.notna(sh.loc[y]) for y in yrs):
            S.append(np.nan)
        else:
            S.append(float(sh.loc[yrs].mean()))
    pr["S_spell"] = S

    # Covariates and T at the start year
    pk = pr[["iso3", "y0"]].rename(columns={"y0": "year"})
    panel_t = panel[["iso3", "year", "T_climai"]]
    tm = pk.merge(panel_t, on=["iso3", "year"], how="left")
    pr["T_climai_y0"] = tm["T_climai"].values
    x_cols = ["ln_gdppc", "gini", "internet", "agri_share", "urban", "ln_pop", "readiness",
              "ln_ai_pc_lag", "sovereignty", "ren_share"]
    xs = asof_lookup(pr.rename(columns={}), panel, x_cols)
    for c in x_cols:
        pr[c] = xs[c].values
    tv = panel.groupby("iso3")[["base_temp", "shock_sd_i", "emde", "region", "income_group"]].first()
    for c in ["base_temp", "shock_sd_i", "emde", "region", "income_group"]:
        pr[c] = pr["iso3"].map(tv[c])

    pr["spell_id"] = pr["iso3"] + "_" + pr["y0"].astype(str) + "_" + pr["y1"].astype(str)
    pr["weight"] = pr["gap"]
    pr["survey_comparability"] = pr["survey_comparability_0"]

    recs = []
    for d in range(1, 11):
        t = pr.copy()
        t["d"] = d
        t["y_d0"] = t["mean_0"] * t[f"decile{d}_0"] * 10
        t["y_d1"] = t["mean_1"] * t[f"decile{d}_1"] * 10
        t["Y"] = (np.log(t["y_d1"]) - np.log(t["y_d0"])) / t["gap"] * 100
        recs.append(t)
    long = pd.concat(recs, ignore_index=True)
    long["complete_main"] = long[["Y", "S_spell", "T_climai_y0", "ln_gdppc", "gini", "internet",
                                  "agri_share", "urban", "ln_pop", "readiness", "ln_ai_pc_lag",
                                  "base_temp"]].notna().all(axis=1).astype(int)
    out_cols = ["spell_id", "iso3", "country", "region", "income_group", "emde", "d",
                "y0", "y1", "gap", "weight", "reporting_level", "welfare_type", "survey_comparability",
                "mean_0", "mean_1", "y_d0", "y_d1", "Y", "S_spell", "T_climai_y0",
                "ln_gdppc", "gini", "internet", "agri_share", "urban", "ln_pop", "readiness",
                "ln_ai_pc_lag", "sovereignty", "ren_share", "base_temp", "shock_sd_i", "complete_main"]
    long = long[out_cols]
    long = long.sort_values(["iso3", "y0", "y1", "d"]).reset_index(drop=True)
    out = PROC / "spells_deciles_v2.csv"
    long.to_csv(out, index=False, float_format="%.6g")
    print(f"[build] wrote {out} ({len(long)} rows, {long.spell_id.nunique()} spells, "
          f"{long.iso3.nunique()} countries)")
    return long


def summary(panel, spells):
    print("\n== Country-year panel ==")
    print(f"rows {len(panel)}, countries {panel.iso3.nunique()}, years {panel.year.min()}-{panel.year.max()}")
    for c in ["temp", "shock", "shock_hot", "shock_sd_i", "base_temp", "gdppc_growth", "agri_growth",
              "ln_gdppc", "gini", "internet", "agri_share", "urban", "readiness", "ai_pc", "climai_pc",
              "climai_pc_3y", "T_climai", "ln_ai_pc_lag", "sovereignty", "ren_share", "ln_elec_pc",
              "ln_co2pc", "precip_mm", "precip_shock"]:
        s = panel[c]
        print(f"  {c:14s} n={s.notna().sum():5d} miss={s.isna().mean():6.3f} "
              f"mean={s.mean():10.4f} sd={s.std():9.4f} min={s.min():10.4f} max={s.max():10.4f}")
    t = panel.dropna(subset=["T_climai"])
    print(f"  T_climai=1 share (country-years with T): {t.T_climai.mean():.4f} (n={len(t)})")
    print("\n== Spells ==")
    sp = spells[spells.d == 1]
    print(f"spells {len(sp)}, countries {sp.iso3.nunique()}, gap distribution: "
          f"{sp.gap.value_counts().sort_index().to_dict()}")
    print(f"spells with T_climai_y0 available: {sp.T_climai_y0.notna().sum()}; "
          f"share T=1 among them: {sp.T_climai_y0.mean():.4f}")
    print(f"spells with S_spell available: {sp.S_spell.notna().sum()}")
    print(f"complete_main spells: {sp.complete_main.sum()} (countries "
          f"{sp.loc[sp.complete_main == 1, 'iso3'].nunique()})")
    for c in ["S_spell", "T_climai_y0", "ln_gdppc", "gini", "internet", "agri_share", "urban", "ln_pop",
              "readiness", "ln_ai_pc_lag", "sovereignty", "ren_share", "base_temp"]:
        print(f"  {c:14s} miss={sp[c].isna().mean():6.3f}")
    y = spells.Y.dropna()
    print(f"  Y (spell x decile): n={len(y)} mean={y.mean():.4f} sd={y.std():.4f} "
          f"p5={y.quantile(.05):.4f} p50={y.median():.4f} p95={y.quantile(.95):.4f}")
    print("  Y by decile (mean):", spells.groupby("d").Y.mean().round(3).to_dict())
    print("  S_spell mean/sd:", round(sp.S_spell.mean(), 4), round(sp.S_spell.std(), 4))


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    wb = load_wb()
    panel, shock_long, uni = build_country_year(wb)
    spells = build_spells(panel, shock_long, uni)
    summary(panel, spells)
    print("\nBuild step (v2) complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
