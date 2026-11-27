"""Build the agricultural-resilience panel: data/processed/agri_panel.csv.

Inputs (raw): data/processed/country_year_v2.csv and spells_deciles_v2.csv (already built),
data/raw/owid_v2/average-monthly-surface-temperature.csv (monthly mean surface temperature),
WDI cereal yield (AG.YLD.CREL.KG), agriculture employment (SL.AGR.EMPL.ZS) and fertilizer use
(AG.CON.FERT.ZS), downloaded here to data/raw/wdi/ if missing (1997-2023).

Unit: country-year, 2000-2023 (treatment at t-1 needs 1999). Outcome dly = 100 * (ln Y_t - ln Y_{t-1}).
Run: python code/12_build_agri.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from treatment_v2 import add_treatments  # noqa: E402

RAW = ROOT / "data" / "raw"
WDI_DIR = RAW / "wdi"
PROC = ROOT / "data" / "processed"
WDI_NEW = {"AG.YLD.CREL.KG": "cereal_yield", "SL.AGR.EMPL.ZS": "agri_empl", "AG.CON.FERT.ZS": "fert"}
USER_AGENT = "Mozilla/5.0 (research data build)"


def fetch(url: str) -> bytes:
    last = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except Exception as err:  # noqa: BLE001
            last = err
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed {url}: {last}")


def ensure_wdi() -> None:
    WDI_DIR.mkdir(parents=True, exist_ok=True)
    for code in WDI_NEW:
        path = WDI_DIR / f"{code}.json"
        if not path.exists():
            url = (f"https://api.worldbank.org/v2/country/all/indicator/{code}"
                   "?format=json&per_page=20000&date=1997:2023")
            path.write_bytes(fetch(url))
            time.sleep(0.5)
            print(f"[wdi] saved {code}")


def read_wdi(code: str) -> pd.DataFrame:
    payload = json.loads((WDI_DIR / f"{code}.json").read_text())[1]
    rows = [{"iso3": r["countryiso3code"], "year": int(r["date"]), "value": r["value"]}
            for r in payload if r.get("countryiso3code") and r.get("value") is not None]
    return pd.DataFrame(rows)


def monthly_tmax() -> pd.DataFrame:
    t = pd.read_csv(RAW / "owid_v2" / "average-monthly-surface-temperature.csv").dropna(subset=["Code"])
    t["year"] = t["Month"].str.slice(0, 4).astype(int)
    g = t.groupby(["Code", "year"])["Monthly average"].agg(["max", "size"]).reset_index()
    g = g[g["size"] == 12].rename(columns={"Code": "iso3", "max": "tmax"})   # full years only
    return g[["iso3", "year", "tmax"]]


def hot_shock(tm: pd.DataFrame) -> pd.DataFrame:
    out = []
    for iso, g in tm.groupby("iso3"):
        h = g[(g.year >= 1980) & (g.year <= 2023)]
        if len(h) < 15:
            continue
        b = np.polyfit(h.year.to_numpy(float), h.tmax.to_numpy(float), 1)
        d = g.copy()
        d["hotm"] = d["tmax"] - np.polyval(b, d["year"].to_numpy(float))
        sd = d[(d.year >= 2000) & (d.year <= 2023)]["hotm"]
        d["hotm_sd_i"] = sd.std(ddof=1) if len(sd) >= 20 else np.nan
        out.append(d)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ensure_wdi()
    cy_raw = pd.read_csv(PROC / "country_year_v2.csv")
    cy, tinfo = add_treatments(cy_raw)
    print(f"[treatment] residual R2 {tinfo['r2']:.3f}, sd {tinfo['sd_r']:.3f}")

    # WDI indicators
    wdi = None
    for code, name in WDI_NEW.items():
        d = read_wdi(code).rename(columns={"value": name})
        wdi = d if wdi is None else wdi.merge(d, on=["iso3", "year"], how="outer")
    cy = cy.merge(wdi, on=["iso3", "year"], how="left")
    cy["ln_fert"] = np.log(cy["fert"].where(cy["fert"] > 0))

    # outcome: 100 * log change of cereal yield, consecutive years only
    cy = cy.sort_values(["iso3", "year"])
    prev_year = cy.groupby("iso3")["year"].shift(1)
    prev_y = cy.groupby("iso3")["cereal_yield"].shift(1)
    raw_dly = 100 * (np.log(cy["cereal_yield"]) - np.log(prev_y))
    consecutive = (cy["year"] - prev_year) == 1
    dly = raw_dly.where(consecutive)
    big = dly.abs() >= 100
    n_big = int(big.sum())
    cy["dly"] = dly.where(~big)
    cy["agri_growth_v"] = cy["agri_growth"]

    # temperature shocks
    tm = hot_shock(monthly_tmax())
    cy = cy.merge(tm[["iso3", "year", "tmax", "hotm", "hotm_sd_i"]].rename(columns={"hotm_sd_i": "hotm_sd_i_tm"}),
                  on=["iso3", "year"], how="left")
    cy["hotm_sd_i"] = cy.groupby("iso3")["hotm_sd_i_tm"].transform("first")
    cy = cy.drop(columns=["hotm_sd_i_tm"])

    # income deciles: latest PIP survey per country (decile means, PPP $/day)
    sp = pd.read_csv(PROC / "spells_deciles_v2.csv", keep_default_na=False, na_values=[""])
    sp = sp.sort_values(["iso3", "y1", "d"])
    last_y1 = sp.groupby("iso3")["y1"].max()
    lat = sp[sp["y1"] == sp["iso3"].map(last_y1)].pivot(index="iso3", columns="d", values="y_d1")
    lat.columns = [f"y_dec{int(c)}" for c in lat.columns]
    lat["pip_latest_year"] = last_y1
    cy = cy.merge(lat.reset_index(), on="iso3", how="left")
    pop23 = cy[cy.year == 2023].set_index("iso3")["pop"]
    cy["pop_2023"] = cy["iso3"].map(pop23)

    keep_years = cy[cy.year >= 2000].copy()
    cols = ["iso3", "country", "region", "emde", "year", "dly", "cereal_yield", "agri_growth_v", "gdppc_growth",
            "hotm", "tmax", "hotm_sd_i", "shock", "shock_sd_i", "precip_shock", "base_temp",
            "T_prev", "T_stock_prev", "r_prev", "ln_kpc_lag", "ln_gdppc", "gini", "internet", "agri_share",
            "urban", "ln_pop", "pop", "pop_2023", "readiness", "ln_ai_pc_lag", "sovereignty", "sov_z",
            "agri_empl", "ln_fert", "fert", "ln_elec_pc", "ln_co2pc", "ren_share", "r_inc", "T_res", "T_stock"] + \
           [c for c in cy.columns if c.startswith("y_dec")] + ["pip_latest_year"]
    keep_years = keep_years.rename(columns={"T_prev": "T_res_l1", "T_stock_prev": "T_stock_l1",
                                            "r_prev": "r_inc_l1", "T_res": "T_res_t"})
    out = keep_years.rename(columns={"T_res_l1": "T_res", "T_stock_l1": "T_stock", "r_inc_l1": "r_inc"})
    out["T_res_t"] = keep_years["T_res_t"]
    out_cols = ["iso3", "country", "region", "emde", "year", "dly", "cereal_yield", "agri_growth_v", "gdppc_growth",
                "hotm", "tmax", "hotm_sd_i", "shock", "shock_sd_i", "precip_shock", "base_temp",
                "T_res", "T_stock", "r_inc", "ln_kpc_lag", "ln_gdppc", "gini", "internet", "agri_share", "urban",
                "ln_pop", "pop", "pop_2023", "readiness", "ln_ai_pc_lag", "sovereignty", "sov_z", "agri_empl",
                "ln_fert", "fert", "ln_elec_pc", "ln_co2pc", "ren_share", "T_res_t"] + \
               [c for c in cy.columns if c.startswith("y_dec")] + ["pip_latest_year"]
    out = out[out_cols].sort_values(["iso3", "year"])
    out.to_csv(PROC / "agri_panel.csv", index=False)

    d = out.dropna(subset=["dly"])
    print(f"[agri] rows {len(out)}; countries {out.iso3.nunique()}; dly non-missing {len(d)}; "
          f"dropped |dly| >= 100: {n_big}")
    miss = out[["dly", "hotm", "T_res", "ln_gdppc", "gini", "internet", "agri_share", "urban", "ln_pop",
                "readiness", "ln_ai_pc_lag", "base_temp", "precip_shock", "ln_fert", "agri_empl"]].isna().mean()
    print("[agri] missing share:", {k: round(float(v), 3) for k, v in miss.items()})
    print(f"[agri] hotm SD {out.hotm.std():.3f}; countries with hotm_sd_i: {out.dropna(subset=['hotm_sd_i']).iso3.nunique()}")
    print("[agri] income deciles available for", out.dropna(subset=["y_dec1"]).iso3.nunique(), "countries")


if __name__ == "__main__":
    main()
