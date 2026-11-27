"""
Step 2: build the country-year panel (2000-2023) keyed by iso3 and year.

Inputs : data/raw/ (written by 01_download.py)
Outputs: data/processed/panel.csv, data/processed/codebook.md
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

YEARS = list(range(2000, 2024))
WIDE_YEARS = list(range(1995, 2025))   # WDI series are interpolated on the full span
SHIFT_SHARE_YEARS = list(range(2000, 2005))
SEED = 42  # no random draws in this step; kept for consistency across scripts


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_wb_countries():
    """World Bank country metadata without aggregates."""
    meta = json.loads((RAW / "wb_countries.json").read_text())[1]
    rows = []
    for c in meta:
        if c["region"]["value"] == "Aggregates":
            continue
        iso3 = c["id"]
        if not iso3 or len(iso3) != 3:
            continue
        rows.append({"iso3": iso3, "iso2": c["iso2Code"], "country": c["name"],
                     "region": c["region"]["value"], "income_level": c["incomeLevel"]["value"]})
    return pd.DataFrame(rows).drop_duplicates("iso3")


def load_wdi(code, name):
    """Long WDI series for all countries, years 1995-2024."""
    data = json.loads((RAW / "wdi" / f"{code}.json").read_text())[1] or []
    rows = [{"iso3": d["countryiso3code"], "year": int(d["date"]), name: d["value"]}
            for d in data if d.get("countryiso3code") and d["value"] is not None]
    return pd.DataFrame(rows, columns=["iso3", "year", name])


def load_ndgain_csv(pattern):
    """Read a wide ND-GAIN file (ISO3, Name, years...) into long form for 2000-2023."""
    matches = glob.glob(str(RAW / "ndgain" / "**" / pattern), recursive=True)
    if not matches:
        sys.exit(f"ND-GAIN file not found: {pattern}. Run code/01_download.py first.")
    wide = pd.read_csv(matches[0])
    cols = ["ISO3"] + [str(y) for y in YEARS]
    long = wide[cols].melt(id_vars="ISO3", var_name="year", value_name="value")
    long["year"] = long["year"].astype(int)
    return long.rename(columns={"ISO3": "iso3"})


def loader_owid(fname, columns):
    """OWID file restricted to country rows (3-letter ISO code, not an OWID aggregate)."""
    df = pd.read_csv(RAW / "owid" / fname, usecols=["iso_code", "year"] + columns)
    df = df[df["iso_code"].notna() & (df["iso_code"].str.len() == 3)
            & ~df["iso_code"].str.startswith("OWID")]
    df = df[df["year"].between(2000, 2023)]
    return df.rename(columns={"iso_code": "iso3"})


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build():
    PROC.mkdir(parents=True, exist_ok=True)
    meta = load_wb_countries()
    print(f"[meta] {len(meta)} WB countries (non-aggregate, iso3 present)")

    # Balanced grid: every WB country, 2000-2023
    grid = pd.MultiIndex.from_product([meta["iso3"], YEARS], names=["iso3", "year"]).to_frame(index=False)
    panel = grid.merge(meta, on="iso3", how="left")

    # ---- OpenAlex publication counts --------------------------------------
    oa = pd.read_csv(RAW / "openalex_counts.csv", keep_default_na=True)
    iso2_to_iso3 = dict(zip(meta["iso2"], meta["iso3"]))
    oa["iso3"] = oa["iso2"].map(iso2_to_iso3)
    unmapped = sorted(oa.loc[oa["iso3"].isna(), "iso2"].dropna().unique())
    print(f"[openalex] unmapped ISO2 codes dropped: {unmapped}")
    oa = oa.dropna(subset=["iso3"]).drop(columns="iso2")
    oa["year"] = oa["year"].astype(int)
    panel = panel.merge(oa, on=["iso3", "year"], how="left")

    # Rule: a missing AI/ML row is a true zero only if the country has any works that year.
    has_all = panel["all_pubs"].notna()
    for col in ["ai_pubs", "ml_pubs"]:
        panel[col] = np.where(panel[col].isna() & has_all, 0.0, panel[col])
        panel.loc[~has_all, col] = np.nan
    panel.loc[~has_all, "all_pubs"] = np.nan

    # ---- WDI ----------------------------------------------------------------
    wdi_series = {
        "SI.POV.GINI": "gini", "IT.NET.USER.ZS": "internet", "NY.GDP.PCAP.PP.KD": "gdppc",
        "SP.URB.TOTL.IN.ZS": "urban", "NE.TRD.GNFS.ZS": "trade", "SP.POP.TOTL": "pop",
        "SE.TER.ENRR": "tertiary",
    }
    wdi = None
    for code, name in wdi_series.items():
        s = load_wdi(code, name)
        wdi = s if wdi is None else wdi.merge(s, on=["iso3", "year"], how="outer")
    # Gini: linear interpolation within country over the full 1995-2024 span, inside
    # observed points only (no extrapolation). Flag interpolated values.
    gini_wide = (meta[["iso3"]].merge(
        wdi[["iso3", "year", "gini"]].dropna(), on="iso3", how="left"))
    gini_full = (pd.MultiIndex.from_product([meta["iso3"], WIDE_YEARS], names=["iso3", "year"])
                 .to_frame(index=False)
                 .merge(gini_wide, on=["iso3", "year"], how="left"))
    gini_full["gini_raw_obs"] = gini_full["gini"].notna()
    gini_full = gini_full.sort_values(["iso3", "year"])
    gini_full["gini_filled"] = gini_full.groupby("iso3")["gini"].transform(
        lambda s: s.interpolate(method="linear", limit_area="inside"))
    gini_full["gini_interp"] = (~gini_full["gini_raw_obs"] & gini_full["gini_filled"].notna()).astype(int)
    gini_full = gini_full[gini_full["year"].isin(YEARS)][["iso3", "year", "gini_filled", "gini_interp"]]
    gini_full = gini_full.rename(columns={"gini_filled": "gini"})

    wdi_panel = wdi.drop(columns=["gini"]).merge(gini_full, on=["iso3", "year"], how="right")
    panel = panel.merge(wdi_panel, on=["iso3", "year"], how="left")
    panel["gini_interp"] = panel["gini_interp"].fillna(0).astype(int)
    panel.loc[panel["gini"].isna(), "gini_interp"] = 0

    # ---- ND-GAIN -----------------------------------------------------------
    ng = {
        "vuln": "vulnerability/vulnerability.csv", "ready": "readiness/readiness.csv",
        "gain": "gain/gain.csv", "vuln_food": "food.csv", "vuln_water": "water.csv",
        "vuln_health": "health.csv", "vuln_eco": "ecosystems.csv",
        "vuln_habitat": "habitat.csv", "vuln_infra": "infrastructure.csv",
    }
    for name, pattern in ng.items():
        long = load_ndgain_csv(pattern.split("/")[-1] if "/" not in pattern else pattern)
        long = long.rename(columns={"value": name})
        if name != "gain":          # ND-GAIN vulnerability/readiness are 0-1; gain is already 0-100
            long[name] = long[name] * 100
        panel = panel.merge(long, on=["iso3", "year"], how="left")

    # ---- Energy and CO2 (OWID) --------------------------------------------
    energy = loader_owid("owid-energy-data.csv", ["per_capita_electricity", "renewables_share_elec"])
    co2 = loader_owid("owid-co2-data.csv", ["co2_per_capita"])
    panel = panel.merge(energy, on=["iso3", "year"], how="left")
    panel = panel.merge(co2, on=["iso3", "year"], how="left")
    panel = panel.rename(columns={"renewables_share_elec": "ren_share"})

    # ---- Derived variables ---------------------------------------------------
    pop_m = panel["pop"] / 1e6
    panel["ai_pc"] = panel["ai_pubs"] / pop_m
    panel["ln_ai"] = np.log1p(panel["ai_pc"])
    panel["ai_share"] = np.where(panel["all_pubs"] >= 50, 100 * panel["ai_pubs"] / panel["all_pubs"], np.nan)
    panel["ln_ml"] = np.log1p(panel["ml_pubs"] / pop_m)
    panel["ln_gdppc"] = np.log(panel["gdppc"].where(panel["gdppc"] > 0))
    panel["ln_pop"] = np.log(panel["pop"].where(panel["pop"] > 0))
    panel["ln_elec"] = np.log(panel["per_capita_electricity"].where(panel["per_capita_electricity"] > 0))
    panel["ln_co2pc"] = np.log(panel["co2_per_capita"].where(panel["co2_per_capita"] > 0))
    panel["emde"] = np.where(panel["income_level"].isna(), np.nan,
                             (panel["income_level"] != "High income").astype(float))

    # ---- Shift-share instrument ------------------------------------------------
    # s_i: country's share of AI publications in 2000-2004 (pre-period).
    pre = panel[panel["year"].isin(SHIFT_SHARE_YEARS)].groupby("iso3")[["ai_pubs", "all_pubs"]].sum(min_count=1)
    s_i = (pre["ai_pubs"] / pre["all_pubs"]).where(pre["all_pubs"] > 0)
    panel["s_i"] = panel["iso3"].map(s_i)
    # Leave-one-out world AI count: W_{-i,t} = sum over all countries in the grid minus own AI.
    world = panel.groupby("year")["ai_pubs"].sum(min_count=1).rename("W_t")
    panel = panel.merge(world, left_on="year", right_index=True, how="left")
    panel["W_loo"] = panel["W_t"] - panel["ai_pubs"]
    base = panel.loc[panel["year"] == 2000, ["iso3", "W_loo"]].rename(columns={"W_loo": "W_loo_2000"})
    panel = panel.merge(base, on="iso3", how="left")
    panel["iv_ss"] = panel["s_i"] * (np.log(panel["W_loo"].where(panel["W_loo"] > 0))
                                     - np.log(panel["W_loo_2000"].where(panel["W_loo_2000"] > 0)))
    panel["iv_ss_lvl"] = np.log1p(panel["s_i"] * panel["W_loo"] / pop_m)
    panel = panel.drop(columns=["W_t", "W_loo", "W_loo_2000"])

    # ---- Sample restriction: population >= 1 million in 2010 ----------------------
    pop2010 = panel.loc[panel["year"] == 2010].set_index("iso3")["pop"]
    keep = pop2010[pop2010 >= 1e6].index
    n_before = panel["iso3"].nunique()
    panel = panel[panel["iso3"].isin(keep)].copy()
    print(f"[sample] countries before pop filter: {n_before}; after (pop2010>=1m): {panel['iso3'].nunique()}")

    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    keep_cols = ["iso3", "year", "country", "region", "income_level", "emde",
                 "ai_pubs", "all_pubs", "ml_pubs", "pop", "ai_pc", "ln_ai", "ai_share", "ln_ml",
                 "vuln", "ready", "gain", "vuln_food", "vuln_water", "vuln_health", "vuln_eco",
                 "vuln_habitat", "vuln_infra",
                 "gini", "gini_interp", "internet", "ln_gdppc", "urban", "trade", "ln_pop", "tertiary",
                 "ln_elec", "ren_share", "ln_co2pc", "s_i", "iv_ss", "iv_ss_lvl"]
    panel = panel[keep_cols]
    panel.to_csv(PROC / "panel.csv", index=False)
    print(f"[panel] wrote {PROC / 'panel.csv'}: {len(panel)} rows, "
          f"{panel['iso3'].nunique()} countries, years {panel['year'].min()}-{panel['year'].max()}")
    write_codebook(panel)
    return panel


CODEBOOK = [
    ("iso3", "ISO 3166-1 alpha-3 country code (panel key)", "World Bank country metadata", "code"),
    ("year", "Calendar year (panel 2000-2023)", "-", "year"),
    ("country", "Country name", "World Bank", "text"),
    ("region", "World Bank region", "World Bank", "text"),
    ("income_level", "World Bank income group (current classification)", "World Bank", "text"),
    ("emde", "1 if income group is not High income (emerging and developing economy)", "World Bank", "dummy"),
    ("ai_pubs", "OpenAlex works tagged with concept C154945302 (Artificial intelligence), authored by country; zero if country has any works that year", "OpenAlex", "count"),
    ("all_pubs", "OpenAlex works with at least one author from the country", "OpenAlex", "count"),
    ("ml_pubs", "OpenAlex works tagged with concept C119857082 (Machine learning), by country", "OpenAlex", "count"),
    ("pop", "Total population", "World Bank SP.POP.TOTL", "persons"),
    ("ai_pc", "AI publications per million population = ai_pubs / (pop/1e6)", "derived", "papers per million"),
    ("ln_ai", "AI capability: log(1 + ai_pc)", "derived", "log"),
    ("ai_share", "AI share of national publications = 100*ai_pubs/all_pubs (NaN if all_pubs<50)", "derived", "percent"),
    ("ln_ml", "log(1 + ML publications per million population)", "derived", "log"),
    ("vuln", "ND-GAIN vulnerability index x100 (higher = more vulnerable)", "ND-GAIN country index", "index points 0-100"),
    ("ready", "ND-GAIN readiness index x100 (higher = more ready)", "ND-GAIN country index", "index points 0-100"),
    ("gain", "ND-GAIN overall GAIN score (higher = better)", "ND-GAIN country index", "index points 0-100"),
    ("vuln_food", "ND-GAIN food vulnerability sector x100", "ND-GAIN sector files", "index points"),
    ("vuln_water", "ND-GAIN water vulnerability sector x100", "ND-GAIN sector files", "index points"),
    ("vuln_health", "ND-GAIN health vulnerability sector x100", "ND-GAIN sector files", "index points"),
    ("vuln_eco", "ND-GAIN ecosystems vulnerability sector x100", "ND-GAIN sector files", "index points"),
    ("vuln_habitat", "ND-GAIN habitat vulnerability sector x100", "ND-GAIN sector files", "index points"),
    ("vuln_infra", "ND-GAIN infrastructure vulnerability sector x100", "ND-GAIN sector files", "index points"),
    ("gini", "Gini index of income inequality, linearly interpolated within country (no extrapolation)", "World Bank SI.POV.GINI", "0-100"),
    ("gini_interp", "1 if gini was interpolated (no observation in that year)", "derived", "dummy"),
    ("internet", "Individuals using the Internet", "World Bank IT.NET.USER.ZS", "% of population"),
    ("ln_gdppc", "log GDP per capita, PPP (constant 2021 int$)", "World Bank NY.GDP.PCAP.PP.KD", "log"),
    ("urban", "Urban population share", "World Bank SP.URB.TOTL.IN.ZS", "% of population"),
    ("trade", "Trade openness (exports + imports of goods and services)", "World Bank NE.TRD.GNFS.ZS", "% of GDP"),
    ("ln_pop", "log total population", "World Bank SP.POP.TOTL", "log persons"),
    ("tertiary", "Gross enrolment ratio, tertiary", "World Bank SE.TER.ENRR", "%"),
    ("ln_elec", "log electricity demand per capita", "OWID energy-data per_capita_electricity", "log kWh per person"),
    ("ren_share", "Renewables share of electricity generation", "OWID energy-data renewables_share_elec", "%"),
    ("ln_co2pc", "log CO2 emissions per capita", "OWID co2-data co2_per_capita", "log tonnes per person"),
    ("s_i", "Shift-share exposure: country AI share of publications, 2000-2004 pooled", "derived from OpenAlex", "share 0-1"),
    ("iv_ss", "Shift-share instrument: s_i x (log W_{-i,t} - log W_{-i,2000}), W = world AI pubs excluding own", "derived", "product"),
    ("iv_ss_lvl", "Alternative instrument: log(1 + s_i * W_{-i,t} / (pop_{i,t}/1e6))", "derived", "log"),
]


def write_codebook(panel):
    lines = ["# Codebook: data/processed/panel.csv", "",
             f"Country-year panel, years 2000-2023, keyed by `iso3` and `year`. "
             f"{panel['iso3'].nunique()} countries (population >= 1 million in 2010), "
             f"{len(panel)} rows.", "",
             "Lagging and centring are applied in code/03_analysis.py, not in this file.", "",
             "| Variable | Definition | Source | Unit |", "|---|---|---|---|"]
    for var, desc, src, unit in CODEBOOK:
        lines.append(f"| `{var}` | {desc} | {src} | {unit} |")
    lines += ["", "## Build notes",
              "- OpenAlex `group_by` does not accept `page`; where a year's country list is truncated at 200 groups, "
              "every World Bank country missing from the list was queried directly (`code/01_download.py`).",
              "- A country-year absent from the OpenAlex AI or ML output is coded 0 only if the country has "
              "all-works output in that year; otherwise missing.",
              "- ND-GAIN vulnerability, readiness and sector indices are multiplied by 100; GAIN is kept as published.",
              "- The shift-share world total uses every country in the OpenAlex output for the grid (before the population filter).",
              ""]
    (PROC / "codebook.md").write_text("\n".join(lines))
    print(f"[codebook] wrote {PROC / 'codebook.md'}")


if __name__ == "__main__":
    build()
