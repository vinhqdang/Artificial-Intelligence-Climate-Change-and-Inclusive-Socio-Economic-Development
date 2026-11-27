"""
Step 1: download and cache all raw inputs.

Idempotent: every file is skipped when it already exists in data/raw/.

Sources
-------
* ND-GAIN country index (manual download, unzipped locally)
* OpenAlex works counts by country (AI concept, all works, machine learning concept)
* World Bank WDI indicators and country metadata
* Our World in Data energy and CO2 datasets
* IMF AI Preparedness Index (2023 cross-section)

Usage: python code/01_download.py
"""
import csv
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
NDGAIN_ZIP = RAW / "ndgain_countryindex_2026.zip"
NDGAIN_DIR = RAW / "ndgain"
OPENALEX_DIR = RAW / "openalex"
WDI_DIR = RAW / "wdi"
OWID_DIR = RAW / "owid"

USER_AGENT = "ai-climate-panel-pipeline/1.0"
SLEEP = 0.2          # seconds between API calls
MAX_RETRIES = 4      # retries after the first attempt

AI_CONCEPT = "C154945302"   # Artificial intelligence
ML_CONCEPT = "C119857082"   # Machine learning
YEARS = range(2000, 2024)   # 2000..2023 inclusive
WDI_CODES = ["SI.POV.GINI", "IT.NET.USER.ZS", "NY.GDP.PCAP.PP.KD", "SP.URB.TOTL.IN.ZS",
             "NE.TRD.GNFS.ZS", "SP.POP.TOTL", "SE.TER.ENRR"]
OWID_FILES = {
    "owid-energy-data.csv": "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv",
    "owid-co2-data.csv": "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv",
}
IMF_URL = "https://www.imf.org/external/datamapper/api/v1/AI_PI"


def fetch_bytes(url):
    """GET with up to MAX_RETRIES retries and exponential backoff. Sleeps SLEEP after success."""
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            time.sleep(SLEEP)
            return data
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as err:
            last_err = err
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {url} ({last_err})")


def fetch_json(url):
    return json.loads(fetch_bytes(url).decode("utf-8"))


def save_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


# ---------------------------------------------------------------------------
# 1. ND-GAIN (manual download; unzip locally)
# ---------------------------------------------------------------------------
def step_ndgain():
    if not NDGAIN_ZIP.exists() and not glob.glob(str(NDGAIN_DIR / "**" / "gain.csv"), recursive=True):
        print("ND-GAIN archive not found. Download it manually from:\n"
              "  https://gain.nd.edu/our-work/country-index/download-data/\n"
              f"and save it as {NDGAIN_ZIP}")
        sys.exit(1)
    if glob.glob(str(NDGAIN_DIR / "**" / "vulnerability.csv"), recursive=True):
        print("[ndgain] already unzipped, skipping")
        return
    NDGAIN_DIR.mkdir(parents=True, exist_ok=True)
    base = NDGAIN_DIR.resolve()
    with zipfile.ZipFile(NDGAIN_ZIP) as zf:
        for member in zf.namelist():
            if member.startswith("__MACOSX") or member.endswith("/"):
                continue
            target = (NDGAIN_DIR / member).resolve()
            if base not in target.parents:          # guard against path traversal
                raise RuntimeError(f"Unsafe path in archive: {member}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(member))
    print(f"[ndgain] unzipped to {NDGAIN_DIR}")


# ---------------------------------------------------------------------------
# 2. World Bank metadata and WDI indicators
# ---------------------------------------------------------------------------
def step_worldbank():
    meta_path = RAW / "wb_countries.json"
    if not meta_path.exists():
        data = fetch_bytes("https://api.worldbank.org/v2/country?format=json&per_page=400")
        save_bytes(meta_path, data)
        print("[wb] country metadata saved")
    for code in WDI_CODES:
        path = WDI_DIR / f"{code}.json"
        if path.exists():
            continue
        url = (f"https://api.worldbank.org/v2/country/all/indicator/{code}"
               "?format=json&per_page=20000&date=1995:2024")
        data = fetch_bytes(url)
        save_bytes(path, data)
        print(f"[wdi] saved {code}")


# ---------------------------------------------------------------------------
# 3. OpenAlex country counts
# ---------------------------------------------------------------------------
def openalex_group(filter_expr):
    """Top-200 country groups for a works filter. The endpoint rejects page=, so
    truncation (groups_count > 200) is handled by the caller with direct queries."""
    url = ("https://api.openalex.org/works?"
           + urllib.parse.urlencode({"filter": filter_expr, "group_by": "authorships.countries",
                                     "per_page": 200}))
    d = fetch_json(url)
    return d["meta"]["groups_count"], {g["key"].rsplit("/", 1)[-1]: g["count"] for g in d.get("group_by", [])}


def openalex_count(filter_expr):
    """Work count for one filter. Returns None when OpenAlex rejects the filter (HTTP 400),
    which happens for codes that are not ISO countries (e.g. WB code JG, Channel Islands)."""
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode({"filter": filter_expr, "per_page": 1})
    try:
        return fetch_json(url)["meta"]["count"]
    except RuntimeError as err:
        if "HTTP Error 400" in str(err):
            print(f"[openalex] rejected filter, skipped: {filter_expr}")
            return None
        raise


def step_openalex(wb_iso2_codes):
    """Save per-year country counts for AI, all works and ML.

    For a year whose group_by result is truncated (200 groups returned), countries
    missing from the list are queried one by one, so that no country is silently
    set to zero. Results cached as JSON per concept-year.
    """
    OPENALEX_DIR.mkdir(parents=True, exist_ok=True)
    specs = {
        "ai": lambda y: f"concepts.id:{AI_CONCEPT},publication_year:{y}",
        "all": lambda y: f"publication_year:{y}",
        "ml": lambda y: f"concepts.id:{ML_CONCEPT},publication_year:{y}",
    }
    for name, make_filter in specs.items():
        for y in YEARS:
            path = OPENALEX_DIR / f"{name}_{y}.json"
            if path.exists():
                continue
            n_groups, counts = openalex_group(make_filter(y))
            truncated = len(counts) >= 200 or n_groups > 200
            direct = {}
            if truncated:
                # Countries outside the top-200 list: query each WB country not returned.
                for iso2 in wb_iso2_codes:
                    if iso2 in counts:
                        continue
                    cnt = openalex_count(f"{make_filter(y)},authorships.countries:{iso2}")
                    if cnt is not None:
                        direct[iso2] = cnt
            path.write_text(json.dumps({"groups_count": n_groups, "truncated": truncated,
                                        "group_counts": counts, "direct_counts": direct}))
            print(f"[openalex] {name} {y}: groups={n_groups} truncated={truncated} direct={len(direct)}")


def assemble_openalex_csv(wb_iso2_codes):
    """Build data/raw/openalex_counts.csv (iso2, year, ai_pubs, all_pubs, ml_pubs).
    A country-year row is absent when OpenAlex returned no group and no direct count."""
    out = RAW / "openalex_counts.csv"
    if out.exists():
        print("[openalex] openalex_counts.csv exists, skipping assembly")
        return
    rows = {}
    for name, col in [("ai", "ai_pubs"), ("all", "all_pubs"), ("ml", "ml_pubs")]:
        for y in YEARS:
            d = json.loads((OPENALEX_DIR / f"{name}_{y}.json").read_text())
            merged = dict(d["group_counts"])
            merged.update(d["direct_counts"])
            for iso2, cnt in merged.items():
                rows.setdefault((iso2, y), {"iso2": iso2, "year": y})[col] = cnt
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iso2", "year", "ai_pubs", "all_pubs", "ml_pubs"])
        w.writeheader()
        for key in sorted(rows, key=lambda k: (k[0], k[1])):
            r = rows[key]
            w.writerow({"iso2": r["iso2"], "year": r["year"],
                        "ai_pubs": r.get("ai_pubs", ""), "all_pubs": r.get("all_pubs", ""),
                        "ml_pubs": r.get("ml_pubs", "")})
    print(f"[openalex] wrote {out} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# 4. OWID energy and CO2
# ---------------------------------------------------------------------------
def step_owid():
    OWID_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in OWID_FILES.items():
        path = OWID_DIR / fname
        if path.exists():
            continue
        save_bytes(path, fetch_bytes(url))
        print(f"[owid] saved {fname}")


# ---------------------------------------------------------------------------
# 5. IMF AI Preparedness Index (2023 cross-section)
# ---------------------------------------------------------------------------
def step_imf():
    json_path = RAW / "imf_aipi.json"
    csv_path = RAW / "imf_aipi.csv"
    if csv_path.exists():
        return
    if not json_path.exists():
        save_bytes(json_path, fetch_bytes(IMF_URL))
    values = json.loads(json_path.read_text())["values"]["AI_PI"]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iso3", "aipi"])
        for iso3, by_year in sorted(values.items()):
            if "2023" in by_year and by_year["2023"] is not None:
                w.writerow([iso3, by_year["2023"]])
    print(f"[imf] wrote {csv_path}")


def wb_iso2_from_metadata():
    """ISO2 codes of non-aggregate World Bank countries (used for OpenAlex direct queries)."""
    meta = json.loads((RAW / "wb_countries.json").read_text())[1]
    return sorted({c["iso2Code"] for c in meta
                   if c["region"]["value"] != "Aggregates" and c["iso2Code"]})


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    step_ndgain()
    step_worldbank()
    step_owid()
    step_openalex(wb_iso2_from_metadata())
    assemble_openalex_csv(wb_iso2_from_metadata())
    # The IMF index is only used by the cross-sectional table (T9). A failed fetch is
    # reported and the rest of the pipeline still runs; T9 is then skipped.
    try:
        step_imf()
    except RuntimeError as err:
        print(f"[imf] WARNING: {err}. T9 will be skipped until data/raw/imf_aipi.csv exists.")
    print("Download step complete.")


if __name__ == "__main__":
    main()
