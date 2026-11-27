"""
Step 1 (design v2): download and cache every raw input used by code/11_build_v2.py.

Sources
-------
* OpenAlex (public API, no mailto). Per year: works by country for all works, AI or ML
  works, and climate-AI works (AI or ML AND climate change, concept C132651083). Countries
  missing from a capped group_by list (> 200 groups) are queried one by one. Per country and
  period (1998-2007, 2008-2015, 2016-2023): climate-AI works in total and domestic-only
  (countries_distinct_count:1).
* OWID grapher: average monthly surface temperature (fallback: annual temperature anomalies)
  and average precipitation per year.
* World Bank PIP: survey-level distributions (poverty line 2.15 USD/day, all countries, years).
* World Bank WDI: agriculture VA share and growth, GDP per capita growth (added to the
  indicators downloaded by code/01_download.py, which are reused when present).

Behaviour
---------
* Idempotent: every response is cached as JSON or CSV and reused on later runs.
* Every request sleeps 0.25 s. 429 and 5xx responses are retried with exponential backoff.
* 400 responses (filter rejected, e.g. a code that is not an OpenAlex country) are cached as
  tombstones and counted as missing. 403 and other refusals are NOT retried and NOT worked
  around; the source is skipped and the failure is recorded in
  data/raw/download_failures_v2.json.

Outputs
-------
* data/raw/openalex_v2/*.json (response cache)
* data/raw/openalex_v2_counts.csv   (iso2, year, all_pubs, ai_pubs, climai_pubs)
* data/raw/openalex_v2_domestic.csv (iso2, period, climai_domestic, climai_total)
* data/raw/owid_v2/*.csv, data/raw/pip_surveys.csv, data/raw/wdi/<code>.json

Usage: python code/10_download_v2.py
"""
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OA_DIR = RAW / "openalex_v2"
OWID_V2 = RAW / "owid_v2"
WDI_DIR = RAW / "wdi"
WB_META = RAW / "wb_countries.json"
FAILURES = RAW / "download_failures_v2.json"

USER_AGENT = "ai-climate-panel-pipeline/2.0"
SLEEP = 0.25            # seconds after every request
MAX_RETRIES = 4
TIMEOUT = 180

OA_BASE = "https://api.openalex.org/works"
AI_FILTER = "concepts.id:C154945302|C119857082"            # Artificial intelligence OR Machine learning
CLIMAI_FILTER = AI_FILTER + ",concepts.id:C132651083"       # ... AND Climate change
YEARS = range(1998, 2024)
PERIODS = [("1998-2007", 1998, 2007), ("2008-2015", 2008, 2015), ("2016-2023", 2016, 2023)]
GROUP_CAP = 200

OWID_URLS = {
    "average-monthly-surface-temperature.csv":
        "https://ourworldindata.org/grapher/average-monthly-surface-temperature.csv",
    "average-precipitation-per-year.csv":
        "https://ourworldindata.org/grapher/average-precipitation-per-year.csv",
}
TEMP_FALLBACK = ("annual-temperature-anomalies.csv",
                 "https://ourworldindata.org/grapher/annual-temperature-anomalies.csv")
PIP_URL = ("https://api.worldbank.org/pip/v1/pip?country=all&year=all&povline=2.15"
           "&fill_gaps=false&format=csv")
WDI_NEW_CODES = ["NV.AGR.TOTL.ZS", "NV.AGR.TOTL.KD.ZG", "NY.GDP.PCAP.KD.ZG"]

FAILS = []   # failures seen in this run (merged into FAILURES at the end)


def get(url):
    """GET with retries. Returns (status, body); status 200 on success, otherwise the HTTP
    code (0 for network errors). Never changes request headers beyond a plain User-Agent."""
    status, body = 0, None
    for attempt in range(MAX_RETRIES + 1):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                body = resp.read()
            status = 200
            time.sleep(SLEEP)
            return status, body
        except urllib.error.HTTPError as err:
            status = err.code
            time.sleep(SLEEP)
            if status in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            return status, None          # 400/403/404 etc.: not retried
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            status = 0
            time.sleep(SLEEP)
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            return status, None
    return status, body


def record_failure(url, status, note=""):
    FAILS.append({"url": url, "status": status, "note": note})
    print(f"  [skip] HTTP {status} {note}: {url[:160]}")


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------
def oa_fetch(params, cache):
    """Return the parsed JSON response, a tombstone dict for HTTP 400, or None if skipped."""
    if cache.exists():
        d = json.loads(cache.read_text())
        if "http_error" in d:
            FAILS.append({"url": d["url"], "status": d["http_error"], "note": "cached rejection"})
            return None
        return d
    url = OA_BASE + "?" + urllib.parse.urlencode(params)
    status, body = get(url)
    if status == 200:
        d = json.loads(body.decode("utf-8"))
        cache.write_text(json.dumps(d))
        return d
    if status == 400:
        cache.write_text(json.dumps({"http_error": 400, "url": url}))
    record_failure(url, status, "OpenAlex")
    return None


def oa_count(d):
    if d is None:
        return None
    return int(d["meta"]["count"])


def year_filter(measure, year):
    parts = {"all": [], "ai": [AI_FILTER], "climai": [CLIMAI_FILTER]}[measure]
    return ",".join(parts + [f"publication_year:{year}"])


def wb_iso2_codes():
    meta = json.loads(WB_META.read_text())[1]
    return sorted({c["iso2Code"] for c in meta
                   if c["region"]["value"].strip() != "Aggregates" and c["iso2Code"]})


def step_openalex_counts(iso2_list):
    OA_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}                      # (measure, iso2, year) -> int
    for measure in ["all", "ai", "climai"]:
        for y in YEARS:
            f = year_filter(measure, y)
            d = oa_fetch({"filter": f, "group_by": "authorships.countries", "per_page": GROUP_CAP},
                         OA_DIR / f"{measure}_{y}_groups.json")
            if d is None:
                print(f"[openalex] {measure} {y}: group query skipped")
                continue
            n_groups = d["meta"]["groups_count"]
            groups = {g["key"].rsplit("/", 1)[-1]: int(g["count"]) for g in d.get("group_by", [])}
            for iso2, c in groups.items():
                counts[(measure, iso2, y)] = c
            # OpenAlex caps groups_count at 200 in its response, so truncation is detected from
            # the number of groups actually returned (a full list is always < 200 groups).
            truncated = len(groups) >= GROUP_CAP or (n_groups is not None and n_groups > GROUP_CAP)
            n_direct = 0
            if truncated:
                # group_by returned only the top 200 groups: query the missing countries one by one.
                for iso2 in iso2_list:
                    if iso2 in groups:
                        continue
                    dc = oa_fetch({"filter": f + f",authorships.countries:{iso2}", "per_page": 1},
                                  OA_DIR / f"{measure}_{y}_{iso2}.json")
                    c = oa_count(dc)
                    if c is not None:
                        counts[(measure, iso2, y)] = c
                        n_direct += 1
            else:
                # Complete list: a country absent from it has zero works that year.
                for iso2 in iso2_list:
                    counts.setdefault((measure, iso2, y), 0)
            print(f"[openalex] {measure} {y}: groups={n_groups} truncated={truncated} "
                  f"direct={n_direct}")

    iso2_all = sorted({k[1] for k in counts} | set(iso2_list))
    out = RAW / "openalex_v2_counts.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iso2", "year", "all_pubs", "ai_pubs", "climai_pubs"])
        for iso2 in iso2_all:
            for y in YEARS:
                row = [counts.get((m, iso2, y), "") for m in ["all", "ai", "climai"]]
                if all(v == "" for v in row):
                    continue
                w.writerow([iso2, y] + row)
    print(f"[openalex] wrote {out}")


def step_openalex_domestic(iso2_list):
    rows = []
    for iso2 in iso2_list:
        for pname, a, b in PERIODS:
            base = f"publication_year:{a}-{b},authorships.countries:{iso2}"
            tot = oa_count(oa_fetch({"filter": CLIMAI_FILTER + "," + base, "per_page": 1},
                                    OA_DIR / f"climai_total_{iso2}_{pname}.json"))
            dom = oa_count(oa_fetch({"filter": CLIMAI_FILTER + "," + base + ",countries_distinct_count:1",
                                     "per_page": 1},
                                    OA_DIR / f"climai_domestic_{iso2}_{pname}.json"))
            rows.append([iso2, pname, "" if dom is None else dom, "" if tot is None else tot])
    out = RAW / "openalex_v2_domestic.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iso2", "period", "climai_domestic", "climai_total"])
        w.writerows(rows)
    print(f"[openalex] wrote {out} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# OWID, World Bank PIP and WDI
# ---------------------------------------------------------------------------
def fetch_csv_to(path, url, header_prefix):
    if path.exists():
        return True
    status, body = get(url)
    if status != 200 or not body.startswith(header_prefix):
        record_failure(url, status, "CSV not returned")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    print(f"[download] saved {path.relative_to(ROOT)}")
    return True


def step_owid():
    OWID_V2.mkdir(parents=True, exist_ok=True)
    ok = fetch_csv_to(OWID_V2 / "average-monthly-surface-temperature.csv",
                      OWID_URLS["average-monthly-surface-temperature.csv"], b"Entity,Code,Month")
    if not ok:
        print("[owid] monthly temperature unavailable; trying annual anomalies fallback")
        fetch_csv_to(OWID_V2 / TEMP_FALLBACK[0], TEMP_FALLBACK[1], b"Entity,Code,Year")
    fetch_csv_to(OWID_V2 / "average-precipitation-per-year.csv",
                 OWID_URLS["average-precipitation-per-year.csv"], b"Entity,Code,Year")


def step_pip():
    fetch_csv_to(RAW / "pip_surveys.csv", PIP_URL, b"region_name,")


def step_wdi():
    WDI_DIR.mkdir(parents=True, exist_ok=True)
    for code in WDI_NEW_CODES:
        path = WDI_DIR / f"{code}.json"
        if path.exists():
            continue
        url = (f"https://api.worldbank.org/v2/country/all/indicator/{code}"
               "?format=json&per_page=20000&date=1995:2024")
        status, body = get(url)
        if status != 200:
            record_failure(url, status, "WDI")
            continue
        parsed = json.loads(body.decode("utf-8"))
        if not (isinstance(parsed, list) and len(parsed) == 2 and parsed[1]):
            record_failure(url, status, "WDI returned no data")
            continue
        path.write_bytes(body)
        print(f"[wdi] saved {code}")


def merge_failures():
    old = json.loads(FAILURES.read_text()) if FAILURES.exists() else []
    seen = {f["url"] for f in FAILS}
    merged = FAILS + [f for f in old if f["url"] not in seen]
    FAILURES.write_text(json.dumps(merged, indent=1))
    print(f"[download] {len(merged)} recorded failures in {FAILURES.relative_to(ROOT)}")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    iso2_list = wb_iso2_codes()
    print(f"[download] {len(iso2_list)} World Bank country codes")
    step_openalex_counts(iso2_list)
    step_openalex_domestic(iso2_list)
    step_owid()
    step_pip()
    step_wdi()
    merge_failures()
    print("Download step (v2) complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
