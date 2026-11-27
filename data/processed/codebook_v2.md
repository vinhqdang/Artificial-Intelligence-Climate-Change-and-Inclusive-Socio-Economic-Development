# Codebook: data layer v2

Produced by `python code/10_download_v2.py && python code/11_build_v2.py`.
Outputs: `data/processed/country_year_v2.csv` and `data/processed/spells_deciles_v2.csv`.
Design reference: `docs/research_design_v2.md`, sections 2, 5 and 7.

## 1. Samples

| Item | Count |
|---|---|
| Country universe (World Bank non-aggregate countries, population >= 1 million in 2010) | 159 |
| Country-year rows, 1998-2023 | 4,134 (159 countries x 26 years) |
| Countries with country-year temperature available | 157 (Singapore and Kosovo have no OWID temperature series) |
| Survey rows in PIP, universe, all levels and welfare types | 2,410 (2,303 national; 36 urban-only rows of ARG; 71 urban/rural rows of countries that also have a national series, not used) |
| National survey rows after the one-welfare-type rule | 2,123 kept, 180 dropped (ARG's 36 urban-only rows are used under the level rule) |
| Consecutive survey pairs (same country) | 2,015 |
| Pairs kept (gap 1-10 years and same survey_comparability) | 1,723 spells, 130 countries |
| Spells with T_climai at y0 available | 1,343 (y0 >= 2001 required; 374 spells with y0 < 2001 have T missing) |
| Spells with S_spell available | 1,649 (74 missing; 36 of them with y0 < 1980) |
| Spells in the main complete-case sample (`complete_main = 1`) | 1,283 spells, 123 countries, y0 in 2001-2022 |
| Spell x decile rows | 17,230 (one row per spell and decile d = 1..10) |

Share T = 1 (T_climai):
- country-years with T available: 0.447 (n = 3,634)
- spells with T available: 0.653 (n = 1,343)
- spells in the main sample: 0.675 (n = 1,283)

Income group of the 159 universe countries: High 50, Upper middle 44, Lower middle 40, Low 25. EMDE = 109. Regions: Europe & Central Asia 46, Sub-Saharan Africa 44, Latin America & Caribbean 23, Middle East/North Africa/Afghanistan/Pakistan 21, East Asia & Pacific 19, South Asia 4, North America 2.

## 2. Country-year file: `country_year_v2.csv`

Key: `iso3`, `year` (1998-2023). Rows are sorted by iso3, year.

| Variable | Definition | Source | Unit |
|---|---|---|---|
| iso3, iso2, country | World Bank country codes and name | `data/raw/wb_countries.json` | - |
| region | World Bank region | same | - |
| income_group | World Bank income level | same | - |
| emde | 1 if income_group != "High income" | derived | 0/1 |
| year | calendar year | - | - |
| temp | annual mean surface temperature, mean of 12 monthly values (12 months required) | OWID grapher `average-monthly-surface-temperature` (underlying dataset as published by OWID) | deg C |
| shock | temp minus country linear trend fitted by OLS on 1980-2023 (all available years in that window) | derived from temp | deg C |
| shock_hot | max(shock, 0). Main hot-side variant | derived | deg C |
| shock_sd_i | SD of shock over 2000-2023 (requires >= 20 years), constant within country | derived | deg C |
| base_temp | mean temp over 1980-2000 (requires >= 15 years), constant within country | derived | deg C |
| slope_temp_trend | slope of the country trend fitted on 1980-2023, constant within country | derived | deg C per year |
| precip_mm | annual precipitation | OWID grapher `average-precipitation-per-year` | mm |
| precip_shock | precipitation minus country linear trend (1980-2023), same method as shock. Not part of the main design, robustness only | derived | mm |
| gdppc_growth | GDP per capita growth | WDI NY.GDP.PCAP.KD.ZG | % |
| agri_growth | agriculture value added growth | WDI NV.AGR.TOTL.KD.ZG | % |
| ln_gdppc | log GDP per capita, PPP, constant 2021 int$ (as in WDI series) | WDI NY.GDP.PCAP.PP.KD | log |
| gini | Gini index. Linear interpolation inside the observed WDI range only (1995-2024), no extrapolation | WDI SI.POV.GINI | 0-100 |
| internet | individuals using the internet, not interpolated | WDI IT.NET.USER.ZS | % of population |
| agri_share | agriculture value added share of GDP | WDI NV.AGR.TOTL.ZS | % |
| urban | urban population share | WDI SP.URB.TOTL.IN.ZS | % |
| pop | total population | WDI SP.POP.TOTL | persons |
| ln_pop | log population | derived | log |
| readiness | ND-GAIN readiness, lagged one year (value at t-1), x100 | ND-GAIN country index (readiness) | index 0-100 |
| readiness_t0 | ND-GAIN readiness, same year, x100 | same | index 0-100 |
| all_works | OpenAlex works, all, by publication year, with at least one author from the country | OpenAlex (10_download_v2) | count |
| ai_works | works with concept C154945302 (Artificial intelligence) OR C119857082 (Machine learning) | OpenAlex | count |
| climai_works | ai_works AND concept C132651083 (Climate change) | OpenAlex | count |
| ai_pc | ai_works per million population | derived | per 1e6 |
| climai_pc | climai_works per million population | derived | per 1e6 |
| climai_pc_3y | 3-year rolling sum of climai_pc over t-2..t (all three years required) | derived | per 1e6 |
| T_climai | 1 if climai_pc_3y(t-1) > median across universe countries of climai_pc_3y(t-1) in that year. Missing if the lag is missing. Available from 2001 | derived | 0/1 |
| ln_ai_pc_lag | log(1 + ai_pc(t-1)) | derived | log |
| period | OpenAlex period containing year: 1998-2007, 2008-2015, 2016-2023 | derived | - |
| climai_domestic | climate-AI works in the period with exactly one distinct country (countries_distinct_count:1) and that country equal to the country | OpenAlex (filters verified) | count |
| climai_period_n | climate-AI works of the country in the period (all affiliations) | OpenAlex | count |
| sovereignty | climai_domestic / climai_period_n. Missing if climai_period_n < 5 | derived | share 0-1 |
| ren_share | share of renewables in electricity generation | OWID energy-data `renewables_share_elec` | % |
| ln_elec_pc | log electricity per capita (kWh) | OWID energy-data `per_capita_electricity` | log |
| ln_co2pc | log CO2 per capita (tonnes) | OWID co2-data `co2_per_capita` | log |

Country-year missingness (of 4,134 rows): temp 1.3% (Singapore and Kosovo), gdppc_growth 1.9%, agri_growth 7.2%, ln_gdppc 4.1%, gini 26.7%, internet 3.9%, agri_share 4.3%, readiness 3.1%, T_climai 12.1%, sovereignty 55.5%, ren_share 5.0%, ln_co2pc 1.3%.

## 3. Survey spell file: `spells_deciles_v2.csv`

Long format. One row per spell x decile (`d` = 1..10). `spell_id` = `iso3_y0_y1`. Covariates are constant across the 10 decile rows of a spell.

| Variable | Definition | Source | Unit |
|---|---|---|---|
| spell_id | iso3_y0_y1 | derived | - |
| iso3, country, region, income_group, emde | as in country-year file | - | - |
| d | decile, 1 = poorest | PIP | 1..10 |
| y0, y1 | reporting years of the two consecutive surveys (`reporting_year`) | PIP | year |
| gap | y1 - y0, between 1 and 10 | derived | years |
| weight | spell weight = gap | derived | years |
| reporting_level | "national" (1,691 spells) or "urban" (ARG only, 32 spells); counts are at d = 1 | PIP | - |
| welfare_type | "income" or "consumption", one per country (most survey rows) | PIP | - |
| survey_comparability | value at y0 (equal at y1 by construction) | PIP | code |
| mean_0, mean_1 | survey mean at y0 and y1 | PIP `mean` | PPP $/day (2017 PPP) |
| y_d0, y_d1 | decile mean income = mean x decile_d share x 10 | derived | PPP $/day |
| Y | annualised log growth of decile mean: (ln y_d1 - ln y_d0) / gap x 100 | derived | % per year |
| S_spell | mean of country annual shock over years y0+1..y1 (all years required) | derived | deg C |
| T_climai_y0 | T_climai in the country-year y0 (exact year, no lag search) | derived | 0/1 |
| ln_gdppc, gini, internet, agri_share, urban, ln_pop, readiness, ln_ai_pc_lag, sovereignty, ren_share | covariates at y0. If missing at y0, the nearest earlier non-missing value within 3 years | country-year file | as in country-year |
| base_temp, shock_sd_i | country constants | country-year file | deg C |
| complete_main | 1 if Y, S_spell, T_climai_y0, ln_gdppc, gini, internet, agri_share, urban, ln_pop, readiness, ln_ai_pc_lag and base_temp are all non-missing. sovereignty and ren_share are excluded from this definition | derived | 0/1 |

Summary (spell x decile rows):
- Y: n = 17,230 (all spells). Mean 2.30, SD 7.15, p5 -7.73, median 2.07, p95 12.66. Mean by decile: d1 2.83, d2 2.62, d3 2.50, d4 2.39, d5 2.32, d6 2.26, d7 2.19, d8 2.11, d9 1.99, d10 1.80.
- Main sample (complete_main = 1): n = 12,830. Mean 2.51, SD 6.42, median 2.12, min -53.04, max 69.20.
- S_spell (spell level, all with S): n = 1,649. Mean -0.016, SD 0.439, min -2.25, max 1.81.

Missingness of covariates (spell level, d = 1 rows, n = 1,723): S_spell 4.3%, T_climai_y0 22.1%, ln_gdppc 17.8%, gini 17.2%, internet 17.9%, agri_share 18.0%, urban 17.2%, ln_pop 17.2%, readiness 18.2%, ln_ai_pc_lag 18.9%, sovereignty 46.3%, ren_share 18.9%, base_temp 0.6%.

Most of the missingness in covariates comes from spells starting before 1998, where no country-year covariate exists.

## 4. Sources and access

| Source | Access | Status |
|---|---|---|
| OpenAlex works API | public, no mailto, plain User-Agent, 0.25 s sleep, retries on 429/5xx | all calls OK except 40 HTTP 400 for country code JG (Channel Islands), which OpenAlex rejects. The failure log contains no 403 or other refusals. Full log: `data/raw/download_failures_v2.json` |
| OWID grapher temperature (monthly) | HTTP 200 | used. Fallback (annual anomalies) not needed |
| OWID grapher precipitation | HTTP 200 | used |
| World Bank PIP | HTTP 200, `povline=2.15`, all countries, all years | 2,588 rows, 172 countries |
| World Bank WDI | HTTP 200 for the three new indicators (NV.AGR.TOTL.ZS, NV.AGR.TOTL.KD.ZG, NY.GDP.PCAP.KD.ZG). The other indicators come from `data/raw/wdi` (code/01_download.py) | used |
| ND-GAIN | manually downloaded archive, unzipped by code/01_download.py | used |
| OWID energy and CO2 | from code/01_download.py | used |

Cache: OpenAlex responses are in `data/raw/openalex_v2/` (JSON). Count tables: `data/raw/openalex_v2_counts.csv` (iso2, year, all_pubs, ai_pubs, climai_pubs) and `data/raw/openalex_v2_domestic.csv` (iso2, period, climai_domestic, climai_total). The `climai_total` column is an extra column added to the requested file layout so that sovereignty can be computed.

## 5. Deviations from the design and open choices

1. **Temperature source.** The design names ERA5. The OWID grapher series was used as the design allows. The underlying dataset is as published by OWID and was not verified independently. Annual values are means over 12 monthly values; a year with fewer than 12 months is missing.
2. **Singapore and Kosovo** have no temperature series, so they have missing shock and base_temp. They remain in the panel with missing temperature variables.
3. **Trend fit.** The fit uses all available years in 1980-2023 (OWID temperature series starts in 1940 but only 1980+ is used). Shock is defined for 1980-2023 only.
4. **Climate-AI concept.** Only climate change (C132651083) was added to AI or ML (C154945302 | C119857082). "Climate model" and "Adaptation" were not added. OpenAlex concepts are a legacy classification that OpenAlex has since replaced by topics. Counts are therefore concept-based and may change if the classifier changes.
5. **group_by truncation.** OpenAlex reports `groups_count` capped at 200, so truncation is detected from the number of groups returned (200 or more). In truncated years, missing countries were queried one by one with `authorships.countries:{iso2}`. In non-truncated years, countries absent from the list are recorded as 0.
6. **Domestic-only counts.** Computed per period (1998-2007, 2008-2015, 2016-2023), not per year. Sovereignty is therefore constant within a period. climai_total is the denominator and is required to be >= 5.
7. **T_climai.** The 3-year window needs 1998-2000 OpenAlex data, so T is first available in 2001. The median is taken over the 159 universe countries with non-missing values in that year. Strict inequality is used.
8. **Readiness.** The design lists readiness as lagged. The lagged value is `readiness`, and the contemporaneous value is `readiness_t0`.
9. **Renewable share.** Defined as renewables share of electricity (OWID `renewables_share_elec`), since the design did not specify the base. The primary energy share is available in the raw OWID file if preferred.
10. **ln_ai_pc_lag** is log(1 + ai_pc(t-1)) because ai_pc can be zero.
11. **climai_pc_3y** is the sum of the three annual per-million values (not per-million of the summed counts). The two differ only by the population change over the window.
12. **Gini interpolation.** Linear inside the observed WDI range only. The other WDI variables are not interpolated.
13. **Universe.** World Bank non-aggregate countries with population >= 1 million in 2010 (WDI SP.POP.TOTL), 159 countries. Spells are restricted to the same universe. Countries outside the universe are not in either file.
14. **EMDE.** Defined as income_group != "High income". The universe contains no "Not classified" countries, so EMDE = Low/Lower-middle/Upper-middle.
15. **PIP reporting level.** National where a national series exists; otherwise the only available level. ARG has only an urban series and is included under that rule (32 spells). No country was skipped.
16. **PIP welfare type.** One type per country: the type with most survey rows (ties broken alphabetically). This drops 180 national survey rows (33 countries had both income and consumption series).
17. **PIP years.** `reporting_year` is used for y0 and y1, not `survey_year`.
18. **Spell pairing.** Consecutive surveys of a country are paired. Pairs whose gap is outside 1-10 years, or whose survey_comparability changes, are dropped rather than spanned: 2,015 pairs, 1,723 kept. `comparable_spell` is not used.
19. **S_spell** is the mean of annual shocks over years y0+1..y1 (the growth interval), as in the build instructions. Section 7 of the design says "spell years"; the two differ by the start year.
20. **Covariates at y0.** Fallback to up to 3 earlier years only when y0 is missing. Spells starting before 1998 have no country-year covariates, and spells starting before 2001 have no T. These spells are kept with missing values and are not imputed. The main analysis sample is defined by `complete_main`.
21. **Decile means** use PIP decile shares x 10 x mean. Every spell's shares sum to 1 within 0.01 (0 exceptions).
22. **Precipitation.** precip_shock is a mm deviation (not a percent) from the trend. It is included for the robustness variant only and is not used in the main variables.
23. **Not built.** Robustness variants (absolute shock, precipitation added to S) are not constructed as separate variables. ln_elec_pc and ln_co2pc are in the country-year file for the cost module but are not in the spell file. Region dummies are left to the analysis step; `region` is provided as a string.
24. **Codebook counts** were taken from the script output and checked with separate queries after the build.
