# Agricultural yield resilience: application results

Outcome dly (100 x log change in cereal yield), hotm exposure, T_res at t-1, one-year lagged controls, 20 fold seeds (42-61), Gamma = median across seeds. k = 0.3; B = 500; random draws 1000.

## sample

- panel rows 3816, countries 159; analysis rows 3336, countries 145, years 2000-2023; share T_res=1 0.520; EMDE countries 99
- missing shares (panel): dly 0.036, hotm 0.013, T_res 0.041, gini 0.257, ln_fert_l1 0.029, agri_empl 0.007, precip_shock 0.013; gini dropped from X (missing > 25%), kept in Z via country mean
- dropped |dly| >= 100 (data errors): 18 country-years (12_build_agri.py)

## controls

- CONTROL TIMING CHECK (before revision): ln_gdppc, internet, agri_share, urban, ln_pop, ln_fert were CONTEMPORANEOUS; ln_ai_pc_lag was lagged; readiness was already one-year lagged (t-1) in the build; base_temp time-invariant; precip_shock contemporaneous (weather).
- REVISION: ln_gdppc, internet, agri_share, urban, ln_pop and ln_fert enter as one-year lags (_l1); readiness is not lagged again (already t-1; a further shift would make it t-2).

## firststage

- all countries: beta -3.572 (SE 0.530, p 0.0000), N 3679, countries 155
- EMDE only: beta -2.684 (SE 0.759, p 0.0004), N 2538, countries 107
- base_temp >= 20: beta -0.336 (SE 0.908, p 0.7111), N 1971, countries 83
- annual shock (all): beta -0.612 (SE 1.006); N 3679

## fit

- EW-DRPL over 20 fold seeds in 24s; X = 17 covariates (lagged controls)

## ate

- ATE of rho (median-over-seeds Gamma, mean over rows) = 0.843 (cluster SE 1.014; N 3336, countries 145)
- seed distribution of ATE: min -2.563, p25 0.137, median 1.022, p75 2.084, max 7.469
- beta(T=1) -1.446; beta(T=0) -1.593; overall mean psi* -1.517
- Gamma tails: p50 |Gamma| 11.086, p95 107.367, p99 297.474, max 939.01

## split

- split-adjusted ATE: median theta 1.022, SE_adj 2.118, 95% CI [-3.128, 5.173] (per-seed theta range -2.563 to 7.469; per-seed SE median 1.545)
- split-adjusted beta(T=1) (psi*): -1.489, SE_adj 1.188, CI [-3.817, 0.839]; beta(T=0): -1.310, SE_adj 1.214, CI [-3.689, 1.070]

## inference

- mean-over-seeds ATE 1.185 (SE 1.300); winsorised median ATE 0.505 (SE 0.738); winsorisation bounds -204.542, 172.094
- two-way cluster SE 0.937; Driscoll-Kraay SE 1.172 (lag 2); cluster SE 1.014

## reconciliation

- (a) TWFE country+year FE on estimation sample: -3.579 (SE 0.552), N 3336
- (b) year+region FE + X (no country FE): -3.171 (SE 0.531), N 3336
- (c) mean psi* overall (median over seeds): -1.517; split-adjusted -1.382

## overlap

- T_res propensity: outside [0.1,0.9] 0.485; outside [0.05,0.95] 0.344; mean e T=1 0.784, T=0 0.219
- max |SMD| before 0.546, after IPW 0.275

## hetero

- BLP: const 0.843 (SE 0.998, p 0.398); ln_gdppc 0.285 (SE 3.355, p 0.932); gini -1.632 (SE 1.229, p 0.184); internet 0.134 (SE 1.808, p 0.941); agri_share 0.664 (SE 2.638, p 0.801); base_temp -0.129 (SE 1.455, p 0.929); sov_z -1.576 (SE 1.471, p 0.284)
- GATES quintile means: q1 1.033 (SE 2.570), q2 -1.091 (SE 2.062), q3 -0.182 (SE 2.384), q4 -1.049 (SE 2.395), q5 5.501 (SE 2.700)
- GATES top minus bottom 4.468 (SE 3.728)
- CLAN top minus bottom: ln_gdppc -0.58; gini 1.46; internet -22.16; agri_share 3.32; base_temp -1.18; sov_z -0.12

## baseline

- TWFE dly (lagged controls): hotm -3.386 (SE 0.820); hotm x T_res 0.315 (SE 0.970); T_res 0.515 (SE 0.572); N 3336
- TWFE agri VA growth (lagged controls): hotm -1.766 (SE 0.379); hotm x T_res 0.817 (SE 0.457); N 3305

## policy

- policy sample N = 126 countries (EMDE 86); budget floor(0.3 N) = 37; ASEAN in sample: IDN, LAO, MMR, MYS, PHL, THA, VNM
- countries with positive benefit B_i: eps=1 69; eps=2 69
- held-out matrix (rows: rule eps; columns: evaluation eps 0.0, 0.5, 1.0, 2.0): eps 0.0: 0.1901, 0.1854, 0.0722, -0.2821 | eps 0.5: 0.2535, 0.3441, 0.4057, 0.6691 | eps 1.0: 0.2511, 0.3397, 0.3993, 0.6614 | eps 2.0: 0.2100, 0.2349, 0.1634, -0.1185
- own-eps held-out value eps 0.0: 0.1901 [-0.1749, 0.7447]
- own-eps held-out value eps 0.5: 0.3441 [-0.2526, 1.1333]
- own-eps held-out value eps 1.0: 0.3993 [-0.4034, 1.3045]
- own-eps held-out value eps 2.0: -0.1185 [-1.0971, 0.8433]
- eps 1.0: equity gain 0.3271 [-0.0109, 0.8953]; equity price -0.0610 [-0.1589, 0.0053]
- eps 2.0: equity gain 0.1636 [-0.0366, 0.5247]; equity price -0.0199 [-0.0884, 0.0285]
- Viet Nam treated (full-sample tree): eps 0.0: no, eps 0.5: no, eps 1.0: no, eps 2.0: no
- eps 0.0 rule: root [ln_gdppc <= 8.367]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); treated 31
- eps 0.0 treated (31): BEN, BFA, BIH, CHN, COD, ETH, GIN, GMB, GNB, IDN, IND, LBR, LVA, MDG, MKD, MLI, MMR, MOZ, NER, PHL, RUS, SEN, SLE, TCD, TGO, TUR, TZA, UGA, UKR, USA, UZB
- eps 0.5 rule: root [ln_gdppc <= 8.367]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); treated 31
- eps 0.5 treated (31): BEN, BFA, BIH, CHN, COD, ETH, GIN, GMB, GNB, IDN, IND, LBR, LVA, MDG, MKD, MLI, MMR, MOZ, NER, PHL, RUS, SEN, SLE, TCD, TGO, TUR, TZA, UGA, UKR, USA, UZB
- eps 1.0 rule: root [ln_gdppc <= 8.367]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); treated 31
- eps 1.0 treated (31): BEN, BFA, BIH, CHN, COD, ETH, GIN, GMB, GNB, IDN, IND, LBR, LVA, MDG, MKD, MLI, MMR, MOZ, NER, PHL, RUS, SEN, SLE, TCD, TGO, TUR, TZA, UGA, UKR, USA, UZB
- eps 2.0 rule: root [internet <= 7.105]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); treated 22
- eps 2.0 treated (22): BFA, BIH, CHN, COD, ETH, GNB, IDN, IND, LVA, MDG, MKD, MOZ, NER, PHL, RUS, SLE, TCD, TUR, UGA, UKR, USA, UZB
- budget use: m = 37; eps=1 tree treats 31; countries with B>0 = 69; value of tree 1.11773 vs value of treating all B>0 countries 1.41615 (same objective, no budget)
- eps=1 tree leaf sizes (LL, LR, RL, RR): [7, 19, 88, 12]; treated leaves: (False, True, False, True)

## differences

- eps 1.0: EW-DRPL minus poorest-first -0.2479 [-0.7665, 0.1439]; capable-first minus poorest-first -0.6544 [-1.8032, 0.3053]
- eps 2.0: EW-DRPL minus poorest-first -1.3860 [-3.2631, -0.0167]; capable-first minus poorest-first -1.2763 [-3.2683, 0.5785]

## leaves

- eps 1.0 tree: LL [no] n=7: BDI Burundi; MWI Malawi; NPL Nepal; PAK Pakistan; RWA Rwanda; TJK Tajikistan; ZMB Zambia | LR [yes] n=19: BEN Benin; BFA Burkina Faso; COD Congo, Dem. Rep.; ETH Ethiopia; GIN Guinea; GMB Gambia, The; GNB Guinea-Bissau; LBR Liberia; MDG Madagascar; MLI Mali; MMR Myanmar; MOZ Mozambique; NER Niger; SEN Senegal; SLE Sierra Leone; TCD Chad; TGO Togo; TZA Tanzania; UGA Uganda | RL [no] n=88: ALB Albania; ARE United Arab Emirates; ARG Argentina; ARM Armenia; AUS Australia; AUT Austria; AZE Azerbaijan; BEL Belgium; BGD Bangladesh; BGR Bulgaria; BLR Belarus; BOL Bolivia; BRA Brazil; BWA Botswana; CAN Canada; CHE Switzerland; CHL Chile; CIV Cote d'Ivoire; CMR Cameroon; COG Congo, Rep.; COL Colombia; CRI Costa Rica; CYP Cyprus; CZE Czechia; DEU Germany; DNK Denmark; DOM Dominican Republic; ECU Ecuador; EGY Egypt, Arab Rep.; ESP Spain; EST Estonia; FIN Finland; FRA France; GBR United Kingdom; GEO Georgia; GHA Ghana; GRC Greece; GTM Guatemala; HND Honduras; HRV Croatia; HUN Hungary; IRL Ireland; IRN Iran, Islamic Rep.; IRQ Iraq; ISR Israel; ITA Italy; JAM Jamaica; JOR Jordan; JPN Japan; KAZ Kazakhstan; KEN Kenya; KGZ Kyrgyz Republic; KOR Korea, Rep.; LAO Lao PDR; LKA Sri Lanka; LTU Lithuania; MAR Morocco; MDA Moldova; MEX Mexico; MNG Mongolia; MRT Mauritania; MUS Mauritius; MYS Malaysia; NAM Namibia; NGA Nigeria; NIC Nicaragua; NLD Netherlands; NOR Norway; PAN Panama; PER Peru; POL Poland; PRT Portugal; PRY Paraguay; ROU Romania; SLV El Salvador; SRB Serbia; SVK Slovak Republic; SVN Slovenia; SWE Sweden; SWZ Eswatini; SYR Syrian Arab Republic; THA Thailand; TLS Timor-Leste; TUN Tunisia; URY Uruguay; VNM Viet Nam; ZAF South Africa; ZWE Zimbabwe | RR [yes] n=12: BIH Bosnia and Herzegovina; CHN China; IDN Indonesia; IND India; LVA Latvia; MKD North Macedonia; PHL Philippines; RUS Russian Federation; TUR Turkiye; UKR Ukraine; USA United States; UZB Uzbekistan
- eps 2.0 tree: LL [no] n=3: BDI Burundi; MWI Malawi; ZMB Zambia | LR [yes] n=10: BFA Burkina Faso; COD Congo, Dem. Rep.; ETH Ethiopia; GNB Guinea-Bissau; MDG Madagascar; MOZ Mozambique; NER Niger; SLE Sierra Leone; TCD Chad; UGA Uganda | RL [no] n=101: ALB Albania; ARE United Arab Emirates; ARG Argentina; ARM Armenia; AUS Australia; AUT Austria; AZE Azerbaijan; BEL Belgium; BEN Benin; BGD Bangladesh; BGR Bulgaria; BLR Belarus; BOL Bolivia; BRA Brazil; BWA Botswana; CAN Canada; CHE Switzerland; CHL Chile; CIV Cote d'Ivoire; CMR Cameroon; COG Congo, Rep.; COL Colombia; CRI Costa Rica; CYP Cyprus; CZE Czechia; DEU Germany; DNK Denmark; DOM Dominican Republic; ECU Ecuador; EGY Egypt, Arab Rep.; ESP Spain; EST Estonia; FIN Finland; FRA France; GBR United Kingdom; GEO Georgia; GHA Ghana; GIN Guinea; GMB Gambia, The; GRC Greece; GTM Guatemala; HND Honduras; HRV Croatia; HUN Hungary; IRL Ireland; IRN Iran, Islamic Rep.; IRQ Iraq; ISR Israel; ITA Italy; JAM Jamaica; JOR Jordan; JPN Japan; KAZ Kazakhstan; KEN Kenya; KGZ Kyrgyz Republic; KOR Korea, Rep.; LAO Lao PDR; LBR Liberia; LKA Sri Lanka; LTU Lithuania; MAR Morocco; MDA Moldova; MEX Mexico; MLI Mali; MMR Myanmar; MNG Mongolia; MRT Mauritania; MUS Mauritius; MYS Malaysia; NAM Namibia; NGA Nigeria; NIC Nicaragua; NLD Netherlands; NOR Norway; NPL Nepal; PAK Pakistan; PAN Panama; PER Peru; POL Poland; PRT Portugal; PRY Paraguay; ROU Romania; RWA Rwanda; SEN Senegal; SLV El Salvador; SRB Serbia; SVK Slovak Republic; SVN Slovenia; SWE Sweden; SWZ Eswatini; SYR Syrian Arab Republic; TGO Togo; THA Thailand; TJK Tajikistan; TLS Timor-Leste; TUN Tunisia; TZA Tanzania; URY Uruguay; VNM Viet Nam; ZAF South Africa; ZWE Zimbabwe | RR [yes] n=12: BIH Bosnia and Herzegovina; CHN China; IDN Indonesia; IND India; LVA Latvia; MKD North Macedonia; PHL Philippines; RUS Russian Federation; TUR Turkiye; UKR Ukraine; USA United States; UZB Uzbekistan
- Viet Nam eps=2 leaf: RL (treated False); Z: ln GDP pc 8.913, gini 36.26, internet 33.996, agri share 17.38, base temp 23.54, sov_z 0.309
- Viet Nam B(eps=2) -0.02955; B(eps=1) -0.07184; agri share 0.463

## unweighted

- eps 0.0: tree root [gini <= 36.2]; left [internet <= 39.46]; right [base_temp <= 26.43]; treated leaves (LL, LR, RL, RR) = (True, False, False, True); treated 35: ALB, BEN, BFA, BGD, BIH, EGY, ETH, GHA, GIN, GMB, GNB, IDN, IND, IRQ, JOR, KGZ, LAO, LBR, LKA, MDA, MLI, MMR, MNG, MRT, NER, NGA, NPL, PAK, SEN, SLE, TGO, TJK, TLS, UKR, UZB
- eps 0.0: held-out value -0.1258 [-0.5322, 0.2634]
- eps 0.5: tree root [ln_gdppc <= 8.367]; left [base_temp <= 22.17]; right [base_temp <= 6.246]; treated leaves (LL, LR, RL, RR) = (False, True, True, False); treated 31: ARM, AUT, BEN, BFA, CAN, CHE, COD, EST, ETH, FIN, GIN, GMB, GNB, KGZ, LBR, LVA, MDG, MLI, MMR, MNG, MOZ, NER, NOR, RUS, SEN, SLE, SWE, TCD, TGO, TZA, UGA
- eps 0.5: held-out value -0.3260 [-1.1217, 0.3763]
- eps 1.0: tree root [ln_gdppc <= 8.367]; left [base_temp <= 22.17]; right [base_temp <= 6.246]; treated leaves (LL, LR, RL, RR) = (False, True, True, False); treated 31: ARM, AUT, BEN, BFA, CAN, CHE, COD, EST, ETH, FIN, GIN, GMB, GNB, KGZ, LBR, LVA, MDG, MLI, MMR, MNG, MOZ, NER, NOR, RUS, SEN, SLE, SWE, TCD, TGO, TZA, UGA
- eps 1.0: held-out value -0.0927 [-1.1845, 0.8490]
- eps 2.0: tree root [ln_gdppc <= 7.791]; left [base_temp <= 22.17]; right [ln_gdppc <= 8.367]; treated leaves (LL, LR, RL, RR) = (False, True, True, False); treated 23: BEN, BFA, COD, ETH, GIN, GMB, GNB, LBR, MDG, MLI, MMR, MOZ, NER, NPL, PAK, SEN, SLE, TCD, TGO, TJK, TZA, UGA, ZMB
- eps 2.0: held-out value 0.0274 [-0.7497, 0.9058]
- eps 0.0: Jaccard vs weighted main rule 0.375
- eps 0.5: Jaccard vs weighted main rule 0.512
- eps 1.0: Jaccard vs weighted main rule 0.512
- eps 2.0: Jaccard vs weighted main rule 0.286

## uniform

- uniform exposure eps=2: treated 22; Jaccard vs main eps=2 1.000; held-out value -0.2298 [-0.8773, 0.3613]; tree root [internet <= 7.105]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True)
- uniform exposure treated (eps=2): BFA, BIH, CHN, COD, ETH, GNB, IDN, IND, LVA, MDG, MKD, MOZ, NER, PHL, RUS, SLE, TCD, TUR, UGA, UKR, USA, UZB

## budget

- k=0.2 eps=0.0: treated 22; root [internet <= 7.105]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); held-out -0.0596
- k=0.2 eps=0.5: treated 22; root [internet <= 7.105]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); held-out 0.0258
- k=0.2 eps=1.0: treated 22; root [internet <= 7.105]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); held-out 0.0778
- k=0.2 eps=2.0: treated 22; root [internet <= 7.105]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); held-out -0.2818
- k=0.2 eps=1.0: equity gain 0.2576 [-0.2093, 0.8320]; price -0.0716 [-0.2030, 0.0289]
- k=0.2 eps=2.0: equity gain -0.2348 [-1.1830, 0.4225]; price -0.0357 [-0.1514, 0.0472]
- k=0.5 eps=0.0: treated 31; root [ln_gdppc <= 8.367]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); held-out 0.3322
- k=0.5 eps=0.5: treated 31; root [ln_gdppc <= 8.367]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); held-out 0.4357
- k=0.5 eps=1.0: treated 31; root [ln_gdppc <= 8.367]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); held-out 0.4246
- k=0.5 eps=2.0: treated 22; root [internet <= 7.105]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); held-out 0.4399
- k=0.5 eps=1.0: equity gain 0.1760 [-0.1611, 0.7048]; price -0.0356 [-0.1291, 0.0551]
- k=0.5 eps=2.0: equity gain 0.5883 [-0.3352, 2.2533]; price 0.1205 [-0.0868, 0.4239]

## sensitivity

- (a) raw psi: ATE 0.727 (SE 1.126); Jaccard eps=1 0.710, eps=2 0.514
- (b) propensity trimming [0.05, 0.95]: rows 3336, countries 126; ATE 0.978 (SE 1.070); split-adjusted 1.356 (SE 2.161); Jaccard eps=1 1.000, eps=2 1.000 (treated 31, 22)
- (c) annual shock instead of hotm: rows 3336, countries 126; ATE -5.594 (SE 1.916); split-adjusted -5.392 (SE 3.172); Jaccard eps=1 0.208, eps=2 0.111 (treated 33, 28)
- (d) random forest nuisances: rows 3336, countries 126; ATE -0.138 (SE 0.760); split-adjusted -0.056 (SE 0.919); Jaccard eps=1 0.710, eps=2 0.417 (treated 22, 12)
- (e) EMDE-only sample: rows 2247, countries 86; ATE 0.669 (SE 1.417); split-adjusted 1.955 (SE 4.600); Jaccard eps=1 0.514, eps=2 0.278 (treated 25, 24)
- (f) T_stock instead of T_res: rows 3336, countries 126; ATE 2.258 (SE 1.672); split-adjusted 2.796 (SE 2.768); Jaccard eps=1 0.143, eps=2 0.085 (treated 33, 29)
- (g) agricultural VA growth outcome: rows 3305, countries 125; ATE 0.001 (SE 0.610); split-adjusted 0.066 (SE 1.014); Jaccard eps=1 0.018, eps=2 0.082 (treated 26, 31)
- Jaccard eps=2 by variant: mai 1.000; (a) 0.514; (b) 1.000; (c) 0.111; (d) 0.417; (e) 0.278; (f) 0.085; (g) 0.082; (h) 1.000
- (g) agri VA growth split-adjusted ATE 0.066 (SE 1.014), 95% CI [-1.923, 2.054]

## revision

Items refer to the referee-review round. Every number below is from the revised run that wrote this file (`code/31_application_agri.py`, 20 fold seeds 42-61, Gamma = median across seeds unless stated).

### 1. Control timing (bad controls)
- Before revision the regression controls ln GDP pc, internet, agriculture share, urban, ln population and ln fertilizer were CONTEMPORANEOUS (same year as dly). ln AI pc was already lagged. Readiness was already one-year lagged (t-1) in the build, so it is not lagged again (a further shift would make it t-2). Base temperature is time-invariant; precipitation shock is weather and stays contemporaneous.
- Revision: one-year lags (`_l1`) of ln GDP pc, internet, agriculture share, urban, ln population, ln fertilizer in X and in the TWFE baseline. The estimation sample is 3,336 country-years (was 3,338), 145 countries.
- First stage (unchanged, no controls): all countries -3.572 (SE 0.530), N 3,679; EMDE -2.684 (0.759), N 2,538; base_temp >= 20 -0.336 (0.908), N 1,971; annual shock -0.612 (1.006).

### 2. Split-adjusted inference (Chernozhukov et al. 2018)
- ATE of rho: median-over-seeds estimate 1.022; SE_adj 2.118; 95% CI [-3.128, 5.173]. Per-seed theta_s range -2.563 to 7.469; median per-seed cluster SE 1.545. Current median-of-rows estimate: 0.843 (cluster SE 1.014; 95% CI [-1.144, 2.829]).
- Seed distribution of ATE: min -2.563, p25 0.137, median 1.022, p75 2.084, max 7.469.
- beta(T=1) (mean psi*, split-adjusted): -1.489 (SE_adj 1.188), 95% CI [-3.817, 0.839]; cluster-SE version of the row mean: -1.446 (see agri_ate.tex).
- beta(T=0): -1.310 (SE_adj 1.214), 95% CI [-3.689, 1.070].
- Overall mean psi* (median over seeds): -1.517 (cluster SE 0.536); split-adjusted -1.382.
- Agricultural VA growth outcome (variant g), split-adjusted ATE 0.066 (SE_adj 1.014), 95% CI [-1.923, 2.054]; median-of-rows 0.001 (cluster SE 0.610).

### 3. Damage reconciliation (-3.4 vs -1.6)
- (a) TWFE dly on hotm, country + year FE, on the 3,336-row estimation sample: -3.579 (SE 0.552), N 3,336, 145 countries.
- (b) Year FE + region FE + X (lagged, no country FE), same sample: -3.171 (SE 0.531), N 3,336, 145 countries.
- (c) Mean psi* overall (EW-DRPL orthogonal score): -1.517 (SE 0.536); split-adjusted -1.382.
- Reading: most of the gap between the TWFE slope and the orthogonal slope is not from country fixed effects (b is close to a) but from the orthogonal estimator itself (cross-fitted nuisances, 1/v_hat weighting and heavy-tailed scores). The score targets a conditional slope averaged differently from the regression. Not fully resolved here.

### 4. Inference robustness for the ATE (median over seeds)
- Cluster by country (main): 0.843, SE 1.014, CI [-1.144, 2.829].
- Two-way cluster (country, year; Cameron-Gelbach-Miller): SE 0.937, CI [-0.995, 2.680].
- Driscoll-Kraay on yearly sums (Bartlett lag 2): SE 1.172, CI [-1.455, 3.140].
- Split-adjusted over seeds: 1.022, SE_adj 2.118, CI [-3.128, 5.173].
- Conclusion: none of the main ATE intervals excludes zero. The main ATE and the raw-psi, trimming, RF, EMDE, T_stock and agri VA variants are positive or near zero; the annual-shock variant (c) is -5.594 (SE 1.916), which is distinguishable from zero at about 3 SE (see section 6 table) and is discussed in the final report.

### 5. Mean-aggregated vs median-aggregated; winsorised outlier check
- Mean over seeds: 1.185 (cluster SE 1.300), CI [-1.363, 3.733].
- Median over seeds (main): 0.843 (SE 1.014).
- Winsorised median Gamma at 1st/99th percentiles (bounds -204.542, 172.094): 0.505 (SE 0.738), CI [-0.941, 1.951].
- Gamma tails (rows, median aggregation): p50 |Gamma| 11.09, p95 107.37, p99 297.47, max 939.01.

### 6. Policy robustness
(a) Jaccard stability vs main rule (main eps=1 treated 31; main eps=2 treated 22):
| variant | eps=1 treated | Jaccard eps=1 | eps=2 treated | Jaccard eps=2 |
|---|---|---|---|---|
| main | 31 | 1.000 | 22 | 1.000 |
| (a) raw psi | 22 | 0.710 | 34 | 0.514 |
| (b) trimming [0.05,0.95] | 31 | 1.000 | 22 | 1.000 |
| (c) annual shock | 33 | 0.208 | 28 | 0.111 |
| (d) RF nuisances | 22 | 0.710 | 12 | 0.417 |
| (e) EMDE only | 25 | 0.514 | 24 | 0.278 |
| (f) T_stock | 33 | 0.143 | 29 | 0.085 |
| (g) agri VA growth | 26 | 0.018 | 31 | 0.082 |
| (h) uniform exposure | 22 | 0.710 | 22 | 1.000 |
(b) Uniform exposure at eps=2: 22 treated, Jaccard 1.000 vs main eps=2; held-out value -0.230 (CI [-0.877, 0.361]). Its eps=2 rule uses root internet <= 7.105, the same as the main rule.
(c) Unweighted objective (equal country weights; no n_i scaling; budget floor(0.3 N) = 37 countries):
- eps=0: 35 treated; held-out -0.126 [-0.532, 0.263]; Jaccard vs weighted main rule 0.375.
- eps=0.5: 31 treated; held-out -0.326 [-1.122, 0.376]; Jaccard 0.512.
- eps=1: 31 treated; held-out -0.093 [-1.185, 0.849]; Jaccard 0.512.
- eps=2: 23 treated; held-out 0.027 [-0.750, 0.906]; Jaccard 0.286.
(d) Bootstrap differences in held-out value (B = 500):
- eps=1: EW-DRPL minus poorest-first -0.248 [-0.767, 0.144]; capable-first minus poorest-first -0.654 [-1.803, 0.305].
- eps=2: EW-DRPL minus poorest-first -1.386 [-3.263, -0.017]; capable-first minus poorest-first -1.276 [-3.268, 0.579].
(e) Why 31 (eps <= 1) and 22 (eps = 2) rather than 37: the depth-2 tree treats a union of its four leaves. Countries with positive benefit B_i: 69 (eps = 1 and eps = 2). Leaf sizes for the eps=1 tree (LL, LR, RL, RR) = (7, 19, 88, 12); treated leaves LR and RR give 31. For eps=2 the treated leaves LR (10) and RR (12) give 22. The best depth-2 union is below the budget because no other leaf combination adds positive welfare. Value of the eps=1 tree 1.118 versus 1.416 for treating all 69 positive-benefit countries (no tree or budget restriction), so granularity and the tree family, not the budget, bind.
(f) Leaves (main trees, treated status in brackets):
- eps=1 (root ln GDP pc <= 8.367; left base_temp <= 22.17; right sov_z <= 0.5756):
  - LL [no], n=7: BDI, MWI, NPL, PAK, RWA, TJK, ZMB.
  - LR [yes], n=19: BEN, BFA, COD, ETH, GIN, GMB, GNB, LBR, MDG, MLI, MMR, MOZ, NER, SEN, SLE, TCD, TGO, TZA, UGA.
  - RL [no], n=88: listed in agri_leaves.csv.
  - RR [yes], n=12: BIH, CHN, IDN, IND, LVA, MKD, PHL, RUS, TUR, UKR, USA, UZB.
- eps=2 (root internet <= 7.105; left base_temp <= 22.17; right sov_z <= 0.5756):
  - LL [no], n=3: BDI, MWI, ZMB.
  - LR [yes], n=10: BFA, COD, ETH, GNB, MDG, MOZ, NER, SLE, TCD, UGA.
  - RL [no], n=101: listed in agri_leaves.csv.
  - RR [yes], n=12: the same 12 countries as the eps=1 RR leaf.
- Viet Nam: not treated under any eps (weighted main). At eps=2 it sits in the RL leaf (not root: internet 33.996 > 7.105; right: sov_z 0.309 <= 0.5756) and is NOT treated. Its Z values: ln GDP pc 8.913, gini 36.26, internet 33.996, agricultural share 17.38, base temperature 23.54, sov_z 0.309. B(eps=2) = -0.0296; B(eps=1) = -0.0718. This differs from the previous run, where Viet Nam was treated at eps=2.
(g) Budget sensitivity: k = 0.2 gives 22 treated for every eps (root internet <= 7.105, eps=2 held-out -0.282); k = 0.5 gives 31 treated for eps <= 1 and 22 for eps = 2. Equity gains and prices at k = 0.2: eps=1 gain 0.258 [-0.209, 0.832], price -0.072 [-0.203, 0.029]; eps=2 gain -0.235 [-1.183, 0.423], price -0.036 [-0.151, 0.047]. At k = 0.5: eps=1 gain 0.176 [-0.161, 0.705], price -0.036 [-0.129, 0.055]; eps=2 gain 0.588 [-0.335, 2.253], price 0.121 [-0.087, 0.424].

### 7. Main policy values (budget k = 0.3, population-weighted, bottom-first exposure)
- Held-out own-eps values: eps 0: 0.190 [-0.175, 0.745]; eps 0.5: 0.344 [-0.253, 1.133]; eps 1: 0.399 [-0.403, 1.305]; eps 2: -0.119 [-1.097, 0.843].
- Held-out matrix (rows = rule eps; columns = evaluation eps 0, 0.5, 1, 2): eps 0: 0.190, 0.185, 0.072, -0.282; eps 0.5: 0.254, 0.344, 0.406, 0.669; eps 1: 0.251, 0.340, 0.399, 0.661; eps 2: 0.210, 0.235, 0.163, -0.119.
- Comparators at eps=1 / eps=2: utilitarian 0.072 [-0.689, 1.008] / -0.282 [-1.359, 0.715]; plug-in 0.475 [-0.186, 1.367] / 0.789 [-0.217, 2.082]; capable-first -0.007 [-0.037, 0.016] / -0.009 [-0.053, 0.016]; poorest-first 0.647 [-0.332, 1.788] / 1.268 [-0.617, 3.261]; most-agricultural-first 0.581 [-0.407, 1.697] / 1.169 [-0.641, 3.162]; random 0.213 [-0.075, 0.538] / 0.358 [-0.174, 0.927].
- Equity gain / price: eps=1 gain 0.327 [-0.011, 0.895], price -0.061 [-0.159, 0.005]; eps=2 gain 0.164 [-0.037, 0.525], price -0.020 [-0.088, 0.029].
- Trees: eps 0, 0.5, 1 share one tree (root ln GDP pc <= 8.367; left base_temp <= 22.17; right sov_z <= 0.5756) treating 31; eps=2 tree (root internet <= 7.105; left base_temp <= 22.17; right sov_z <= 0.5756) treats 22: BFA, BIH, CHN, COD, ETH, GNB, IDN, IND, LVA, MDG, MKD, MOZ, NER, PHL, RUS, SLE, TCD, TUR, UGA, UKR, USA, UZB.
- ASEAN in the policy sample: IDN, LAO, MMR, MYS, PHL, THA, VNM. Treated: IDN and PHL at eps 0-1; MMR at eps 0-1; IDN, PHL at eps 2.

### 8. Other revision numbers
- BLP (row Z with lagged controls): const 0.843 (SE 0.998); ln GDP pc 0.285 (3.355); Gini -1.632 (1.229, p 0.184); internet 0.134 (1.808); agricultural share 0.664 (2.638); base temperature -0.129 (1.455); sovereignty -1.576 (1.471, p 0.284). No coefficient is distinguishable from zero.
- GATES quintile means: q1 1.033 (SE 2.570), q2 -1.091 (2.062), q3 -0.182 (2.384), q4 -1.049 (2.395), q5 5.501 (2.700). Top minus bottom 4.468 (SE 3.728).
- Overlap (cross-fitted propensity, T_res): share outside [0.1,0.9] 0.485; outside [0.05,0.95] 0.344; mean e given T=1 0.784, T=0 0.219. Max |SMD| 0.546 before IPW, 0.275 after.
- Baseline TWFE (lagged controls): dly on hotm -3.386 (SE 0.820); hotm x T_res 0.315 (0.970); T_res 0.515 (0.572); N 3,336. Agricultural VA growth: hotm -1.766 (0.379); hotm x T_res 0.817 (0.457); N 3,305.

### 9. What changed and what did not
- ATE sign: positive before (0.258) and after (0.843); not distinguishable from zero for the main ATE (cluster, two-way, Driscoll-Kraay, split-adjusted). The annual-shock variant (c) is negative and distinguishable from zero (-5.594, SE 1.916).
- Policy: the eps=1 equity gain changed sign and is now borderline positive (0.327, CI [-0.011, 0.895]) versus negative before (-0.022). The eps=1 equity price is negative (-0.061, CI [-0.159, 0.005]). That means the held-out eps=1 rule shows higher utilitarian welfare than the held-out utilitarian rule, which is unexpected and reflects cross-fitting noise in the depth-2 trees.
- Viet Nam is not treated at eps=2 (previous run: treated).
- Unexpected: the orthogonal mean psi* (-1.5) is about half the TWFE slope (-3.2 to -3.6) even on the same conditioning set; treat the orthogonal estimator's level with caution.
