# Agricultural yield resilience: application results

Outcome dly (100 x log change in cereal yield), hotm exposure, T_res at t-1, 20 fold seeds (42-61), Gamma = median across seeds. k = 0.3; B = 500; random draws 1000.

## sample

- panel rows 3816, countries 159; analysis rows 3338, countries 145, years 2000-2023; share T_res=1 0.520; EMDE countries 99
- missing shares (panel, 2000-2023): dly 0.036, hotm 0.013, T_res 0.041, gini 0.257, ln_fert 0.027, agri_empl 0.007, precip_shock 0.013; gini dropped from X (missing > 25%), kept in Z via country mean
- dropped |dly| >= 100 (data errors): 18 country-years (see 12_build_agri.py output)

## firststage

- all countries: beta -3.572 (SE 0.530, p 0.0000), N 3679, countries 155
- EMDE only: beta -2.684 (SE 0.759, p 0.0004), N 2538, countries 107
- base_temp >= 20: beta -0.336 (SE 0.908, p 0.7111), N 1971, countries 83
- annual shock (all): beta -0.612 (SE 1.006); N 3679
- binned first-stage plot: FE-residualised slope -3.572 (20 quantile bins)

## fit

- EW-DRPL over 20 fold seeds in 22s; X = 17 covariates

## ate

- ATE of rho (median-over-seeds Gamma, mean over rows) = 0.258; mean-over-seeds 0.184
- seed distribution of ATE: min -2.264, p25 -0.781, median 0.178, p75 0.669, max 3.837
- beta(T=1) (mean psi*) -1.532 (SE 0.699); beta(T=0) -1.632 (SE 0.826)
- Gamma quantiles (rows): p50 10.800, p95 107.421, p99 298.298, max |Gamma| 777.01

## hetero

- Gini missing for all years in some countries: 202 country-year rows filled with the sample mean for Z
- BLP: const 0.258 (SE 1.054); ln_gdppc 0.379 (SE 3.652); gini -1.762 (SE 1.062); internet -0.695 (SE 1.764); agri_share -0.204 (SE 2.489); base_temp 0.623 (SE 1.755); sov_z -0.992 (SE 1.523)
- GATES quintile means: q1 -1.486 (SE 3.014), q2 2.759 (SE 2.306), q3 -0.056 (SE 2.591), q4 -3.823 (SE 2.144), q5 3.902 (SE 1.953)
- GATES top minus bottom 5.388 (SE 3.591)
- CLAN top minus bottom: ln_gdppc -0.38; gini 2.41; internet -19.21; agri_share 1.75; base_temp 0.17; sov_z -0.11

## overlap

- T_res propensity: outside [0.1,0.9] 0.497; outside [0.05,0.95] 0.342; mean e T=1 0.783, T=0 0.216
- max |SMD| before 0.547, after IPW 0.260

## baseline

- TWFE dly: hotm -3.388 (SE 0.804); hotm x T_res 0.245 (SE 1.000); T_res 0.871 (SE 0.575); N 3338
- TWFE agri VA growth: hotm -1.722 (SE 0.378); hotm x T_res 0.848 (SE 0.455); N 3304

## policy

- policy sample N = 126 countries (EMDE 86); income deciles, Z and agriculture available; budget floor(0.3 N) = 37; ASEAN in sample: IDN, LAO, MMR, MYS, PHL, THA, VNM
- mean agricultural employment share (policy sample) 0.284
- held-out matrix (rows: rule eps; columns: evaluation eps 0.0, 0.5, 1.0, 2.0): eps 0.0: 0.1072, 0.1128, 0.0935, 0.2411 | eps 0.5: 0.1002, 0.0996, 0.0729, 0.2122 | eps 1.0: 0.0901, 0.0872, 0.0717, 0.3324 | eps 2.0: 0.0889, 0.0862, 0.0716, 0.3339
- own-eps held-out value eps 0.0: 0.1072 [-0.1730, 0.4907]
- own-eps held-out value eps 0.5: 0.0996 [-0.3466, 0.6445]
- own-eps held-out value eps 1.0: 0.0717 [-0.6491, 0.8071]
- own-eps held-out value eps 2.0: 0.3339 [-1.0400, 1.6721]
- utilitarian rule evaluated at eps 0.0: 0.1072 [-0.1730, 0.4907]
- utilitarian rule evaluated at eps 0.5: 0.1128 [-0.3388, 0.6552]
- utilitarian rule evaluated at eps 1.0: 0.0935 [-0.5586, 0.7358]
- utilitarian rule evaluated at eps 2.0: 0.2411 [-1.0296, 1.5783]
- eps 1.0: equity gain -0.0218 [-0.1762, 0.1447]; equity price 0.0170 [-0.0176, 0.0600]
- eps 2.0: equity gain 0.0928 [-0.2326, 0.5778]; equity price 0.0183 [-0.0164, 0.0642]
- eps 0.0 rule (full sample): root [internet <= 8.292]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True)
- eps 0.0 treated (22): BFA, BIH, CHN, COD, ETH, GNB, IDN, IND, LVA, MDG, MKD, MOZ, NER, PHL, RUS, SLE, TCD, TUR, UGA, UKR, USA, UZB
- eps 0.5 rule (full sample): root [internet <= 8.292]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True)
- eps 0.5 treated (22): BFA, BIH, CHN, COD, ETH, GNB, IDN, IND, LVA, MDG, MKD, MOZ, NER, PHL, RUS, SLE, TCD, TUR, UGA, UKR, USA, UZB
- eps 1.0 rule (full sample): root [internet <= 8.292]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True)
- eps 1.0 treated (22): BFA, BIH, CHN, COD, ETH, GNB, IDN, IND, LVA, MDG, MKD, MOZ, NER, PHL, RUS, SLE, TCD, TUR, UGA, UKR, USA, UZB
- eps 2.0 rule (full sample): root [sov_z <= 0.2593]; left [internet <= 8.292]; right [base_temp <= 22.17]; treated leaves (LL, LR, RL, RR) = (True, False, False, True)
- eps 2.0 treated (31): ARE, BDI, BFA, BGD, BRA, CIV, COD, COL, EGY, ETH, GHA, GNB, IDN, IND, JAM, LKA, MDG, MOZ, MYS, NER, NGA, PAN, PHL, SEN, SLE, TCD, THA, TZA, UGA, VNM, ZMB
- Viet Nam treated (full-sample tree): eps 0.0: no, eps 0.5: no, eps 1.0: no, eps 2.0: yes

## budget

- k=0.2 eps=0.0: treated 22; root [internet <= 8.292]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); held-out -0.0642
- k=0.2 eps=0.5: treated 22; root [internet <= 8.292]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); held-out -0.1186
- k=0.2 eps=1.0: treated 22; root [internet <= 8.292]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); held-out -0.1559
- k=0.2 eps=2.0: treated 12; root [ln_gdppc <= 7.81]; left [base_temp <= 22.17]; right [internet <= 8.292]; treated leaves (LL, LR, RL, RR) = (False, True, True, False); held-out 0.3967
- k=0.2 eps=1.0: equity gain -0.0012 [-0.1590, 0.1555]; price 0.0101 [-0.0228, 0.0530]
- k=0.2 eps=2.0: equity gain 0.3102 [-0.1506, 0.9638]; price -0.0333 [-0.1507, 0.0429]
- k=0.5 eps=0.0: treated 22; root [internet <= 8.292]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); held-out 0.1850
- k=0.5 eps=0.5: treated 22; root [internet <= 8.292]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); held-out 0.1601
- k=0.5 eps=1.0: treated 22; root [internet <= 8.292]; left [base_temp <= 22.17]; right [sov_z <= 0.5756]; treated leaves (LL, LR, RL, RR) = (False, True, False, True); held-out 0.1077
- k=0.5 eps=2.0: treated 59; root [sov_z <= 0.2593]; left [sov_z <= 0.1532]; right [base_temp <= 22.17]; treated leaves (LL, LR, RL, RR) = (True, False, False, True); held-out 0.3257
- k=0.5 eps=1.0: equity gain -0.0420 [-0.3438, 0.2162]; price 0.0146 [-0.0495, 0.0968]
- k=0.5 eps=2.0: equity gain 0.1033 [-0.2985, 0.7213]; price 0.1127 [-0.0285, 0.3121]

## sensitivity

- (a) raw psi: ATE 0.129 (SE 1.139); Jaccard 0.645
- (b) propensity trimming [0.05, 0.95]: N rows 3338, countries 126; ATE 0.507 (SE 1.172); seed median 0.511 [min -2.203, max 4.405]; eps=1 treated 22; Jaccard 1.000
- (c) annual shock instead of hotm: N rows 3338, countries 126; ATE -6.060 (SE 2.050); seed median -6.244 [min -12.579, max -2.853]; eps=1 treated 7; Jaccard 0.208
- (d) random forest nuisances: N rows 3338, countries 126; ATE 0.376 (SE 0.748); seed median 0.274 [min -0.630, max 1.153]; eps=1 treated 22; Jaccard 1.000
- (e) EMDE-only sample: N rows 2250, countries 86; ATE 0.314 (SE 1.282); seed median 0.284 [min -2.644, max 10.072]; eps=1 treated 13; Jaccard 0.591
- (f) T_stock instead of T_res: N rows 3338, countries 126; ATE 1.791 (SE 1.625); seed median 1.863 [min -1.100, max 7.945]; eps=1 treated 35; Jaccard 0.239
- (g) agricultural VA growth outcome: N rows 3304, countries 125; ATE 0.467 (SE 0.550); seed median 0.786 [min -0.403, max 2.484]; eps=1 treated 26; Jaccard 0.000
- (h) uniform exposure: eps=1 treated 22; Jaccard 1.000; held-out eps=1 0.0604; equity gain eps=1 -0.0089 [-0.1301, 0.1114]
