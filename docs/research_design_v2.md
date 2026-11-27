# Research design v2: equity-weighted policy learning for AI-enabled climate resilience

Supersedes the regression-and-interaction design in `brainstorm.md`. The methodological core is
Layer 3: a new estimator for learning *where to build climate-AI capacity* when society cares
about who within each country gains resilience.

## 1. Question

If a donor, development bank or government can build climate-AI capacity (research groups,
early-warning and agro-advisory systems, skills) in a limited number of countries, **which
countries should it prioritise, and how does the answer change once the welfare of the poorest
deciles counts more than that of the richest?**

This answers the special issue's three questions in one object:
who benefits (decile-level resilience effects), who bears the cost (energy/CO2 module), and under
what conditions (the learned allocation rule is a set of interpretable conditions).

## 2. Setting and notation

Unit: country *i*, period *t* (country-year for GDP outcomes; survey spell for decile outcomes).

- `S_it`: **climate shock**, exogenous. Annual mean surface temperature (ERA5) minus a
  country-specific linear trend (fit on 1980–2023), in °C. Main variant: S = hot-side deviation;
  robustness: absolute deviation, precipitation added. For survey spells, S = mean of annual
  shocks over the spell years.
- `T_it ∈ {0,1}`: **climate-AI capability** — 1 if the country's climate-AI publications per
  million population (OpenAlex works tagged both "Artificial intelligence" or "Machine learning"
  AND "Climate change"/related, 3-year rolling sum, lagged) are above the cross-country median of
  that year. Confounded by development; handled by X and the propensity model.
- `X_it`: **covariates / effect modifiers**: log GDP pc PPP, Gini, internet %, agriculture VA
  share of GDP, urban %, baseline (1980–2000) mean temperature, governance proxy (ND-GAIN
  readiness lagged), general AI pubs per million (lagged), knowledge sovereignty (share of
  country's climate-AI papers with only domestic affiliations), renewable share, population,
  region dummies, EMDE dummy.
- `Y_idt`: **outcome** for decile *d* = 1..10 (PIP): annualised log growth of decile mean
  consumption/income over a comparable survey spell. Country-level outcome (Layer 1): annual GDP
  per capita growth and agricultural value-added growth.

## 3. Structural model

For each decile d (suppress i,t):

    Y_d = β_d(X, T) · S + g_d(X, T) + ε_d,      E[ε_d | S, X, T] = 0                  (1)

`β_d(X,T)` is the marginal damage of a 1 °C shock for decile d. The **resilience effect** of
climate-AI capability is

    ρ_d(X) = β_d(X, 1) − β_d(X, 0)                                                     (2)

(ρ > 0 means capability makes decile d's income less sensitive to hot shocks, when S is a hot
anomaly and β < 0).

Identification:
- A1 (shock exogeneity): S is as-good-as-random given (X, T) after de-trending; we still estimate
  E[S | X, T] and residualise (Robinson orthogonalisation).
- A2 (selection on observables for the *slope*): conditional on X, T is independent of the
  potential shock sensitivities {β_d(X,0), β_d(X,1)}. Weaker than unconfoundedness of T for the
  outcome level: T may correlate with growth levels g_d; we only need it not to correlate with
  unobserved sensitivity to shocks given X.
- A3 (overlap): 0 < e(X) = P(T=1|X) < 1; trim to e ∈ [0.05, 0.95].
- Sensitivity analysis for A2: Rosenbaum-style Γ on the propensity (odds ratio bound) or the
  omitted-variable bias bound of Chernozhukov et al. (2022) on the score.

## 4. Estimator: Equity-Weighted Doubly Robust Policy Learning (EW-DRPL)

### Step 1 — Cross-fitted nuisances (K = 5 folds, folds by country)
Using gradient boosting / random forests (any learner), fit on out-of-fold data:
- `ŝ(X,T) = E[S|X,T]`, `v̂(X,T) = E[(S − ŝ)² | X,T]` (floor at 1e-3)
- `ℓ̂_d(X,T) = E[Y_d | X,T]`
- `ê(X) = P(T=1|X)`

### Step 2 — Slope pseudo-outcome (Robinson)
    ψ_d = (Y_d − ℓ̂_d(X,T)) · (S − ŝ(X,T)) / v̂(X,T)
Under (1), E[ψ_d | X, T] = β_d(X, T).

### Step 3 — Orthogonalised slope score and doubly robust resilience score
The raw ψ_d is orthogonal to ℓ̂ and ŝ but NOT to v̂ (∂/∂v of E[ψ] = −β/v ≠ 0). Fix with a
preliminary slope fit `m̂_{d,t}(X) = E[ψ_d | X, T=t]` (cross-fitted, t = 0,1) and the
orthogonalised score

    ψ*_d = m̂_{d,T}(X) + [ (Y_d − ℓ̂_d)(S − ŝ) − m̂_{d,T}(X)(S − ŝ)² ] / v̂(X,T)

which has zero first-order sensitivity to (ℓ̂, ŝ, v̂, m̂) and E[ψ*_d | X,T] = β_d(X,T). Then

    Γ_d = m̂_{d,1}(X) − m̂_{d,0}(X) + T(ψ*_d − m̂_{d,1}(X))/ê(X) − (1−T)(ψ*_d − m̂_{d,0}(X))/(1 − ê(X))

E[Γ_d | X] = ρ_d(X) if either m̂ or ê is correct (double robustness in the T dimension), and
Γ_d is Neyman-orthogonal with respect to all nuisances (ŝ, v̂, ℓ̂, m̂, ê).
Implementation: nested cross-fitting — m̂ used in a unit's ψ* must be fit on folds that exclude
that unit's country, and on ψ values that were themselves built from nuisances excluding the
fitting fold (two-stage cross-fitting with the same country folds is acceptable).

### Step 4 — Equity weighting (the new part)
Benefit of capacity in country i, under expected shock exposure `s̄_i` (e.g. the country's
projected or recent mean hot anomaly; main: SD of the de-trended series 2000–2023):

    B_i(ε) = s̄_i · Σ_d p_d · ω_id(ε) · Γ_id

- `p_d = 0.1` (population share of each decile) × country population share if a
  population-weighted world welfare is wanted (main: population-weighted; robustness: unweighted).
- `ω_id(ε) = (y_id / ȳ)^(−ε)`: Atkinson marginal social welfare weight evaluated at decile mean
  income y_id (PPP $/day, latest survey) relative to the global population-weighted mean ȳ.
  ε = 0 → utilitarian (every dollar equal); ε = 1 → log utility; ε = 2 → strongly prioritarian.
  Weights normalised so that the population-weighted mean weight is 1.
- Because Γ is in growth units (log points), ω-weighting of growth effects is equivalent to a
  first-order approximation of the change in Atkinson-type social welfare.

### Step 5 — Policy learning
Policy class Π: depth-2 decision trees on a small, interpretable feature set Z ⊂ X
(log GDP pc, Gini, internet, agriculture share, baseline temperature, knowledge sovereignty).
With budget share k (capacity built in at most k·N countries; main k = 0.3):

    π̂_ε = argmax_{π ∈ Π}  (1/N) Σ_i π(Z_i) · B̂_i(ε)    s.t.  mean π ≤ k

Solved by exhaustive search over splits at feature deciles (exact for depth 2). Each leaf is
either fully treated or not; for a candidate tree, every subset of its (≤ 4) leaves is a candidate
policy, and subsets whose treated share exceeds k are infeasible. Unconstrained (k = 1) is also
reported (treat the leaves with positive mean B̂).

Country-level aggregation: Γ_id averaged over a country's spells before Step 4 (one B_i per
country) so the policy is a country-level rule; spells enter via the pseudo-outcomes.

### Step 6 — Evaluation and inference
- **Policy value** V(π) = mean of π(Z)·B̂ evaluated on *held-out* folds (cross-fitting the tree:
  learn on K−1 country folds, evaluate on the remaining fold).
- Compare: π̂_ε (EW-DRPL), utilitarian π̂_0 evaluated under ε weights, plug-in rule (rank by
  m̂_1 − m̂_0 without DR correction), "AI-to-the-capable" rule (top-k by existing AI capability),
  "poorest-first" rule (lowest GDP pc), random-k.
- **Equity price**: V_0(π̂_0) − V_0(π̂_ε) = utilitarian welfare given up to adopt the
  equity-weighted rule; and the gain V_ε(π̂_ε) − V_ε(π̂_0).
- Confidence intervals: country-cluster bootstrap (B = 500) of the held-out values.
- **Regret bound** statement (theory section): with DR scores and a VC class of depth-2 trees,
  regret is O_p(sqrt(VC(Π)/N)) (Athey & Wager 2021) and the weights ω are bounded, so the bound
  carries over with constant max ω(ε); state as a proposition with proof sketch.

## 5. Layers 1–2 (supporting evidence, also produced by the same scores)
- Layer 1: country-year GDP growth and agricultural VA growth; report ATE of Γ (average
  resilience effect) with SE, the best linear projection (BLP) of Γ on Z, and GATES (quintiles of
  out-of-fold predicted ρ̂) with CLAN (means of Z by quintile).
- Layer 2: decile-level average resilience effects Γ̄_d with SEs: the **resilience incidence
  curve** (d = 1..10). Pro-poor if Γ̄_1 > Γ̄_10.
- Cost module: DML partially linear model of ln electricity pc and ln CO2 pc on ln climate-AI and
  ln general AI capability with X as controls (country-clustered); heterogeneity by renewable share
  via BLP.
- Baseline: one classical TWFE table (shock × T interaction) for comparison only.

## 6. Monte Carlo (validates the estimator; required for a methods-centred paper)
DGP: N ∈ {150, 500} countries × 10 deciles × J spells; X ~ correlated normals mimicking
(log GDP, Gini, internet, agri share); T ~ Bernoulli(logit(1.5·loggdp + 0.5·internet));
S ~ N(0, σ(X)); β_d(X,T) = −(1 + 0.5·agri − 0.1·d) + T · ρ_d(X) with ρ_d(X) = 0.8·(1 − d/10)·1{internet > 0}
(capability helps the poor only where connected); g_d nonlinear in X and correlated with T
(confounding of levels). Decile income y_id from a lognormal with Gini-dependent spread.
Compare per ε ∈ {0,1,2}: (a) EW-DRPL, (b) utilitarian DR policy, (c) plug-in (no AIPW),
(d) IPW-only, (e) naive TWFE-interaction ranking. Metrics: regret relative to oracle policy under
the ε-welfare, share of oracle-treated countries recovered, bias/RMSE of ATE of ρ. R = 200 reps.

## 7. Data sources
ERA5 country temperature (OWID grapher `average-monthly-surface-temperature` or
`annual-temperature-anomalies`), World Bank PIP (survey-level deciles, comparable spells),
WDI (GDP, agri VA share NV.AGR.TOTL.ZS, agri VA growth NV.AGR.TOTL.KD.ZG, Gini, internet,
urban, population), OpenAlex (AI, ML, climate-change concept counts; domestic-only counts),
ND-GAIN readiness, OWID energy and CO2.

## 8. Key references to position against
Athey & Wager (2021, Econometrica) policy learning; Kitagawa & Tetenov (2018, Econometrica) EWM;
Kitagawa & Tetenov (2021, JBES?) equality-minded treatment choice; Chernozhukov et al. (2018) DML;
Wager & Athey (2018) causal forests; Chernozhukov, Demirer, Duflo & Fernández-Val generic ML
inference (GATES/CLAN); Atkinson (1970); Dell, Jones & Olken (2012); Kalkuhl & Wenz (2020);
Burke, Hsiang & Miguel (2015); Robinson (1988). Verify every citation via Crossref before use.
