# Brainstorm: IJSE Special Issue "Artificial Intelligence, Climate Change and Inclusive Socio-Economic Development"

Target: International Journal of Social Economics (Emerald), special issue.
Submission window: 15 Nov 2026 to 15 Mar 2027 (ScholarOne, select the SI title).
Guest editors: Dao Le Trang Anh (RMIT Vietnam), Christopher Gan (Lincoln), Binh Nguyen (RMIT Vietnam), Quang Nguyen (UD-DUE).

The call's organising question: **who benefits from AI-driven climate solutions, who bears their
costs, and under what institutional, financial and technological conditions AI can promote
inclusive climate-resilient development.** Emphasis on emerging and developing economies; SDGs
1, 2, 8, 9, 10, 11, 13. The four papers cited in the call (Cazzaniga et al. 2024; Dong et al. 2024;
IEA 2025; Larosa & Wickberg 2024) are the obvious anchors and should be cited and engaged.

## Candidate ideas

| # | Idea | SI theme | Data | Feasibility here | Novelty |
|---|------|----------|------|------------------|---------|
| A | **AI capability and climate resilience: is the dividend conditional on inequality and digital inclusion?** Cross-country panel linking national AI capability to ND-GAIN climate vulnerability/readiness, with inequality and internet access as moderators, plus the energy/emissions cost of AI capability | 1, 2, 3 | OpenAlex (AI publications by country-year), IMF AI Preparedness Index, ND-GAIN, WDI, OWID energy/CO2 | High: all public, all reachable via API | High: links the three strands the call says are fragmented; directly answers "who benefits / who bears costs" |
| B | AI exposure of employment in climate-sensitive sectors (agriculture, fisheries) in Vietnam, using LFS microdata and ILO GenAI exposure scores | 1, 2 | Vietnam LFS (restricted), ILO exposure index | Low: microdata not available here | High |
| C | Data-centre growth and local electricity prices / emissions in ASEAN | 3 | Data-centre registries (commercial), grid data | Low: no open panel of data-centre capacity | Medium |
| D | AI-enabled credit scoring and green credit to SMEs (bank-level) | 5 | Bank filings, ESG ratings | Medium: hand-collection needed | Medium |
| E | Revisiting Dong et al. (2024): does inequality undermine AI's carbon abatement, globally rather than for China | 2, 3, 4 | OpenAlex AI, OWID CO2, WDI Gini | High | Medium (extension) |
| F | Text-mining NDCs / National Adaptation Plans for AI mentions and relating to adaptation finance | 1, 5 | UNFCCC NDC registry, OECD CRS | Medium | Medium |

## Decision: Idea A, absorbing E as the "cost" side

Idea A answers the call's central question in one coherent design, uses only open data (fully
reproducible, which reviewers in social economics increasingly ask for), covers 150+ countries
with emerging and developing economies as the core subsample, and E slots in naturally as the
environmental-cost leg. B, C, D need data we cannot obtain now; they are noted as follow-ups.

### Working title
*Who gains climate resilience from artificial intelligence? Inequality, digital inclusion and the
AI–adaptation nexus across countries*

### Research questions
- **RQ1 (benefit).** Is national AI capability associated with lower climate vulnerability and
  higher adaptation readiness (ND-GAIN), net of income and country/year fixed effects?
- **RQ2 (inclusion conditions).** Does that association weaken where income inequality is high
  and strengthen where digital access (internet use) is broad? Is there an inequality threshold
  beyond which the resilience dividend disappears?
- **RQ3 (cost).** Is AI capability associated with higher per-capita electricity demand and CO2
  emissions, and does a higher renewable share in electricity offset this?
- **RQ4 (heterogeneity).** Do these patterns differ for emerging and developing economies (EMDEs)
  versus advanced economies?

### Hypotheses
- H1: AI capability → lower vulnerability, higher readiness.
- H2a: The AI × Gini interaction is adverse (weaker dividend under high inequality).
- H2b: The AI × internet-access interaction is favourable.
- H3a: AI capability → higher electricity use per capita; H3b: weaker CO2 link where renewables share is high.

### Measurement
- **AI capability (main, panel):** AI publications (OpenAlex concept "Artificial intelligence")
  per million population, logged; alternative: AI share of all national publications (AI
  specialisation). Fractional issues (multi-country papers) noted as a limitation.
- **AI readiness (validation, cross-section 2023):** IMF AI Preparedness Index.
- **Outcomes:** ND-GAIN vulnerability (0–1, lower is better), readiness (0–1), overall GAIN score;
  vulnerability sub-sectors (food, water, health, ecosystems, habitat, infrastructure) for
  livelihood-relevant channels. Electricity demand per capita and CO2 per capita (OWID).
- **Moderators:** Gini (WDI, interpolated within country), internet users (% population),
  renewables share of electricity (OWID).
- **Controls:** log GDP per capita PPP, urbanisation, trade openness, population (log),
  tertiary enrolment where available (robustness).

### Identification strategy
1. Two-way fixed effects (country, year), regressors lagged one year, Driscoll–Kraay or
   country-clustered SEs.
2. Interaction models (AI × Gini, AI × Internet), with marginal-effect plots.
3. Hansen (1999) panel threshold regression with Gini as threshold variable (bootstrap test).
4. Shift-share instrument: country's initial (2000–2004) AI share of publications × global AI
   publication growth (leave-one-out). 2SLS with FE.
5. Heterogeneity: EMDE vs advanced; income groups; region.
6. Robustness: alternative AI measure, 3-year averages, dropping China/US, long differences
   (2005→2022), cross-section with IMF AIPI.

### Contribution (as framed for IJSE)
- First global panel evidence connecting AI capability with climate-adaptation outcomes while
  explicitly modelling the distributional conditions (inequality, digital divide).
- Puts benefit (resilience) and cost (energy/emissions) in one framework — the trade-off the SI
  says is under-studied.
- Policy relevance for EMDEs (Vietnam and ASEAN as illustrative cases).

### Follow-up papers (not pursued now)
B (Vietnam LFS × AI exposure in climate-sensitive jobs), D (AI credit scoring and green credit),
F (AI in NDCs and adaptation finance).
