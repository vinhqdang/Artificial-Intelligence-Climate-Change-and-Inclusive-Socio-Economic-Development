"""
Step 3: econometric analysis, tables, figures and results summary.

Inputs : data/processed/panel.csv, data/raw/imf_aipi.csv (optional, for T9)
Outputs: results/tables/*.csv and *.tex, results/figures/*.pdf and *.png,
         results/results_summary.md

Design (see docs/brainstorm.md)
  * Panel two-way FE (country and year), linearmodels PanelOLS, SEs clustered by country.
  * All regressors lagged one year within country; outcomes contemporaneous.
  * Controls X = ln GDP per capita, urbanisation, trade openness, ln population (lagged).
  * Moderators are centred on the estimation-sample mean before forming interactions.
  * Hansen (1999) threshold regression implemented manually on FE-residualised data.
  * IV: shift-share instrument, two-way demeaned, 2SLS with country-clustered SEs.

Usage: python code/03_analysis.py   (seed 42)
"""
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
TAB = RESULTS / "tables"
FIG = RESULTS / "figures"
SEED = 42
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

X = ["ln_gdppc_l", "urban_l", "trade_l", "ln_pop_l"]          # controls (lagged)
X_LABEL = {"ln_gdppc_l": "ln GDP per capita (t-1)", "urban_l": "Urban population share (t-1)",
           "trade_l": "Trade openness (t-1)", "ln_pop_l": "ln population (t-1)",
           "tertiary_l": "Tertiary enrolment (t-1)"}
LAG_VARS = ["ln_ai", "ai_share", "ln_ml", "gini", "internet", "ren_share", "tertiary",
            "ln_gdppc", "urban", "trade", "ln_pop", "iv_ss", "iv_ss_lvl"]
CENTER = {"gini_cl": "gini_l", "internet_cl": "internet_l", "ren_cl": "ren_share_l"}
INTER = {"ai_x_gini": ("ln_ai_l", "gini_cl"), "ai_x_internet": ("ln_ai_l", "internet_cl"),
         "ai_x_ren": ("ln_ai_l", "ren_cl")}
LABEL = {
    "ln_ai_l": "AI capability, ln(1+AI pubs per m) (t-1)",
    "AI": "AI measure (t-1)",
    "gini_cl": "Gini, centred (t-1)", "ai_x_gini": "AI $\\times$ Gini (centred)",
    "internet_cl": "Internet users %, centred (t-1)", "ai_x_internet": "AI $\\times$ Internet (centred)",
    "ren_cl": "Renewables share %, centred (t-1)", "ai_x_ren": "AI $\\times$ Renewables (centred)",
    "ai_lo": "AI, gini $\\leq$ threshold", "ai_hi": "AI, gini $>$ threshold",
    "dai": "$\\Delta$ AI capability (2005-2022)",
    "const": "Constant", "aipi": "IMF AI Preparedness Index (2023)",
    "gini_c": "Gini, centred (latest)", "aipi_x_gini": "AIPI $\\times$ Gini (centred)",
    "ln_gdppc": "ln GDP per capita (2023)", "gini": "Gini (latest observed)",
    "tertiary_l": "Tertiary enrolment (t-1)",
}
OUTCOME_LABEL = {"vuln": "ND-GAIN vulnerability", "ready": "ND-GAIN readiness",
                 "gain": "ND-GAIN GAIN score", "vuln_food": "Food vulnerability",
                 "vuln_water": "Water vulnerability", "vuln_health": "Health vulnerability",
                 "vuln_eco": "Ecosystems vulnerability", "vuln_habitat": "Habitat vulnerability",
                 "vuln_infra": "Infrastructure vulnerability", "ln_elec": "ln electricity per capita",
                 "ln_co2pc": "ln CO$_2$ per capita"}
MAIN_OUT = ["vuln", "ready"]
SUMMARY_SECTIONS = []        # markdown blocks for results_summary.md
KEY = {}                     # scalar key results, reported in the summary


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_panel():
    P = pd.read_csv(PROC / "panel.csv").sort_values(["iso3", "year"]).set_index(["iso3", "year"])
    g = P.groupby(level=0)
    for v in LAG_VARS:
        P[v + "_l"] = g[v].shift(1)          # balanced grid: shift by one row = one year
    return P


def add_derived(S):
    """Centre moderators on the sample mean of S and form interaction terms."""
    S = S.copy()
    for c, raw in CENTER.items():
        if raw in S.columns:
            S[c] = S[raw] - S[raw].mean()
    for name, (a, b) in INTER.items():
        if a in S.columns and b in S.columns:
            S[name] = S[a] * S[b]
    return S


class Fit:
    """Coefficient container shared by linearmodels and statsmodels results."""

    def __init__(self, b, se, p, N, G, r2=np.nan, cov=None):
        self.b, self.se, self.p = pd.Series(b), pd.Series(se), pd.Series(p)
        self.N, self.G, self.r2 = int(N), G, r2
        self.cov = cov

    @classmethod
    def from_lm(cls, res, G, r2_attr="rsquared_within"):
        return cls(res.params, res.std_errors, res.pvalues, res.nobs, G,
                   getattr(res, r2_attr, np.nan), res.cov)

    @classmethod
    def from_sm(cls, res, G):
        return cls(res.params, res.bse, res.pvalues, res.nobs, G, res.rsquared, res.cov_params())

    def rename(self, mapping):
        f = Fit(self.b.rename(mapping), self.se.rename(mapping), self.p.rename(mapping),
                self.N, self.G, self.r2)
        if self.cov is not None:
            f.cov = self.cov.rename(index=mapping, columns=mapping)
        return f


def fe_ols(S, y, regs, cov="cluster"):
    """Two-way FE panel OLS. cov='cluster' (country-clustered) or 'dk' (Driscoll-Kraay)."""
    D = S.dropna(subset=[y] + regs)
    res = PanelOLS(D[y], D[regs], entity_effects=True, time_effects=True).fit(
        **({"cov_type": "kernel", "kernel": "bartlett"} if cov == "dk"
           else {"cov_type": "clustered", "cluster_entity": True}))
    return Fit.from_lm(res, D.index.get_level_values(0).nunique())


def model_sample(P, y, base_cols):
    """Estimation sample for outcome y: complete cases on y and base_cols, with derived terms."""
    return add_derived(P[[y] + base_cols].dropna())


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def cell(fit, var):
    if fit is None or var not in fit.b.index:
        return "", ""
    return f"{fit.b[var]:.3f}{stars(fit.p[var])}", f"({fit.se[var]:.3f})"


# ---------------------------------------------------------------------------
# Table writers (CSV, LaTeX fragment, markdown for the summary)
# ---------------------------------------------------------------------------
def tex_text(s):
    s = str(s)
    s = s.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
    if "$" not in s:
        s = re.sub(r"(?<!\\)_", r"\\_", s)
    s = re.sub(r"p<(0\.\d+)", r"p$<$\1", s)
    if "$^{" not in s:
        s = re.sub(r"\*{1,3}", lambda m: f"$^{{{m.group(0)}}}$", s)
    return s


def write_table(stem, caption, heads, body, notes, group=None):
    """body: list of rows (first element = row label); ('__panel__', text) for panel headers;
    None for a rule. Writes CSV, LaTeX fragment and returns markdown."""
    TAB.mkdir(parents=True, exist_ok=True)
    ncol = len(heads)
    csv_rows = [[""] + heads]
    tex = ["% Requires \\usepackage{booktabs}", "\\begin{table}[htbp]", "\\centering",
           f"\\caption{{{tex_text(caption)}}}", f"\\label{{tab:{stem}}}",
           "\\begin{tabular}{l" + "c" * ncol + "}", "\\toprule"]
    md = ["| | " + " | ".join(heads) + " |", "|---|" + "---|" * ncol]
    if group:
        gl, grp_cells = [], []
        for label, span in group:
            gl.append(f"\\multicolumn{{{span}}}{{c}}{{{tex_text(label)}}}")
            grp_cells += [label] + [""] * (span - 1)
        tex.append(" & " + " & ".join(gl) + " \\\\")
        csv_rows.insert(0, [""] + grp_cells)
    tex.append(" & " + " & ".join(f"{tex_text(h)}" for h in heads) + " \\\\")
    tex.append("\\midrule")
    for row in body:
        if row is None:
            tex.append("\\midrule")
            md.append("| | " + " | ".join([""] * ncol) + " |")
            continue
        if row[0] == "__panel__":
            tex.append(f"\\multicolumn{{{ncol + 1}}}{{l}}{{\\textit{{{tex_text(row[1])}}}}} \\\\")
            csv_rows.append([row[1]] + [""] * ncol)
            md.append(f"| *{row[1]}* |" + " |" * ncol)
            continue
        csv_rows.append(list(row))
        tex.append(" & ".join(tex_text(c) for c in row) + " \\\\")
        md.append("| " + " | ".join(str(c) for c in row) + " |")
    tex += ["\\bottomrule", "\\end{tabular}", "\\par\\smallskip",
            f"\\footnotesize \\textit{{Notes:}} {tex_text(notes)}", "\\end{table}", ""]
    (TAB / f"{stem}.tex").write_text("\n".join(tex))
    pd.DataFrame(csv_rows).to_csv(TAB / f"{stem}.csv", index=False, header=False)
    md_block = f"### {caption}\n\n" + "\n".join(md) + f"\n\n*Notes:* {notes}\n"
    SUMMARY_SECTIONS.append(md_block)
    print(f"[table] {stem}")


def coef_rows(cols, terms):
    """Two rows per term (coefficient with stars, standard error in parentheses)."""
    rows = []
    for var, label in terms:
        if not any(var in c.b.index for c in cols if c is not None):
            continue
        rows.append([label] + [cell(c, var)[0] for c in cols])
        rows.append([""] + [cell(c, var)[1] for c in cols])
    return rows


def stat_rows(cols, controls=None, fe=True):
    rows = [None,
            ["Country FE"] + ["Yes" if fe else "No"] * len(cols),
            ["Year FE"] + ["Yes" if fe else "No"] * len(cols)]
    if controls is not None:
        rows.append(["Controls (t-1)"] + [controls] * len(cols))
    rows += [["Observations"] + [f"{c.N:,}" for c in cols],
             ["Countries"] + [f"{c.G}" for c in cols],
             ["Within R2"] + [f"{c.r2:.3f}" if np.isfinite(c.r2) else "" for c in cols]]
    return rows


def sample_size_line(P):
    return f"Full panel: {P.reset_index()['iso3'].nunique()} countries, {len(P)} country-years."


# ---------------------------------------------------------------------------
# T1 Descriptives
# ---------------------------------------------------------------------------
DESC_VARS = ["vuln", "ready", "gain", "vuln_food", "vuln_water", "vuln_health", "vuln_eco",
             "vuln_habitat", "vuln_infra", "ai_pc", "ln_ai", "ai_share", "ln_ml", "gini", "internet",
             "ln_gdppc", "urban", "trade", "ln_pop", "tertiary", "ren_share", "ln_elec", "ln_co2pc"]
DESC_LABEL = {"ai_pc": "AI publications per million", "ln_ai": "AI capability, ln(1+AI pc)",
              "ai_share": "AI share of publications (%)", "ln_ml": "ML capability, ln(1+ML pc)",
              "gini": "Gini index", "internet": "Internet users (%)", "ln_gdppc": "ln GDP per capita",
              "urban": "Urban population (%)", "trade": "Trade openness (% GDP)", "ln_pop": "ln population",
              "tertiary": "Tertiary enrolment (%)", "ren_share": "Renewables share of electricity (%)",
              "ln_elec": "ln electricity per capita", "ln_co2pc": "ln CO$_2$ per capita", **OUTCOME_LABEL}


def table_T1(P):
    E = P[P["emde"] == 1]
    rows = []
    for v in DESC_VARS:
        r = [DESC_LABEL.get(v, v)]
        for D in (P, E):
            s = D[v].dropna()
            r += [f"{s.mean():.2f}", f"{s.std():.2f}", f"{s.min():.2f}", f"{s.max():.2f}", f"{len(s):,}"]
        rows.append(r)
    KEY["T1_N_full"] = int(P.reset_index()["iso3"].nunique())
    KEY["T1_N_emde"] = int(E.reset_index()["iso3"].nunique())
    heads = ["Mean", "SD", "Min", "Max", "N"] * 2
    group = [("Full sample", 5), ("EMDE", 5)]
    write_table("T1_descriptives", "Descriptive statistics, 2000-2023 country-year panel",
                heads, rows,
                f"Levels, contemporaneous (not lagged). Full sample: {KEY['T1_N_full']} countries; "
                f"EMDE (non-high-income): {KEY['T1_N_emde']} countries. N counts non-missing country-years. "
                "Sources: OpenAlex, ND-GAIN, World Bank WDI, OWID (see codebook).", group=group)


# ---------------------------------------------------------------------------
# T2 Baseline
# ---------------------------------------------------------------------------
def table_T2(P):
    E = P[P["emde"] == 1]
    cols = []
    for D in (P, E):
        for y in ["vuln", "ready", "gain"]:
            S = model_sample(D, y, ["ln_ai_l"] + X)
            cols.append(fe_ols(S, y, ["ln_ai_l"] + X))
    KEY["T2"] = cols
    terms = [("ln_ai_l", LABEL["ln_ai_l"])] + [(x, X_LABEL[x]) for x in X]
    body = coef_rows(cols, terms) + stat_rows(cols, controls="Yes")
    heads = [f"({i})" for i in range(1, 7)]
    write_table("T2_baseline", "Baseline: AI capability and climate resilience",
                heads, body,
                "Dependent variables: ND-GAIN vulnerability (higher = more vulnerable), readiness and GAIN score "
                "(higher = better). Columns (1)-(3): full sample; (4)-(6): emerging and developing economies (EMDE). "
                "Two-way FE; standard errors clustered by country in parentheses. Regressors lagged one year. "
                "$^{*}$ p<0.1, $^{**}$ p<0.05, $^{***}$ p<0.01.",
                group=[("Full sample", 3), ("EMDE", 3)])
    return cols


# ---------------------------------------------------------------------------
# T3 Moderation: inequality and internet access
# ---------------------------------------------------------------------------
def table_T3(P):
    cols, meta = [], {}
    base = ["ln_ai_l", "gini_l", "internet_l"] + X
    for y in MAIN_OUT:
        S = model_sample(P, y, base)
        models = [
            ["ln_ai_l", "gini_cl", "ai_x_gini"] + X,
            ["ln_ai_l", "internet_cl", "ai_x_internet"] + X,
            ["ln_ai_l", "gini_cl", "ai_x_gini", "internet_cl", "ai_x_internet"] + X,
        ]
        for regs in models:
            cols.append(fe_ols(S, y, regs))
        meta[y] = {"S": S, "joint": cols[-1]}
    KEY["T3"] = cols
    KEY["T3_meta"] = meta
    terms = [("ln_ai_l", LABEL["ln_ai_l"]), ("gini_cl", LABEL["gini_cl"]), ("ai_x_gini", LABEL["ai_x_gini"]),
             ("internet_cl", LABEL["internet_cl"]), ("ai_x_internet", LABEL["ai_x_internet"])] + \
            [(x, X_LABEL[x]) for x in X]
    body = coef_rows(cols, terms) + stat_rows(cols, controls="Yes")
    write_table("T3_moderation", "Moderation of the AI-resilience association by inequality and internet access",
                [f"({i})" for i in range(1, 7)], body,
                "Columns (1)-(3): ND-GAIN vulnerability; (4)-(6): readiness. Gini and internet use are "
                "centred on the estimation-sample mean (lagged one year). Columns (1)/(4) and (2)/(5) add one moderator "
                "each; (3)/(6) include both. Estimation samples are identical within each outcome. "
                "Two-way FE; country-clustered SEs. $^{*}$ p<0.1, $^{**}$ p<0.05, $^{***}$ p<0.01.",
                group=[("Vulnerability", 3), ("Readiness", 3)])
    return cols, meta


def marginal_effects(meta):
    """Marginal effect of AI capability over the gini (and internet) distribution, from the joint model."""
    S = meta["vuln"]["S"]
    fit = meta["vuln"]["joint"]
    out = {}
    for mod_raw, cen, inter, key in [("gini_l", "gini_cl", "ai_x_gini", "gini"),
                                     ("internet_l", "internet_cl", "ai_x_internet", "internet")]:
        grid_raw = np.linspace(S[mod_raw].quantile(0.10), S[mod_raw].quantile(0.90), 60)
        mu = S[mod_raw].mean()
        c = grid_raw - mu
        b_main, b_int = fit.b["ln_ai_l"], fit.b[inter]
        cov = fit.cov.loc[["ln_ai_l", inter], ["ln_ai_l", inter]].to_numpy()
        A = np.column_stack([np.ones_like(c), c])
        me = b_main + b_int * c
        se = np.sqrt(np.einsum("ij,jk,ik->i", A, cov, A))
        out[key] = {"grid": grid_raw, "me": me, "lo": me - 1.96 * se, "hi": me + 1.96 * se,
                    "p10": S[mod_raw].quantile(0.10), "p50": S[mod_raw].quantile(0.50),
                    "p90": S[mod_raw].quantile(0.90), "mu": mu}
        for lab, q in [("p10", "p10"), ("p50", "p50"), ("p90", "p90")]:
            v = out[key][q]
            cc = np.array([1.0, v - mu])
            m = b_main + b_int * (v - mu)
            s = np.sqrt(cc @ cov @ cc)
            out[key][lab + "_me"] = (m, m - 1.96 * s, m + 1.96 * s)
    KEY["ME"] = out
    return out


def fig_marginal(out):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    specs = [("gini", "Gini index (t-1)", "(a) Inequality"),
             ("internet", "Internet users, % of population (t-1)", "(b) Digital access")]
    for ax, (key, xlab, title) in zip(axes, specs):
        d = out[key]
        ax.fill_between(d["grid"], d["lo"], d["hi"], color="0.8", label="95% CI")
        ax.plot(d["grid"], d["me"], color="black", lw=1.8, label="Marginal effect")
        ax.axhline(0, color="black", lw=0.8, ls=":")
        for q, ls in [("p10", "--"), ("p90", "--")]:
            ax.axvline(d[q], color="0.4", lw=0.8, ls=ls)
        ax.set_xlabel(xlab)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Effect of AI capability\non vulnerability (index points)")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.text(0.5, -0.02, "Dashed lines: 10th and 90th percentiles of the estimation sample.",
             ha="center", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "fig_marginal_effects")


def save_fig(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {stem}")


# ---------------------------------------------------------------------------
# T4 Hansen (1999) panel threshold
# ---------------------------------------------------------------------------
def fe_basis(ctry, yrs):
    """Orthonormal basis of the country and year dummy space (for exact FE partialling out)."""
    D = np.hstack([pd.get_dummies(ctry).values.astype(float),
                   pd.get_dummies(yrs).values.astype(float)])
    U, sv, _ = np.linalg.svd(D, full_matrices=False)
    return U[:, sv > 1e-10 * sv[0]]


def hansen_threshold(P, y, B=300, n_grid=100, lr_crit=7.35):
    """Hansen (1999) threshold model with threshold variable gini_l, slope on ln_ai_l regime-specific,
    controls common. Two-way FE partialled out exactly by projection on the dummy space.
    Bootstrap: wild cluster bootstrap (Rademacher, by country) under the linear null."""
    S = P[[y, "ln_ai_l", "gini_l"] + X].dropna()
    n = len(S)
    ctry = S.index.get_level_values(0)
    Q = fe_basis(ctry, S.index.get_level_values(1))
    resid = lambda Z: Z - Q @ (Q.T @ Z)
    yv = resid(S[y].to_numpy(float))
    Xc = resid(S[X].to_numpy(float))
    xv = S["ln_ai_l"].to_numpy(float)
    qv = S["gini_l"].to_numpy(float)

    grid = np.unique(np.quantile(qv, np.linspace(0.15, 0.85, n_grid)))
    M0 = np.column_stack([resid(xv), Xc])
    b0 = np.linalg.lstsq(M0, yv, rcond=None)[0]
    SSR0 = float(np.sum((yv - M0 @ b0) ** 2))
    Ms, SSRg = [], []
    for g in grid:
        d = (qv <= g).astype(float)
        M = np.column_stack([resid(xv * d), resid(xv * (1 - d)), Xc])
        b = np.linalg.lstsq(M, yv, rcond=None)[0]
        Ms.append(M)
        SSRg.append(float(np.sum((yv - M @ b) ** 2)))
    SSRg = np.array(SSRg)
    k = int(np.argmin(SSRg))
    gam, SSR1 = grid[k], SSRg[k]
    sig2 = SSR1 / n
    LR = (SSRg - SSR1) / sig2
    F = (SSR0 - SSR1) / sig2
    ci_grid = grid[LR <= lr_crit]
    ci = (float(ci_grid.min()), float(ci_grid.max()))

    # Wild cluster bootstrap under the null of no threshold
    f0 = M0 @ b0
    e0 = yv - f0
    codes = pd.factorize(ctry)[0]
    W = rng.choice([-1.0, 1.0], size=(codes.max() + 1, B))
    Ys = f0[:, None] + e0[:, None] * W[codes]
    SSR0b = np.sum((Ys - M0 @ (np.linalg.pinv(M0) @ Ys)) ** 2, axis=0)
    SSR1b = np.full(B, np.inf)
    for M in Ms:
        ssr = np.sum((Ys - M @ (np.linalg.pinv(M) @ Ys)) ** 2, axis=0)
        SSR1b = np.minimum(SSR1b, ssr)
    Fb = (SSR0b - SSR1b) / (SSR1b / n)
    pval = float(np.mean(Fb >= F))

    # Regime-specific slopes with country-clustered SEs (two-way FE, full estimation sample)
    S2 = S.copy()
    S2["ai_lo"] = xv * (qv <= gam)
    S2["ai_hi"] = xv * (qv > gam)
    slopes = fe_ols(S2, y, ["ai_lo", "ai_hi"] + X)
    lin = fe_ols(S, y, ["ln_ai_l"] + X)
    return {"y": y, "grid": grid, "LR": LR, "gamma": float(gam), "ci": ci, "F": float(F),
            "p": pval, "slopes": slopes, "linear": lin, "N_lo": int((qv <= gam).sum()),
            "N_hi": int((qv > gam).sum()), "N": n, "G": int(codes.max() + 1), "SSR0": SSR0,
            "SSR1": SSR1, "sig2": sig2, "B": B, "lr_crit": lr_crit}


def table_T4(P):
    res = {y: hansen_threshold(P, y) for y in MAIN_OUT}
    KEY["T4"] = res
    rows = []

    def add(label, fn):
        rows.append([label] + [fn(res[y]) for y in MAIN_OUT])

    add("Threshold $\\hat\\gamma$ (gini, t-1)", lambda r: f"{r['gamma']:.2f}")
    add("95% CI for $\\hat\\gamma$ (LR$\\leq$7.35), lower", lambda r: f"{r['ci'][0]:.2f}")
    add("95% CI for $\\hat\\gamma$, upper", lambda r: f"{r['ci'][1]:.2f}")
    add("Threshold F statistic", lambda r: f"{r['F']:.2f}")
    add("Bootstrap p-value (300 reps)", lambda r: f"{r['p']:.3f}")
    rows.append(None)
    for lab, sl in [("AI slope, gini $\\leq$ $\\hat\\gamma$ (below)", "ai_lo"),
                    ("AI slope, gini $>$ $\\hat\\gamma$ (above)", "ai_hi")]:
        rows.append([lab] + [f"{r['slopes'].b[sl]:.3f}{stars(r['slopes'].p[sl])}" for r in res.values()])
        rows.append([""] + [f"({r['slopes'].se[sl]:.3f})" for r in res.values()])
    rows.append(["Linear AI slope (reference)"] +
                [f"{r['linear'].b['ln_ai_l']:.3f}{stars(r['linear'].p['ln_ai_l'])}" for r in res.values()])
    rows.append([""] + [f"({r['linear'].se['ln_ai_l']:.3f})" for r in res.values()])
    rows.append(None)
    add("Observations below threshold", lambda r: f"{r['N_lo']:,}")
    add("Observations above threshold", lambda r: f"{r['N_hi']:,}")
    add("Observations (total)", lambda r: f"{r['N']:,}")
    add("Countries", lambda r: f"{r['G']}")
    write_table("T4_threshold", "Panel threshold regression (Hansen, 1999): inequality as threshold",
                [OUTCOME_LABEL[y] for y in MAIN_OUT], rows,
                "Threshold variable: Gini index (t-1). Regime-specific slopes on lagged ln AI capability; controls "
                "common across regimes; two-way FE. Grid of 100 candidate thresholds between the 15th and 85th "
                "percentiles. F and bootstrap p-value from a wild cluster bootstrap (Rademacher weights by country, "
                "300 replications, seed 42). Slope SEs clustered by country. The 95% CI is the set of grid values with "
                "LR$(\\gamma)\\leq$7.35, so its endpoints are grid values. $^{*}$ p<0.1, $^{**}$ p<0.05, $^{***}$ p<0.01.")
    return res


def fig_threshold(res):
    r = res["vuln"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(r["grid"], r["LR"], color="black", lw=1.6)
    ax.axhline(r["lr_crit"], color="0.4", ls="--", lw=1)
    ax.text(r["grid"][0], r["lr_crit"] + 0.4, "5% critical value (7.35)", fontsize=8, color="0.3")
    ax.axvline(r["gamma"], color="black", ls=":", lw=1)
    ax.text(r["gamma"], ax.get_ylim()[1] * 0.92, f" $\\hat\\gamma$={r['gamma']:.1f}", fontsize=8)
    ax.set_xlabel("Candidate threshold: Gini index (t-1)")
    ax.set_ylabel("LR$(\\gamma)$")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "fig_threshold")


# ---------------------------------------------------------------------------
# T5 Sector vulnerabilities
# ---------------------------------------------------------------------------
SECTORS = ["vuln_food", "vuln_water", "vuln_health", "vuln_eco", "vuln_habitat", "vuln_infra"]


def table_T5(P):
    colsA, colsB = [], []
    for y in SECTORS:
        colsA.append(fe_ols(model_sample(P, y, ["ln_ai_l"] + X), y, ["ln_ai_l"] + X))
    for y in SECTORS:
        colsB.append(fe_ols(model_sample(P, y, ["ln_ai_l", "gini_l"] + X), y,
                            ["ln_ai_l", "gini_cl", "ai_x_gini"] + X))
    KEY["T5"] = {"A": colsA, "B": colsB}
    body = [("__panel__", "Panel A: AI capability and controls")]
    body += coef_rows(colsA, [("ln_ai_l", LABEL["ln_ai_l"])] + [(x, X_LABEL[x]) for x in X])
    body += stat_rows(colsA, controls="Yes")
    body += [None, ("__panel__", "Panel B: with AI $\\times$ Gini interaction")]
    body += coef_rows(colsB, [("ln_ai_l", LABEL["ln_ai_l"]), ("gini_cl", LABEL["gini_cl"]),
                              ("ai_x_gini", LABEL["ai_x_gini"])] + [(x, X_LABEL[x]) for x in X])
    body += stat_rows(colsB, controls="Yes")
    write_table("T5_sectors", "Sector-specific vulnerability and the role of inequality",
                [f"({i})" for i in range(1, 7)], body,
                "Dependent variables: ND-GAIN sector vulnerability indices (food, water, health, ecosystems, habitat, "
                "infrastructure; index points, higher = more vulnerable). Two-way FE; country-clustered SEs. "
                "Columns follow the sector order food, water, health, ecosystems, habitat, infrastructure. "
                "$^{*}$ p<0.1, $^{**}$ p<0.05, $^{***}$ p<0.01.")
    return colsA, colsB


# ---------------------------------------------------------------------------
# T6 Environmental cost
# ---------------------------------------------------------------------------
def table_T6(P):
    cols = []
    base = ["ln_ai_l", "ren_share_l"] + X
    for y in ["ln_elec", "ln_co2pc"]:
        S = model_sample(P, y, base)
        cols.append(fe_ols(S, y, ["ln_ai_l"] + X))
        cols.append(fe_ols(S, y, ["ln_ai_l", "ren_cl", "ai_x_ren"] + X))
    KEY["T6"] = cols   # order: electricity, electricity + renewables, CO2, CO2 + renewables
    terms = [("ln_ai_l", LABEL["ln_ai_l"]), ("ren_cl", LABEL["ren_cl"]), ("ai_x_ren", LABEL["ai_x_ren"])] + \
            [(x, X_LABEL[x]) for x in X]
    body = coef_rows(cols, terms) + stat_rows(cols, controls="Yes")
    write_table("T6_environmental_cost", "Energy and emissions costs of AI capability",
                [f"({i})" for i in range(1, 5)], body,
                "Dependent variables: ln electricity demand per capita (columns 1-2) and ln CO$_2$ per capita "
                "(columns 3-4), OWID. Even-numbered columns add renewables share (centred, t-1) and its interaction with AI capability. "
                "All columns share one estimation sample (renewables share available). Two-way FE; country-clustered SEs. "
                "$^{*}$ p<0.1, $^{**}$ p<0.05, $^{***}$ p<0.01.",
                group=[("Electricity", 2), ("CO$_2$", 2)])
    return cols


# ---------------------------------------------------------------------------
# T7 IV (shift-share)
# ---------------------------------------------------------------------------
def iv_fit(P, y, instr):
    S = P[[y, "ln_ai_l", instr] + X].dropna()
    ctry = S.index.get_level_values(0)
    Q = fe_basis(ctry, S.index.get_level_values(1))
    dm = lambda Z: Z - Q @ (Q.T @ Z)
    yv = dm(S[y].to_numpy(float)).reshape(-1, 1)
    en = dm(S["ln_ai_l"].to_numpy(float)).reshape(-1, 1)
    zr = dm(S[instr].to_numpy(float)).reshape(-1, 1)
    Xr = dm(S[X].to_numpy(float))
    codes = pd.factorize(ctry)[0]
    G = int(codes.max() + 1)
    res = IV2SLS(pd.DataFrame(yv, columns=[y]), pd.DataFrame(Xr, columns=X),
                 pd.DataFrame(en, columns=["ln_ai_l"]), pd.DataFrame(zr, columns=[instr])).fit(
        cov_type="clustered", clusters=codes)
    iv = Fit.from_lm(res, G, r2_attr="rsquared")
    # First stage with country-clustered SEs
    fs = sm.OLS(en.ravel(), np.column_stack([zr.ravel(), Xr])).fit(
        cov_type="cluster", cov_kwds={"groups": codes})
    fs_b, fs_se, fs_p = fs.params[0], fs.bse[0], fs.pvalues[0]
    F = float(fs.tvalues[0] ** 2)
    ols = fe_ols(S, y, ["ln_ai_l"] + X)
    return {"iv": iv, "ols": ols, "fs_coef": float(fs_b), "fs_se": float(fs_se), "fs_p": float(fs_p),
            "F": F, "N": len(S), "G": G}


def table_T7(P):
    E = P[P["emde"] == 1]
    combos = []
    for instr in ["iv_ss_l", "iv_ss_lvl_l"]:
        for label, D in [("Full", P), ("EMDE", E)]:
            for y in MAIN_OUT:
                combos.append({"instr": instr, "sample": label, "y": y,
                               "r": iv_fit(D, y, instr)})
    KEY["T7"] = combos
    heads = [f"({i})" for i in range(1, 9)]
    outcome_row = ["Outcome"] + [OUTCOME_LABEL[c["y"]].replace("ND-GAIN ", "") for c in combos]
    sample_row = ["Sample"] + [c["sample"] for c in combos]
    body = [outcome_row, sample_row, None,
            ["AI capability, IV (2SLS)"] + [f"{c['r']['iv'].b['ln_ai_l']:.3f}{stars(c['r']['iv'].p['ln_ai_l'])}" for c in combos],
            [""] + [f"({c['r']['iv'].se['ln_ai_l']:.3f})" for c in combos],
            ["AI capability, OLS (same sample)"] + [f"{c['r']['ols'].b['ln_ai_l']:.3f}{stars(c['r']['ols'].p['ln_ai_l'])}" for c in combos],
            [""] + [f"({c['r']['ols'].se['ln_ai_l']:.3f})" for c in combos],
            None,
            ["First stage: instrument coefficient"] + [f"{c['r']['fs_coef']:.3f}{stars(c['r']['fs_p'])}" for c in combos],
            [""] + [f"({c['r']['fs_se']:.3f})" for c in combos],
            ["First-stage cluster-robust F"] + [f"{c['r']['F']:.2f}" for c in combos],
            None,
            ["Observations"] + [f"{c['r']['N']:,}" for c in combos],
            ["Countries"] + [f"{c['r']['G']}" for c in combos]]
    write_table("T7_iv", "Shift-share IV: AI capability and climate vulnerability/readiness",
                heads, body,
                "Columns 1-4: instrument = s$_i$ x (log W$_{-i,t}$ - log W$_{-i,2000}$), columns 5-8: alternative "
                "instrument log(1+s$_i$ W$_{-i,t}$/pop$_{i,t}$). Instrument and endogenous regressor lagged one year. "
                "All variables two-way demeaned on the estimation sample; 2SLS with country-clustered SEs (no "
                "degrees-of-freedom correction for absorbed FE). First-stage F is the squared cluster-robust t of the "
                "single excluded instrument. $^{*}$ p<0.1, $^{**}$ p<0.05, $^{***}$ p<0.01.",
                group=[("Shift-share instrument", 4), ("Level instrument", 4)])
    return combos


# ---------------------------------------------------------------------------
# T8 Robustness
# ---------------------------------------------------------------------------
def three_year_panel(P, y):
    D = P.reset_index()
    D["blk"] = (D["year"] - 2000) // 3
    cols = [y, "ln_ai_l"] + X
    g = D.groupby(["iso3", "blk"])
    m, c = g[cols].mean(), g[cols].count()
    m = m[(c >= 2).all(axis=1)]
    return m.dropna()


def table_T8(P):
    out = {}
    for y in MAIN_OUT:
        r = {}
        base = ["ln_ai_l"] + X
        # (a) AI share of publications (percentage points)
        r["a"] = fe_ols(model_sample(P, y, ["ai_share_l"] + X), y, ["ai_share_l"] + X).rename({"ai_share_l": "AI"})
        # (b) ML publications
        r["b"] = fe_ols(model_sample(P, y, ["ln_ml_l"] + X), y, ["ln_ml_l"] + X).rename({"ln_ml_l": "AI"})
        # (c) excluding China and the United States
        Pc = P.drop(index=["CHN", "USA"], level=0, errors="ignore")
        r["c"] = fe_ols(model_sample(Pc, y, base), y, ["ln_ai_l"] + X).rename({"ln_ai_l": "AI"})
        # (d) three-year non-overlapping averages (outcome and lagged regressors averaged per block)
        m3 = three_year_panel(P, y)
        r["d"] = fe_ols(m3, y, ["ln_ai_l"] + X).rename({"ln_ai_l": "AI"})
        # (e) Driscoll-Kraay standard errors (baseline specification)
        r["e"] = fe_ols(model_sample(P, y, base), y, ["ln_ai_l"] + X, cov="dk").rename({"ln_ai_l": "AI"})
        # (f) adding tertiary enrolment
        r["f"] = fe_ols(model_sample(P, y, base + ["tertiary_l"]), y, ["ln_ai_l", "tertiary_l"] + X).rename(
            {"ln_ai_l": "AI"})
        # (g) long differences 2005 -> 2022, OLS HC1, 2005 levels of controls
        a, b = P.xs(2005, level=1), P.xs(2022, level=1)
        D = pd.DataFrame({"dy": b[y] - a[y], "dai": b["ln_ai"] - a["ln_ai"], "ln_gdppc": a["ln_gdppc"],
                          "urban": a["urban"], "trade": a["trade"], "ln_pop": a["ln_pop"]}).dropna()
        regs = ["dai", "ln_gdppc", "urban", "trade", "ln_pop"]
        ols = sm.OLS(D["dy"], sm.add_constant(D[regs])).fit(cov_type="HC1")
        r["g"] = Fit.from_sm(ols, G=len(D)).rename({"dai": "AI"})
        r["g"].G = len(D)
        r["g"].N = len(D)
        out[y] = r
    KEY["T8"] = out
    cols = {y: [out[y][k] for k in "abcdefg"] for y in MAIN_OUT}
    heads = [f"({k})" for k in "abcdefg"]
    body = []
    for y in MAIN_OUT:
        body.append(("__panel__", f"Panel {'A' if y == 'vuln' else 'B'}: {OUTCOME_LABEL[y]}"))
        body += coef_rows(cols[y], [("AI", LABEL["AI"]), ("tertiary_l", X_LABEL["tertiary_l"])])
        body += [["Within R2 / R2 (g)"] + [f"{c.r2:.3f}" if np.isfinite(c.r2) else "" for c in cols[y]],
                 ["Observations"] + [f"{c.N:,}" for c in cols[y]],
                 ["Countries"] + [f"{c.G}" for c in cols[y]], None]
    write_table("T8_robustness", "Robustness of the baseline association",
                heads, body[:-1],
                "AI measure: (a) AI share of publications (pp); (b) ln(1+ML publications per million); (c) baseline "
                "excluding China and the USA; (d) three-year non-overlapping averages (outcome and lagged regressors "
                "averaged within blocks, block fixed effects); (e) baseline with Driscoll-Kraay SEs (kernel: Bartlett); "
                "(f) adding tertiary enrolment (t-1); (g) long differences 2005-2022 (OLS, HC1, 2005 control levels, "
                "no lags). Two-way FE with country-clustered SEs in (a)-(d) and (f). $^{*}$ p<0.1, $^{**}$ p<0.05, $^{***}$ p<0.01.")
    return out


# ---------------------------------------------------------------------------
# T9 Cross-section, latest year
# ---------------------------------------------------------------------------
def table_T9(P):
    aipi_path = RAW / "imf_aipi.csv"
    if not aipi_path.exists():
        print("[T9] skipped: data/raw/imf_aipi.csv not present")
        KEY["T9"] = None
        return None
    aipi = pd.read_csv(aipi_path).set_index("iso3")["aipi"]
    last = P.reset_index()
    gini_obs = last[last["gini_interp"] == 0].dropna(subset=["gini"]).sort_values("year").groupby("iso3").tail(1)
    gini_obs = gini_obs.set_index("iso3")[["gini", "year"]].rename(columns={"year": "gini_year"})
    y23 = P.xs(2023, level=1)
    D = pd.DataFrame({"vuln": y23["vuln"], "ready": y23["ready"], "ln_gdppc": y23["ln_gdppc"],
                      "aipi": aipi}).join(gini_obs, how="left")
    D = D.dropna(subset=["vuln", "ready", "ln_gdppc", "aipi", "gini"])
    D["gini_c"] = D["gini"] - D["gini"].mean()
    D["aipi_x_gini"] = D["aipi"] * D["gini_c"]
    KEY["T9_gini_year_median"] = float(D["gini_year"].median())
    cols = []
    for y in MAIN_OUT:
        m1 = sm.OLS(D[y], sm.add_constant(D[["aipi", "ln_gdppc", "gini"]])).fit(cov_type="HC1")
        m2 = sm.OLS(D[y], sm.add_constant(D[["aipi", "ln_gdppc", "gini_c", "aipi_x_gini"]])).fit(cov_type="HC1")
        cols += [Fit.from_sm(m1, G=len(D)), Fit.from_sm(m2, G=len(D))]
    KEY["T9"] = cols
    KEY["T9_N"] = len(D)
    terms = [("aipi", LABEL["aipi"]), ("ln_gdppc", LABEL["ln_gdppc"]), ("gini", LABEL["gini"]),
             ("gini_c", LABEL["gini_c"]), ("aipi_x_gini", LABEL["aipi_x_gini"]), ("const", LABEL["const"])]
    body = coef_rows(cols, terms)
    body += [None, ["Observations"] + [f"{c.N:,}" for c in cols],
             ["R2"] + [f"{c.r2:.3f}" for c in cols]]
    write_table("T9_crosssection", "Cross-section (2023): IMF AI Preparedness Index and climate outcomes",
                [f"({i})" for i in range(1, 5)], body,
                "Outcome year 2023 (ND-GAIN). IMF AI Preparedness Index (AIPI, 2023). Latest observed Gini (last "
                f"non-interpolated value in 2000-2023; median reference year {int(KEY['T9_gini_year_median'])}). "
                "Columns 1-2: vulnerability; columns 3-4: readiness. Columns 2 and 4 centre Gini on the sample mean. "
                "HC1 robust standard errors in parentheses. $^{*}$ p<0.1, $^{**}$ p<0.05, $^{***}$ p<0.01.",
                group=[("Vulnerability", 2), ("Readiness", 2)])
    return cols


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_trends(P):
    INC_ORDER = ["Low income", "Lower middle income", "Upper middle income", "High income"]
    styles = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D")]
    med = P.reset_index().groupby(["income_level", "year"])["ln_ai"].median().unstack(0)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (inc, (ls, mk)) in zip(INC_ORDER, styles):
        if inc not in med.columns:
            continue
        ax.plot(med.index, med[inc], color="black", ls=ls, marker=mk, ms=3.5, lw=1.3, markevery=3,
                label=inc)
    ax.set_xlabel("Year")
    ax.set_ylabel("Median AI capability, ln(1 + AI pubs per million)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "fig_trends")


def fig_scatter(P):
    D = P.reset_index()
    a = D[D["year"] == 2005].set_index("iso3")
    b = D[D["year"] == 2022].set_index("iso3")
    diff = pd.DataFrame({"dai": b["ln_ai"] - a["ln_ai"], "dvuln": b["vuln"] - a["vuln"],
                         "emde": a["emde"]}).dropna()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for emde_val, mk, fc, lab in [(1.0, "o", "black", "EMDE"), (0.0, "^", "white", "High income")]:
        d = diff[diff["emde"] == emde_val]
        ax.scatter(d["dai"], d["dvuln"], marker=mk, facecolor=fc, edgecolor="black", s=22, lw=0.7, label=lab)
    slope, intercept = np.polyfit(diff["dai"], diff["dvuln"], 1)
    xs = np.linspace(diff["dai"].min(), diff["dai"].max(), 50)
    ax.plot(xs, intercept + slope * xs, color="0.4", ls="--", lw=1, label="OLS fit (all)")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Change in AI capability, ln(1+AI pubs per m), 2005-2022")
    ax.set_ylabel("Change in ND-GAIN vulnerability, 2005-2022")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "fig_scatter")
    KEY["scatter"] = {"N": int(len(diff)), "slope": float(slope), "N_emde": int((diff["emde"] == 1).sum()),
                      "N_high": int((diff["emde"] == 0).sum())}


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------
def fmt_fit_line(name, f, var):
    return f"- {name}: {f.b[var]:.3f} (SE {f.se[var]:.3f}, p={f.p[var]:.3f}{stars(f.p[var])}), N={f.N:,}, countries={f.G}"


def write_summary(P, meta_bundle):
    lines = ["# Results summary", "",
             "Generated by `code/03_analysis.py` (seed 42). Every number below is produced by the code; "
             "coefficients are shown as coef (SE), with stars (* p<0.1, ** p<0.05, *** p<0.01). "
             "Standard errors are country-clustered unless stated otherwise.", "",
             "## Samples", "",
             f"- Panel: {P.reset_index()['iso3'].nunique()} countries (population >= 1 million in 2010), "
             f"years 2000-2023, {len(P)} country-years.",
             f"- EMDE subsample (income group other than High income): {KEY['T1_N_emde']} countries.",
             "- Outcomes contemporaneous; AI measures and controls lagged one year within country.",
             "- Centred moderators use the mean of each estimation sample (outcome-specific base samples in T3/T5/T6).",
             "- Within R2: the linearmodels `rsquared_within` after absorbing country and year effects.", ""]

    # Headline numbers
    T2 = KEY["T2"]
    lines += ["## Headline numbers", ""]
    names = ["vuln full", "ready full", "gain full", "vuln EMDE", "ready EMDE", "gain EMDE"]
    for nm, f in zip(names, T2):
        lines.append(fmt_fit_line(f"T2 {nm}: ln AI (t-1)", f, "ln_ai_l"))
    lines.append("")
    for nm, f in zip(["vuln", "ready"], [KEY["T3"][0], KEY["T3"][3]]):
        lines.append(fmt_fit_line(f"T3 {nm} (gini model): ln AI (t-1)", f, "ln_ai_l"))
        lines.append(fmt_fit_line(f"T3 {nm} (gini model): AI x Gini", f, "ai_x_gini"))
    for nm, f in zip(["vuln", "ready"], [KEY["T3"][1], KEY["T3"][4]]):
        lines.append(fmt_fit_line(f"T3 {nm} (internet model): AI x Internet", f, "ai_x_internet"))
    lines.append("")
    me = KEY["ME"]
    lines.append("### Marginal effect of ln AI capability on vulnerability (joint model, T3 col 3)")
    for key, lab in [("gini", "Gini (t-1)"), ("internet", "Internet use % (t-1)")]:
        for q in ["p10", "p50", "p90"]:
            m, lo, hi = me[key][q + "_me"]
            lines.append(f"- over {lab} at {q} = {me[key][q]:.2f}: ME {m:.3f} (95% CI {lo:.3f}, {hi:.3f})")
    lines.append("")
    for y in MAIN_OUT:
        th = KEY["T4"][y]
        lines += [f"### T4 threshold, {y}",
                  f"- gamma-hat (Gini, t-1) = {th['gamma']:.2f}; 95% CI (LR<=7.35 grid) = [{th['ci'][0]:.2f}, {th['ci'][1]:.2f}]",
                  f"- threshold F = {th['F']:.2f}; bootstrap p = {th['p']:.3f} ({th['B']} reps)",
                  f"- AI slope, gini <= gamma-hat: {th['slopes'].b['ai_lo']:.3f} (SE {th['slopes'].se['ai_lo']:.3f}, "
                  f"p={th['slopes'].p['ai_lo']:.3f}), N={th['N_lo']:,}",
                  f"- AI slope, gini > gamma-hat: {th['slopes'].b['ai_hi']:.3f} (SE {th['slopes'].se['ai_hi']:.3f}, "
                  f"p={th['slopes'].p['ai_hi']:.3f}), N={th['N_hi']:,}",
                  f"- linear AI slope (reference): {th['linear'].b['ln_ai_l']:.3f} (SE {th['linear'].se['ln_ai_l']:.3f})",
                  f"- total N = {th['N']:,}, countries = {th['G']}", ""]
    lines += ["### T6 environmental cost"]
    for nm, f in zip(["ln elec", "ln elec + ren", "ln CO2pc", "ln CO2pc + ren"], KEY["T6"]):
        var = "ln_ai_l"
        lines.append(fmt_fit_line(f"T6 {nm}: ln AI (t-1)", f, var))
    lines.append("")
    lines += ["### T7 IV (shift-share)"]
    for c in KEY["T7"]:
        r = c["r"]
        lines.append(f"- {c['instr']} | {OUTCOME_LABEL[c['y']]} | {c['sample']}: IV {r['iv'].b['ln_ai_l']:.3f} "
                     f"(SE {r['iv'].se['ln_ai_l']:.3f}, p={r['iv'].p['ln_ai_l']:.3f}); OLS {r['ols'].b['ln_ai_l']:.3f}; "
                     f"first stage {r['fs_coef']:.3f} (SE {r['fs_se']:.3f}), F={r['F']:.2f}; N={r['N']:,}, "
                     f"countries={r['G']}")
    lines.append("")
    if KEY.get("T9"):
        lines += ["### T9 cross-section 2023", f"- N = {KEY['T9_N']} countries"]
        for nm, f in zip(["vuln M1", "vuln M2", "ready M1", "ready M2"], KEY["T9"]):
            lines.append(f"- {nm}: aipi {f.b['aipi']:.3f} (SE {f.se['aipi']:.3f}, p={f.p['aipi']:.3f}{stars(f.p['aipi'])})")
        lines.append("")
    sc = KEY["scatter"]
    lines += ["### Long differences and scatter (2005-2022)",
              f"- countries in scatter: {sc['N']} (EMDE {sc['N_emde']}, high income {sc['N_high']}); "
              f"bivariate OLS slope of change in vulnerability on change in ln AI: {sc['slope']:.3f}", ""]

    lines += ["## Tables (full output)", "",
              "Each table is also saved as `results/tables/<name>.csv` and `.tex`.", ""]
    lines += SUMMARY_SECTIONS
    lines += ["## Figures", "",
              "- `results/figures/fig_trends` (PDF/PNG): median ln AI by World Bank income group, 2000-2023.",
              "- `results/figures/fig_scatter`: change in ln AI vs change in ND-GAIN vulnerability, 2005-2022; "
              "EMDE (filled circles) vs high income (open triangles), OLS line.",
              "- `results/figures/fig_marginal_effects`: marginal effect of ln AI on vulnerability over Gini and internet "
              "use (10th-90th percentile), joint model, 95% CI.",
              "- `results/figures/fig_threshold`: LR(gamma) for vulnerability, 7.35 line.", ""]
    lines += ["## Method notes and deviations", ""] + [f"- {n}" for n in METHOD_NOTES] + [""]
    (RESULTS / "results_summary.md").write_text("\n".join(lines))
    print("[summary] wrote results/results_summary.md")


METHOD_NOTES = [
    "OpenAlex `group_by` returns at most 200 groups and rejects a `page` parameter. Where the all-works list was "
    "truncated, every World Bank country missing from the list was queried directly (see codebook build notes).",
    "Panel countries: population >= 1 million in 2010; grid is balanced over 2000-2023 before missing values are dropped.",
    "Gini interpolated linearly within country on the 1995-2024 WDI series (inside observed points only).",
    "Hansen threshold: exact FE partialling via projection on country and year dummies; sigma^2 = SSR/n (Hansen 1999); "
    "100 grid points between the 15th and 85th percentiles of lagged Gini; each regime must contain observations by construction.",
    "Hansen wild cluster bootstrap: Rademacher weights by country, 300 replications, under the linear null (FE residuals fixed).",
    "IV: two-way demeaned variables; linearmodels IV2SLS does not correct df for absorbed FE, so IV SEs are slightly "
    "too small relative to a full FE-IV correction.",
    "T8(g) long differences follow the specification as written (2005-2022 changes, 2005 control levels, no lags).",
    "T9 uses year 2023 ND-GAIN outcomes and the latest interpolation-free Gini observed in 2000-2023.",
    "Marginal effects in fig_marginal_effects use the joint model (T3 column 3) with the other moderator held at its mean.",
]


# ---------------------------------------------------------------------------
def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    P = load_panel()
    print(f"[panel] {P.shape}, countries={P.reset_index()['iso3'].nunique()}")
    KEY["N_panel_countries"] = int(P.reset_index()["iso3"].nunique())
    P_all = P.copy()
    table_T1(P)
    table_T2(P)
    cols3, meta3 = table_T3(P)
    out_me = marginal_effects(meta3)
    fig_marginal(out_me)
    res4 = table_T4(P)
    fig_threshold(res4)
    table_T5(P)
    table_T6(P)
    table_T7(P)
    table_T8(P)
    table_T9(P)
    fig_trends(P_all)
    fig_scatter(P)
    write_summary(P, None)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
