"""Helpers for the empirical application (code/30_application.py).

Contents
--------
write_table       CSV + booktabs LaTeX table writer
fig_style         grayscale-friendly matplotlib style and save helper
plr_crossfit      cross-fitted partially linear (DML) residuals and the orthogonal score
plr_estimate      DML point estimate, country-clustered SE and per-unit orthogonal score
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402

# grayscale-friendly palette: black, dark gray, mid gray; distinct markers and line styles
BLACK = "#000000"
DARK = "#404040"
MID = "#8c8c8c"
LIGHT = "#c8c8c8"
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", ":", "-."]


def _esc(s) -> str:
    s = str(s)
    if "$" in s or "\\" in s:  # already LaTeX
        return s
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
        s = s.replace(a, b)
    return s


def write_table(df: pd.DataFrame, stem: Path, caption: str, label: str, notes: str | None = None,
                align: str | None = None) -> None:
    """Write ``stem.csv`` and ``stem.tex`` (booktabs). Cells must already be formatted strings
    or numbers; NaN cells are shown as '--'."""
    stem = Path(stem)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    cols = list(df.columns)
    align = align or ("l" + "r" * (len(cols) - 1))
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{_esc(caption)}}}",
             rf"\label{{{label}}}", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(_esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for v in row.to_list():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("--")
            else:
                cells.append(_esc(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(r"\par\smallskip\footnotesize " + _esc(notes))
    lines.append(r"\end{table}")
    stem.with_suffix(".tex").write_text("\n".join(lines) + "\n")


def fmt(x, nd: int = 3) -> str:
    """Format a float for tables; NaN -> '--'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def fmt_se(x, se, nd: int = 3) -> str:
    """Coefficient with SE in parentheses on one cell, e.g. '0.123 (0.045)'."""
    if se is None or (isinstance(se, float) and np.isnan(se)):
        return fmt(x, nd)
    return f"{x:.{nd}f} ({se:.{nd}f})"


def fig_style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 300,
        "pdf.fonttype": 42, "axes.grid": False,
    })


def save_fig(fig, stem: Path) -> None:
    stem = Path(stem)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- DML (partially linear)
def plr_crossfit(y, d, X, folds, regressor):
    """Out-of-fold E[y|X] and E[d|X]; returns residuals (ry, rd)."""
    y = np.asarray(y, float)
    d = np.asarray(d, float)
    X = np.asarray(X, float)
    y_hat = np.empty(len(y))
    d_hat = np.empty(len(d))
    for k in np.unique(folds):
        te = folds == k
        tr = ~te
        y_hat[te] = clone(regressor).fit(X[tr], y[tr]).predict(X[te])
        d_hat[te] = clone(regressor).fit(X[tr], d[tr]).predict(X[te])
    return y - y_hat, d - d_hat


def plr_estimate(ry, rd, clusters):
    """theta = sum(rd*ry)/sum(rd^2); country-clustered SE (CR1); unit-level orthogonal score
    psi_i = theta + phi_i with phi_i = rd_i (ry_i - theta rd_i) / E[rd^2] (mean(phi) = 0)."""
    ry = np.asarray(ry, float)
    rd = np.asarray(rd, float)
    n = len(ry)
    denom = np.sum(rd ** 2)
    theta = float(np.sum(rd * ry) / denom)
    phi = rd * (ry - theta * rd) / (denom / n)
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    G = inv.max() + 1
    sums = np.bincount(inv, weights=phi, minlength=G)
    var = G / (G - 1) * np.sum(sums ** 2) / n ** 2
    return theta, float(np.sqrt(var)), theta + phi
