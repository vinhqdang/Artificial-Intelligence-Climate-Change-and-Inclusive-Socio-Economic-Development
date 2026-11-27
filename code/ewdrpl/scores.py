"""Pseudo-outcomes and doubly robust resilience scores (design section 4, Steps 1-3).

Notation (per unit i = survey spell, per decile d):
    S = climate shock, T in {0,1} = climate-AI capability, X = covariates,
    Y_d = decile outcome, s_hat = E[S|X,T], v_hat = E[(S-s_hat)^2|X,T], ell_d = E[Y_d|X,T],
    e_hat = P(T=1|X).

Step 2 (raw Robinson slope pseudo-outcome):
    psi_d = (Y_d - ell_d) (S - s_hat) / v_hat,        E[psi_d | X,T] = beta_d(X,T)

Step 3 (orthogonalised slope score, the default):
    psi*_d = m_{d,T}(X) + [ (Y_d - ell_d)(S - s_hat) - m_{d,T}(X) (S - s_hat)^2 ] / v_hat
    where m_{d,t}(X) = E[psi_d | X, T=t] is the cross-fitted preliminary slope fit on raw psi.
    E[psi*_d | X,T] = beta_d(X,T) and psi* has zero first-order sensitivity to v_hat.

    Gamma_d = m_{d,1} - m_{d,0} + T (psi*_d - m_{d,1})/e_hat - (1-T)(psi*_d - m_{d,0})/(1-e_hat)

With ``orthogonal=False`` the raw psi_d replaces psi*_d in the AIPW score (used for the
Monte Carlo comparison method "non-orthogonalised slope score"). The IPW-only score and the
plug-in score are also returned.

Implementation note (two-stage cross-fitting): the m-regressions use the same country folds as
the nuisances, and the preliminary m is fitted on raw psi values that were themselves built from
out-of-fold nuisances. Both m_{d,1} and m_{d,0} are cross-fitted, and m is the same object in
psi* and in Gamma.
"""

from __future__ import annotations

import numpy as np

from .nuisance import (
    crossfit_propensity,
    crossfit_regression,
    crossfit_shock_moments,
    default_classifier,
    default_regressor,
)


def robinson_pseudo_outcome(Y, S, ell, s_hat, v_hat) -> np.ndarray:
    """psi = (Y - ell)(S - s_hat)/v_hat."""
    return (Y - ell) * (S - s_hat) / v_hat


def orthogonal_pseudo_outcome(Y, S, ell, s_hat, v_hat, m_T) -> np.ndarray:
    """psi* = m_T + [(Y - ell)(S - s_hat) - m_T (S - s_hat)^2] / v_hat."""
    r = S - s_hat
    return m_T + ((Y - ell) * r - m_T * r ** 2) / v_hat


def dr_score(psi, T, m1, m0, e) -> np.ndarray:
    """AIPW resilience score m1 - m0 + T(psi - m1)/e - (1-T)(psi - m0)/(1-e)."""
    return m1 - m0 + T * (psi - m1) / e - (1 - T) * (psi - m0) / (1 - e)


def ipw_score(psi, T, e) -> np.ndarray:
    """IPW-only resilience score T psi/e - (1-T) psi/(1-e)."""
    return T * psi / e - (1 - T) * psi / (1 - e)


def plugin_score(m1, m0) -> np.ndarray:
    """Plug-in resilience score m1 - m0 (no doubly robust correction)."""
    return m1 - m0


def compute_scores(X, S, T, Y, folds, groups, *, regressor=None, classifier=None,
                   shock_regressor=None, variance_regressor=None, e_clip=(0.05, 0.95),
                   v_floor=1e-3, v_floor_frac=0.05, seed=42, orthogonal=True) -> dict:
    """Cross-fitted nuisances and all per-decile scores.

    Parameters
    ----------
    X : (n, p) covariates, S : (n,) shock, T : (n,) 0/1, Y : (n, D) decile outcomes,
    folds : (n,) fold ids (country folds), groups : (n,) country ids.

    Returns a dict of arrays. Unit-level: s_hat, v_hat, e_hat (n,).
    Decile-level (n, D): ell, psi_raw, psi_star, m1, m0, gamma (AIPW with the selected
    orthogonal flag), gamma_dr (orthogonal AIPW), gamma_raw (raw-psi AIPW),
    gamma_ipw, rho_plugin.
    """
    regressor = default_regressor() if regressor is None else regressor
    classifier = default_classifier() if classifier is None else classifier
    shock_regressor = regressor if shock_regressor is None else shock_regressor
    variance_regressor = regressor if variance_regressor is None else variance_regressor

    X = np.asarray(X, dtype=float)
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, D = Y.shape
    XT = np.column_stack([X, T])

    s_hat, v_hat = crossfit_shock_moments(
        XT, S, folds, groups, shock_regressor, variance_regressor, seed=seed,
        v_floor=v_floor, v_floor_frac=v_floor_frac,
    )
    e_hat = crossfit_propensity(X, T, folds, classifier, clip=e_clip)

    out = {k: np.empty((n, D)) for k in
           ["ell", "psi_raw", "psi_star", "m1", "m0", "gamma_dr", "gamma_raw", "gamma_ipw",
            "rho_plugin", "gamma"]}
    Tc = T[:, None]
    ec = e_hat[:, None]
    for j in range(D):
        y = Y[:, j]
        ell = crossfit_regression(XT, y, folds, regressor)
        psi_raw = robinson_pseudo_outcome(y, S, ell, s_hat, v_hat)
        # preliminary cross-fitted slope fits m_{d,t}(X) on raw psi
        m1 = crossfit_regression(X, psi_raw, folds, regressor, train_mask=(T == 1))
        m0 = crossfit_regression(X, psi_raw, folds, regressor, train_mask=(T == 0))
        m_T = T * m1 + (1 - T) * m0
        psi_star = orthogonal_pseudo_outcome(y, S, ell, s_hat, v_hat, m_T)

        out["ell"][:, j] = ell
        out["psi_raw"][:, j] = psi_raw
        out["psi_star"][:, j] = psi_star
        out["m1"][:, j] = m1
        out["m0"][:, j] = m0
        out["gamma_dr"][:, j] = dr_score(psi_star, Tc[:, 0], m1, m0, e_hat)
        out["gamma_raw"][:, j] = dr_score(psi_raw, Tc[:, 0], m1, m0, e_hat)
        out["gamma_ipw"][:, j] = ipw_score(psi_raw, Tc[:, 0], e_hat)
        out["rho_plugin"][:, j] = plugin_score(m1, m0)
        out["gamma"][:, j] = out["gamma_dr"][:, j] if orthogonal else out["gamma_raw"][:, j]

    out["s_hat"] = s_hat
    out["v_hat"] = v_hat
    out["e_hat"] = e_hat
    return out
