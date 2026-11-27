"""Equity weights and country-level benefits (design section 4, Step 4).

For country i with D deciles (p_d = 1/D each), mean decile income y_id and mean resilience score
Gamma_id (averaged over the country's spells):

    omega_id(eps) = (y_id / ybar)^(-eps) / norm(eps),
    ybar          = sum_i w_i sum_d (1/D) y_id,
    norm(eps)     = sum_i w_i sum_d (1/D) (y_id / ybar)^(-eps),

so that the population-weighted mean of omega equals 1 (w_i are country population shares, or
1/N if unweighted). eps = 0 gives omega = 1 (utilitarian). eps = 1 is log utility, eps = 2 strongly
prioritarian.

    B_i(eps) = s_bar_i * n_i * (1/D) sum_d omega_id(eps) Gamma_id

with n_i = N w_i (so n_i = 1 when countries are unweighted).
"""

from __future__ import annotations

import numpy as np


def normalise_weights(w) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative with positive sum")
    return w / w.sum()


def atkinson_weights(income, eps: float, w=None) -> np.ndarray:
    """Normalised Atkinson weights omega (G, D) from decile incomes (G, D)."""
    income = np.asarray(income, dtype=float)
    G, D = income.shape
    if np.any(income <= 0):
        raise ValueError("decile incomes must be strictly positive")
    w = np.full(G, 1.0 / G) if w is None else normalise_weights(w)
    ybar = np.sum(w[:, None] * income) / D
    omega = (income / ybar) ** (-float(eps))
    norm = np.sum(w[:, None] * omega) / D
    return omega / norm


def benefit(gamma_bar, omega, s_bar, w=None) -> np.ndarray:
    """Country benefits B_i(eps) = s_bar_i * n_i * mean_d(omega_id * Gamma_id)."""
    gamma_bar = np.asarray(gamma_bar, dtype=float)
    G, D = gamma_bar.shape
    scale = np.ones(G) if w is None else G * normalise_weights(w)
    return np.asarray(s_bar, dtype=float) * scale * np.sum(omega * gamma_bar, axis=1) / D


def population_scale(w) -> np.ndarray:
    """n_i = G * w_i, the mean-one scale used for population-weighted objectives."""
    w = np.asarray(w, dtype=float)
    return len(w) * normalise_weights(w)
