"""Equity-Weighted Doubly Robust Policy Learning (EW-DRPL).

Typical use (long format, one row per spell x decile)::

    from ewdrpl import EWDRPL
    est = EWDRPL(K=5, seed=42).fit(
        df, unit="spell", group="country", decile="d", outcome="Y", shock="S", treat="T",
        covariates=["lgdp", "gini", "internet", "agri", "btemp", "ksov"], income="income")
    B = est.benefit(eps=1)                 # country benefits under Atkinson eps = 1
    tree = est.learn_policy(eps=1, k=0.3)  # depth-2 tree under budget share 0.3

Modules: nuisance (cross-fitting), scores (psi, psi*, Gamma), welfare (Atkinson weights, B_i),
policy (exact depth-2 tree search), inference (cluster SEs, bootstrap, BLP, GATES, CLAN).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .inference import blp as _blp
from .inference import clan as _clan
from .inference import cluster_bootstrap, cluster_mean_se, gates as _gates
from .nuisance import make_folds
from .policy import TreePolicy, crossfit_policy, policy_value, search_tree, topk_policy
from .scores import compute_scores
from .welfare import atkinson_weights, benefit as _benefit, population_scale

__all__ = ["EWDRPL", "TreePolicy", "search_tree", "topk_policy", "crossfit_policy",
           "policy_value", "atkinson_weights", "make_folds", "compute_scores"]

_SCORE_KEYS = {"dr": "gamma_dr", "dr_raw": "gamma_raw", "ipw": "gamma_ipw", "plugin": "rho_plugin"}


class EWDRPL:
    """Equity-weighted doubly robust policy learner.

    Parameters
    ----------
    K : number of country folds for cross-fitting (default 5).
    seed : fold seed (default 42).
    regressor, classifier : sklearn estimators for outcome and propensity nuisances.
    shock_regressor, variance_regressor : estimators for s_hat and v_hat (default: regressor).
    e_clip : propensity clipping interval (default (0.1, 0.9), following the Crump et al. trimming rule; see the Monte Carlo diagnostics).
    v_floor, v_floor_frac : v_hat floor = max(v_floor, v_floor_frac * Var(S)).
    orthogonal : if True (default) the AIPW score uses the orthogonalised psi*; if False the
        raw Robinson psi. Both are always stored (gamma_dr and gamma_raw).
    """

    def __init__(self, K=5, seed=42, regressor=None, classifier=None, shock_regressor=None,
                 variance_regressor=None, e_clip=(0.1, 0.9), v_floor=1e-3, v_floor_frac=0.05,
                 orthogonal=True):
        self.K = K
        self.seed = seed
        self.regressor = regressor
        self.classifier = classifier
        self.shock_regressor = shock_regressor
        self.variance_regressor = variance_regressor
        self.e_clip = e_clip
        self.v_floor = v_floor
        self.v_floor_frac = v_floor_frac
        self.orthogonal = orthogonal

    # ------------------------------------------------------------------ fitting
    def fit(self, df, *, unit, group, decile, outcome, shock, treat, covariates, income,
            population=None, exposure=None):
        """Fit nuisances and scores.

        df : long DataFrame with one row per (unit, decile).
        unit : spell id; group : country id; decile : decile index 1..D;
        outcome : Y_d; shock : S (constant within spell); treat : T in {0,1} (constant within
        spell); covariates : list of X columns (constant within spell); income : decile mean
        income y_id (positive; constant within spell is typical); population : optional
        country population (constant within country); exposure : optional country-level s_bar
        (default: SD of S across the country's spells, ddof=1).
        """
        covariates = list(covariates)
        required = [unit, group, decile, outcome, shock, treat, income, *covariates]
        if population:
            required.append(population)
        if exposure:
            required.append(exposure)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"missing columns: {missing}")
        if not set(pd.unique(df[treat])) <= {0, 1}:
            raise ValueError("treatment must be coded 0/1")

        spell_cols = [group, shock, treat, *covariates]
        if population:
            spell_cols.append(population)
        if exposure:
            spell_cols.append(exposure)
        g = df.groupby(unit, sort=True)
        if (g[spell_cols].nunique() > 1).any().any():
            raise ValueError("group, shock, treatment, covariates, population and exposure "
                             "must be constant within spell")
        sp = g[spell_cols].first()
        Y = df.pivot(index=unit, columns=decile, values=outcome).reindex(sp.index)
        inc = df.pivot(index=unit, columns=decile, values=income).reindex(sp.index)
        if Y.isna().any().any() or inc.isna().any().any():
            raise ValueError("every spell must have every decile")
        if (inc.to_numpy() <= 0).any():
            raise ValueError("decile incomes must be positive")
        decile_levels = np.asarray(Y.columns)

        groups = sp[group].to_numpy()
        countries = np.sort(np.unique(groups))
        folds_spell = make_folds(groups, K=self.K, seed=self.seed)

        cg = sp.groupby(group)
        if exposure:
            if (cg[exposure].nunique() > 1).any():
                raise ValueError("exposure must be constant within country")
            s_bar = cg[exposure].first().reindex(countries).to_numpy(float)
        else:
            s_bar = cg[shock].std(ddof=1).reindex(countries).to_numpy(float)
            if np.isnan(s_bar).any():
                raise ValueError("need at least 2 spells per country to compute exposure")
        if population:
            pop = cg[population].first().reindex(countries).to_numpy(float)
            w = pop / pop.sum()
        else:
            w = None

        X = sp[covariates].to_numpy(float)
        S = sp[shock].to_numpy(float)
        T = sp[treat].to_numpy(float)
        sc = compute_scores(
            X, S, T, Y.to_numpy(float), folds_spell, groups,
            regressor=self.regressor, classifier=self.classifier,
            shock_regressor=self.shock_regressor, variance_regressor=self.variance_regressor,
            e_clip=self.e_clip, v_floor=self.v_floor, v_floor_frac=self.v_floor_frac,
            seed=self.seed, orthogonal=self.orthogonal,
        )

        self.countries_ = countries
        self.deciles_ = decile_levels
        self.covariates_ = covariates
        self.s_bar_ = s_bar
        self.w_ = w
        self.fold_ = np.array([folds_spell[groups == c][0] for c in countries])
        self._groups = groups
        self._spell_index = sp.index.to_numpy()
        self._spell_X = X
        self._spell_scores = sc
        self._spell_gamma = {k: sc[v] for k, v in _SCORE_KEYS.items()}
        self._spell_gamma["dr"] = sc["gamma"] if self.orthogonal else sc["gamma_dr"]
        self._income = inc.to_numpy(float)

        self.country_cov_ = pd.DataFrame(X, columns=covariates).groupby(groups).mean().reindex(countries)
        self.Z_ = self.country_cov_.to_numpy()
        self.mats_ = {k: self._country_mean(v) for k, v in self._spell_gamma.items()}
        self.mats_["income"] = self._country_mean(self._income)

        # long table of scores for inspection
        n, D = sc["gamma_dr"].shape
        self.scores_ = pd.DataFrame({
            "unit": np.repeat(self._spell_index, D),
            "country": np.repeat(groups, D),
            "d": np.tile(decile_levels, n),
            "Y": Y.to_numpy(float).ravel(),
            "S": np.repeat(S, D),
            "T": np.repeat(T, D),
            "e_hat": np.repeat(sc["e_hat"], D),
            "s_hat": np.repeat(sc["s_hat"], D),
            "v_hat": np.repeat(sc["v_hat"], D),
            "ell": sc["ell"].ravel(), "psi_raw": sc["psi_raw"].ravel(),
            "psi_star": sc["psi_star"].ravel(), "m1": sc["m1"].ravel(), "m0": sc["m0"].ravel(),
            "gamma_dr": sc["gamma_dr"].ravel(), "gamma_raw": sc["gamma_raw"].ravel(),
            "gamma_ipw": sc["gamma_ipw"].ravel(), "rho_plugin": sc["rho_plugin"].ravel(),
        })
        return self

    def _country_mean(self, mat):
        return pd.DataFrame(mat, index=self._groups).groupby(level=0).mean().reindex(self.countries_).to_numpy()

    # ---------------------------------------------------------------- benefits
    def country_matrix(self, which="dr") -> pd.DataFrame:
        """Country x decile matrix of averaged scores (which: dr, dr_raw, ipw, plugin)."""
        return pd.DataFrame(self.mats_[which], index=self.countries_, columns=self.deciles_)

    def _weights(self, weighted):
        G = len(self.countries_)
        if weighted:
            if self.w_ is None:
                raise ValueError("weighted=True requires a population column in fit()")
            return self.w_, population_scale(self.w_)
        return np.full(G, 1.0 / G), np.ones(G)

    def benefit(self, eps, which="dr", weighted=False) -> pd.Series:
        """Country benefits B_i(eps) = s_bar_i n_i (1/D) sum_d omega_id(eps) Gamma_id."""
        w, scale = self._weights(weighted)
        omega = atkinson_weights(self.mats_["income"], eps, w)
        gamma = self.mats_[which]
        B = self.s_bar_ * scale * np.sum(omega * gamma, axis=1) / gamma.shape[1]
        return pd.Series(B, index=self.countries_, name=f"B_eps{eps}_{which}")

    def _Z(self, features=None):
        if features is None:
            return self.Z_, list(self.covariates_)
        return self.country_cov_[list(features)].to_numpy(), list(features)

    def learn_policy(self, eps, k=0.3, which="dr", weighted=False, depth=2, features=None) -> TreePolicy:
        """Exact depth-2 tree maximising sum_i pi_i B_i(eps) under mean(pi) <= k.

        B_i already carries the population scale n_i when weighted=True, so no extra weights
        are passed to the search (passing them would count n_i twice)."""
        Z, names = self._Z(features)
        B = self.benefit(eps, which, weighted).to_numpy()
        return search_tree(Z, B, k, depth=depth, feature_names=names)

    def evaluate(self, eps, k=0.3, which="dr", weighted=False, depth=2, features=None,
                 n_boot=500, boot_seed=0) -> dict:
        """Policy value on the sample, cross-fitted held-out value, and country-cluster bootstrap."""
        Z, names = self._Z(features)
        B = self.benefit(eps, which, weighted).to_numpy()  # carries n_i when weighted
        tree = search_tree(Z, B, k, depth=depth, feature_names=names)
        pi = tree.predict(Z)
        pi_cf, v_cf, _ = crossfit_policy(Z, B, k, self.fold_, depth=depth)
        contrib = pi * B
        boot = cluster_bootstrap(contrib, self.countries_, n_boot=n_boot, seed=boot_seed)
        return {"policy": tree, "pi": pi, "value": policy_value(pi, B),
                "crossfit_pi": pi_cf, "crossfit_value": v_cf, "boot": boot, "B": B}

    def compare(self, eps, k=0.3, which="dr", weighted=False, depth=2, n_boot=500, boot_seed=0) -> dict:
        """Equity price and gain of the eps-policy relative to the utilitarian (eps=0) policy."""
        Z, _ = self._Z()
        B0 = self.benefit(0.0, which, weighted).to_numpy()  # carries n_i when weighted
        Be = self.benefit(eps, which, weighted).to_numpy()
        pi0 = search_tree(Z, B0, k, depth=depth).predict(Z).astype(float)
        pie = search_tree(Z, Be, k, depth=depth).predict(Z).astype(float)
        c_price = (pi0 - pie) * B0  # unit contributions to V0(pi0) - V0(pie)
        c_gain = (pie - pi0) * Be   # unit contributions to Ve(pie) - Ve(pi0)
        cp = cluster_bootstrap(c_price, self.countries_, n_boot=n_boot, seed=boot_seed)
        cg = cluster_bootstrap(c_gain, self.countries_, n_boot=n_boot, seed=boot_seed + 1)
        return {
            "equity_price": cp["estimate"], "equity_price_ci": cp["ci"],
            "equity_gain": cg["estimate"], "equity_gain_ci": cg["ci"],
            "pi_utilitarian": pi0, "pi_eps": pie,
        }

    # ---------------------------------------------------- layers 1-2 summaries
    def _spell_frame(self, which="dr", decile=None):
        if decile is None:
            gam = self._spell_gamma[which].mean(axis=1)
            rho = self._spell_gamma["plugin"].mean(axis=1)
        else:
            j = int(np.flatnonzero(self.deciles_ == decile)[0])
            gam = self._spell_gamma[which][:, j]
            rho = self._spell_gamma["plugin"][:, j]
        return gam, rho

    def ate(self, which="dr", decile=None) -> dict:
        """Average resilience effect with country-clustered SE."""
        gam, _ = self._spell_frame(which, decile)
        m, se = cluster_mean_se(gam, self._groups)
        return {"ate": m, "se": se}

    def decile_profile(self, which="dr") -> pd.DataFrame:
        """Resilience incidence curve: mean Gamma_d by decile with cluster SEs."""
        rows = []
        for d in self.deciles_:
            m, se = cluster_mean_se(self._spell_gamma[which][:, list(self.deciles_).index(d)], self._groups)
            rows.append({"d": d, "mean": m, "se": se})
        return pd.DataFrame(rows)

    def blp(self, features=None, which="dr", decile=None):
        gam, _ = self._spell_frame(which, decile)
        Z = self._spell_X if features is None else pd.DataFrame(
            self._spell_X, columns=self.covariates_)[list(features)].to_numpy()
        names = self.covariates_ if features is None else list(features)
        return _blp(gam, Z, self._groups, names)

    def gates(self, which="dr", n_groups=5):
        """GATES by quintile of out-of-fold predicted rho (plug-in m1 - m0, averaged over deciles)."""
        gam, rho = self._spell_frame(which, None)
        return _gates(rho, gam, self._groups, n_groups)

    def clan(self, features=None, n_groups=5):
        _, rho = self._spell_frame("dr", None)
        names = self.covariates_ if features is None else list(features)
        Z = pd.DataFrame(self._spell_X, columns=self.covariates_)[names].to_numpy()
        return _clan(Z, rho, self._groups, names, n_groups)
