"""Semiparametric DR-LP-DiD estimator.

Implements the doubly-robust, IPW, and regression-adjustment variants of the
local-projection DiD estimator with semiparametric nuisance models.  The key
idea is to apply a semiparametric correction—via propensity score weighting
(IPW), outcome regression (RA), or a doubly-robust combination (DR)—to each
horizon-h clean local comparison stack before computing the ATT.

Three estimation methods are supported:

* ``'ra'`` — regression adjustment (outcome model only);
* ``'ipw'`` — inverse probability weighting (propensity score only);
* ``'dr'`` — doubly robust (both models; consistent if *either* is correctly
  specified).

For the DR method, two specifications are available:

* ``'generic'`` — logistic propensity score + OLS outcome regression;
* ``'improved'`` — IPT propensity score (Graham–Pinto–Egel) +
  IPT-weighted least-squares outcome regression on clean controls, following
  the improved DR-DiD logic of Sant'Anna and Zhao.

Inference is handled by stacked GMM influence functions (``'cluster'``),
multiplier/wild bootstrap simultaneous bands (``'multiplier'``), or paired
cluster bootstrap (``'cluster_bootstrap'``).

The public ``fit`` method accepts either a single ``covariates`` list used in
both nuisance equations, or separate ``ps_covariates`` and ``or_covariates``
lists.  This is useful for Monte Carlo designs that require the propensity
score to be misspecified while keeping the untreated-outcome regression
correctly specified, or vice versa.

References
----------
Sant'Anna, P. H. C., & Zhao, J. (2020).
    Doubly robust difference-in-differences estimators.
    *Journal of Econometrics*, 219(1), 101–122.

Graham, B. S., Pinto, C. C., & Egel, D. (2012).
    Inverse probability tilting for moment condition models with missing data.
    *Review of Economic Studies*, 79(3), 1053–1079.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import optimize
from scipy.special import expit

from ._inference import (
    run_cluster_bootstrap,
    run_multiplier_bootstrap,
    se_from_influence,
    stacked_influence,
)
from ._panel_utils import (
    BasePeriod,
    build_local_sample,
    check_columns,
    coerce_base_period,
    infer_windows,
    p_value_two_sided,
    prepare_panel,
    precompute_ccs,
    z_crit,
)
from ._results import DRLPDIDResults


# ---------------------------------------------------------------------------
# Data container for fitted propensity-score models
# ---------------------------------------------------------------------------

@dataclass
class LocalPSResult:
    """Container for a locally fitted propensity-score model.

    Attributes
    ----------
    params : np.ndarray
        Estimated parameter vector (gamma_hat).
    exog : np.ndarray
        Design matrix used in estimation.
    design_info : object
        Patsy ``DesignInfo`` object for out-of-sample prediction.
    formula : str
        Patsy formula string.
    method : str
        ``'logit'`` or ``'ipt'``.
    success : bool
        Whether the optimizer converged.
    optimizer_result : object, optional
        Raw result from ``scipy.optimize.minimize``.
    """

    params: np.ndarray
    exog: np.ndarray
    design_info: object
    formula: str
    method: str
    success: bool
    optimizer_result: object = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _control_columns(
    covariates: Optional[List[str]],
    include_ldy: bool,
    n_lags: int,
) -> List[str]:
    """Collect the list of right-hand-side control column names."""
    cols: List[str] = []
    if include_ldy and n_lags > 0:
        cols.extend([f"ldy{k}" for k in range(1, int(n_lags) + 1)])
    if covariates:
        cols.extend(list(covariates))
    return cols


def _build_formula(
    lhs: str,
    time: str,
    controls: List[str],
    add_time_fe: bool = True,
    time_varying_slopes: bool = False,
) -> str:
    """Construct a patsy formula for nuisance model estimation.

    When ``time_varying_slopes=True`` and time fixed effects are included,
    user-supplied controls enter interacted with calendar time. This yields a
    saturated local propensity-score specification of the form
    ``C(time) + C(time):X`` and imposes IPT balance within calendar-time cells
    of the horizon-specific LP-DiD stack.
    """
    controls = list(controls or [])
    rhs: List[str] = []

    if add_time_fe:
        rhs.append(f"C({time})")

    if controls:
        if add_time_fe and time_varying_slopes:
            rhs.extend([f"C({time}):{c}" for c in controls])
        else:
            rhs.extend(controls)

    if not rhs:
        return f"{lhs} ~ 1"
    return lhs + " ~ " + " + ".join(rhs)


def _restrict_to_treated_time_cells(
    local_sample: pd.DataFrame,
    time: str,
) -> pd.DataFrame:
    """Keep only time cells with both treated entrants and clean controls.

    With saturated time-by-covariate propensity-score models, time cells that
    contain only controls have no treated covariate moments to balance and can
    create separation-like IPT problems. They also do not contribute to a
    treated-entry ATT contrast. This restriction is therefore applied only to
    propensity-score based estimators when ``ps_time_varying_slopes=True``.
    """
    if local_sample.empty:
        return local_sample.copy()

    cell_stats = local_sample.groupby(time)["D_local"].agg(["sum", "count"])
    valid_times = cell_stats.index[
        (cell_stats["sum"] > 0) & (cell_stats["sum"] < cell_stats["count"])
    ]

    out = local_sample.loc[local_sample[time].isin(valid_times)].copy()
    if out.empty:
        raise ValueError(
            "No calendar-time cell contains both treated entrants and clean "
            "controls for propensity-score estimation."
        )
    return out.reset_index(drop=True)


def _odds_from_prob(
    p: np.ndarray,
    clip: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Convert probabilities to odds ratios, with optional clipping."""
    p = np.asarray(p, dtype=float)
    if clip is not None:
        p = np.clip(p, clip[0], clip[1])
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return p / (1.0 - p)


def _clip_linear_index(xg: np.ndarray, lower: float = -50.0, upper: float = 50.0) -> np.ndarray:
    """Numerically stabilize linear indices inside IPT moment functions."""
    return np.clip(np.asarray(xg, dtype=float), lower, upper)


def _build_exog_from_fit(fit, data: pd.DataFrame) -> np.ndarray:
    """Re-evaluate a patsy design matrix on new data using a fitted model."""
    return np.asarray(
        patsy.build_design_matrices(
            [fit.model.data.design_info], data, return_type="dataframe"
        )[0],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Propensity-score estimation
# ---------------------------------------------------------------------------

def _fit_ps_logit(
    local_sample: pd.DataFrame,
    time: str,
    controls: List[str],
    ps_clip: Tuple[float, float],
    add_time_fe: bool = False,
    time_varying_slopes: bool = False,
) -> tuple:
    """Fit a local propensity score via maximum-likelihood logistic regression.

    Returns
    -------
    fit : GLMResultsWrapper or None
        Fitted model object (``None`` when no covariates are present).
    e_hat : np.ndarray
        Clipped propensity-score predictions.
    """
    formula = _build_formula(
        "D_local",
        time,
        controls,
        add_time_fe=add_time_fe,
        time_varying_slopes=time_varying_slopes,
    )
    if formula == "D_local ~ 1":
        p = float(local_sample["D_local"].mean())
        p_clipped = np.clip(p, ps_clip[0], ps_clip[1]) if ps_clip is not None else p
        e_hat = np.full(len(local_sample), p_clipped, dtype=float)
        return None, e_hat
    fit = smf.glm(
        formula, data=local_sample, family=sm.families.Binomial()
    ).fit(disp=False)
    raw_e = np.asarray(fit.predict(local_sample), dtype=float)
    e_hat = np.clip(raw_e, ps_clip[0], ps_clip[1]) if ps_clip is not None else raw_e
    return fit, e_hat


def _fit_ps_ipt(
    local_sample: pd.DataFrame,
    time: str,
    controls: List[str],
    ps_clip: Tuple[float, float],
    add_time_fe: bool = False,
    time_varying_slopes: bool = False,
) -> tuple:
    """Fit a local propensity score via inverse probability tilting (IPT).

    IPT minimises the empirical Kullback–Leibler divergence between the
    treated and reweighted control distributions, yielding a first-order
    efficient moment-condition estimator (Graham et al. 2012).

    Returns
    -------
    fit : LocalPSResult
        Fitted IPT model container.
    e_hat : np.ndarray
        Clipped propensity-score predictions.
    """
    formula = _build_formula(
        "D_local",
        time,
        controls,
        add_time_fe=add_time_fe,
        time_varying_slopes=time_varying_slopes,
    )

    if formula == "D_local ~ 1":
        d = np.asarray(local_sample["D_local"], dtype=float)
        p = float(np.mean(d))
        if ps_clip is not None:
            p = np.clip(p, ps_clip[0], ps_clip[1])
        else:
            p = np.clip(p, 1e-10, 1.0 - 1e-10)
        fit = LocalPSResult(
            params=np.array([np.log(p / (1.0 - p))]),
            exog=np.ones((len(local_sample), 1), dtype=float),
            design_info=None,
            formula=formula,
            method="ipt",
            success=True,
        )
        return fit, np.full(len(local_sample), p, dtype=float)

    y_mat, X_df = patsy.dmatrices(formula, local_sample, return_type="dataframe")
    d = np.asarray(y_mat).reshape(-1).astype(float)
    X = np.asarray(X_df, dtype=float)

    # Warm-start from logistic regression; fall back to intercept-only on failure
    pbar = np.clip(np.mean(d), ps_clip[0] if ps_clip else 1e-6, ps_clip[1] if ps_clip else 1-1e-6)
    x0 = np.zeros(X.shape[1], dtype=float)
    x0[0] = np.log(pbar / (1.0 - pbar))
    try:
        glm_fit = smf.glm(
            formula=formula, data=local_sample, family=sm.families.Binomial()
        ).fit(disp=False)
        if np.all(np.isfinite(glm_fit.params)):
            x0 = np.asarray(glm_fit.params, dtype=float)
    except Exception:
        pass

    def obj(gamma: np.ndarray) -> float:
        eta = _clip_linear_index(X @ gamma)
        return -float(np.mean(d * eta - (1.0 - d) * np.exp(eta)))

    def grad(gamma: np.ndarray) -> np.ndarray:
        eta = _clip_linear_index(X @ gamma)
        score = X * (d - (1.0 - d) * np.exp(eta))[:, None]
        return -np.mean(score, axis=0)

    opt = optimize.minimize(obj, x0=x0, jac=grad, method="BFGS")
    if not opt.success or not np.all(np.isfinite(opt.x)):
        opt = optimize.minimize(obj, x0=x0, jac=grad, method="L-BFGS-B")
    if not opt.success or not np.all(np.isfinite(opt.x)):
        raise RuntimeError(
            f"Local IPT propensity estimation failed: {opt.message}"
        )

    gamma_hat = np.asarray(opt.x, dtype=float)
    # For IPT the natural weight is exp(gamma'X); we expose expit for
    # overlap diagnostics only. When ps_clip is None (default), return
    # exp(xg) directly as the effective "propensity odds" so that the
    # point estimate and the influence function use the same quantity.
    raw_e = expit(_clip_linear_index(X @ gamma_hat))
    e_hat = np.clip(raw_e, ps_clip[0], ps_clip[1]) if ps_clip is not None else raw_e
    fit = LocalPSResult(
        params=gamma_hat,
        exog=X,
        design_info=X_df.design_info,
        formula=formula,
        method="ipt",
        success=bool(opt.success),
        optimizer_result=opt,
    )
    return fit, e_hat


# ---------------------------------------------------------------------------
# Outcome regression
# ---------------------------------------------------------------------------

def _fit_or_ols_controls(
    local_sample: pd.DataFrame,
    time: str,
    controls: List[str],
    add_time_fe: bool = True,
) -> tuple:
    """OLS outcome regression on clean-control observations.

    Returns ``(fitted_model, predicted_values_for_full_sample)``.
    """
    ctrl = local_sample.loc[local_sample["D_local"] == 0].copy()
    if ctrl.empty:
        raise ValueError("No clean controls available for outcome regression.")
    formula = _build_formula("outcome_local", time, controls, add_time_fe=add_time_fe)
    fit = smf.ols(formula, data=ctrl).fit()
    pred = np.asarray(fit.predict(local_sample), dtype=float)
    return fit, pred


def _fit_or_wls_controls(
    local_sample: pd.DataFrame,
    time: str,
    controls: List[str],
    weights: np.ndarray,
    add_time_fe: bool = True,
) -> tuple:
    """IPT-weighted least-squares outcome regression on clean-control observations.

    Used by ``dr_method='improved'`` to match the weighted covariate distribution
    of the treated group in the Sant'Anna--Zhao-style improved DR specification.
    """
    ctrl = local_sample.loc[local_sample["D_local"] == 0].copy()
    if ctrl.empty:
        raise ValueError("No clean controls available for outcome regression.")
    formula = _build_formula("outcome_local", time, controls, add_time_fe=add_time_fe)
    ctrl["_w"] = np.asarray(weights, dtype=float)[local_sample["D_local"].to_numpy() == 0]
    fit = smf.wls(formula, data=ctrl, weights=ctrl["_w"]).fit()
    pred = np.asarray(fit.predict(local_sample), dtype=float)
    return fit, pred


# ---------------------------------------------------------------------------
# Semiparametric estimators
# ---------------------------------------------------------------------------

def _compute_ra(
    local_sample: pd.DataFrame,
    unit: str,
    time: str,
    controls: List[str],
    compute_influence: bool,
) -> dict:
    """Regression-adjustment ATT-h estimator.

    .. math::
        \\hat{\\tau}_h^{RA} = \\frac{1}{N_1}
            \\sum_{i: D_i=1} \\bigl(Y_i - \\hat{m}_0(X_i)\\bigr)

    where :math:`\\hat{m}_0` is estimated on clean controls only.
    """
    y = np.asarray(local_sample["outcome_local"], dtype=float)
    d = np.asarray(local_sample["D_local"], dtype=float)
    cluster_ids = local_sample[unit].to_numpy()
    fit_or, m0_hat = _fit_or_ols_controls(local_sample, time, controls)
    z_all = _build_exog_from_fit(fit_or, local_sample)
    beta_hat = np.asarray(fit_or.params, dtype=float)
    resid = y - m0_hat
    mu1_hat = float(np.mean(resid[d == 1]))

    if not compute_influence:
        return {
            "estimate": mu1_hat, "se": np.nan,
            "n_treat": int((d == 1).sum()), "n_ctrl": int((d == 0).sum()),
        }

    theta_hat = np.concatenate([beta_hat, np.array([mu1_hat])])

    def moments_obs(theta):
        pb = z_all.shape[1]
        beta, mu1 = theta[:pb], theta[pb]
        resid_theta = y - z_all @ beta
        m_beta = ((1.0 - d))[:, None] * z_all * resid_theta[:, None]
        m_mu1 = (d * (resid_theta - mu1))[:, None]
        return np.column_stack([m_beta, m_mu1])

    def target_grad(theta):
        g = np.zeros_like(theta)
        g[-1] = 1.0
        return g

    infl = stacked_influence(
        theta_hat, moments_obs, cluster_ids, target_grad=target_grad
    )
    psi = infl["psi"].reshape(-1)
    return {
        "estimate": mu1_hat,
        "se": se_from_influence(psi),
        "psi_by_cluster": pd.Series(
            psi, index=pd.Index(infl["cluster_labels"]), dtype=float
        ),
        "n_treat": int((d == 1).sum()),
        "n_ctrl": int((d == 0).sum()),
    }


def _compute_ipw(
    local_sample: pd.DataFrame,
    unit: str,
    time: str,
    controls: List[str],
    ps_method: str,
    ps_clip: Tuple[float, float],
    ps_with_time_fe: bool,
    ps_time_varying_slopes: bool,
    compute_influence: bool,
) -> dict:
    """IPW ATT-h estimator.

    .. math::
        \\hat{\\tau}_h^{IPW} = \\bar{Y}_1
            - \\frac{\\sum_{i:D_i=0} o_i Y_i}{\\sum_{i:D_i=0} o_i}

    where :math:`o_i = e(X_i) / (1 - e(X_i))` are propensity-score odds.
    The propensity score is estimated by logistic regression (``'generic'``)
    or IPT (``'improved'``).
    """
    if ps_time_varying_slopes:
        local_sample = _restrict_to_treated_time_cells(local_sample, time)

    y = np.asarray(local_sample["outcome_local"], dtype=float)
    d = np.asarray(local_sample["D_local"], dtype=float)
    cluster_ids = local_sample[unit].to_numpy()

    if ps_method == "improved":
        fit_ps, e_hat = _fit_ps_ipt(
            local_sample,
            time,
            controls,
            ps_clip,
            add_time_fe=ps_with_time_fe,
            time_varying_slopes=ps_time_varying_slopes,
        )
        X = np.asarray(fit_ps.exog, dtype=float)
        gamma_hat = np.asarray(fit_ps.params, dtype=float)
        score_moment = "ipt"
    else:
        fit_ps, e_hat = _fit_ps_logit(
            local_sample,
            time,
            controls,
            ps_clip,
            add_time_fe=ps_with_time_fe,
            time_varying_slopes=ps_time_varying_slopes,
        )
        if fit_ps is None:
            X = np.ones((len(local_sample), 1), dtype=float)
            pbar = np.clip(np.mean(d), ps_clip[0] if ps_clip else 1e-6, ps_clip[1] if ps_clip else 1-1e-6)
            gamma_hat = np.array([np.log(pbar / (1.0 - pbar))])
        else:
            X = np.asarray(fit_ps.model.exog, dtype=float)
            gamma_hat = np.asarray(fit_ps.params, dtype=float)
        score_moment = "logit"

    odds = _odds_from_prob(e_hat)
    mu1_hat = float(np.mean(y[d == 1]))
    lam = odds[d == 0]
    lam_sum = float(lam.sum())
    if lam_sum <= 1e-12:
        raise ValueError(
            "IPW control weight sum is effectively zero; "
            "local overlap is insufficient at this horizon."
        )
    mu0_hat = float(np.dot(lam, y[d == 0]) / lam_sum)
    tau_hat = mu1_hat - mu0_hat

    overlap = {
        "min": float(np.min(e_hat)),
        "p5": float(np.quantile(e_hat, 0.05)),
        "p50": float(np.quantile(e_hat, 0.5)),
        "p95": float(np.quantile(e_hat, 0.95)),
        "max": float(np.max(e_hat)),
    }

    if not compute_influence:
        return {
            "estimate": tau_hat, "se": np.nan,
            "overlap_stats": overlap,
            "n_treat": int((d == 1).sum()), "n_ctrl": int((d == 0).sum()),
        }

    theta_hat = np.concatenate([gamma_hat, np.array([mu1_hat, mu0_hat])])

    def moments_obs(theta):
        pg = X.shape[1]
        gamma, mu1, mu0 = theta[:pg], theta[pg], theta[pg + 1]
        xg = _clip_linear_index(X @ gamma)
        pihat = (np.clip(expit(xg), ps_clip[0], ps_clip[1])
                 if ps_clip is not None else expit(xg))
        if score_moment == "ipt":
            # IPT estimating equation uses exp(gamma'X) as the tilting odds.
            # The ATT component uses the same effective odds as the point
            # estimate; with ps_clip=None this is exactly exp(gamma'X).
            ipt_odds_score = np.exp(xg)
            odds_theta = _odds_from_prob(expit(xg), clip=ps_clip)
            m_gamma = X * (d - (1.0 - d) * ipt_odds_score)[:, None]
            m_mu1 = (d * (y - mu1))[:, None]
            m_mu0 = ((1.0 - d) * odds_theta * (y - mu0))[:, None]
        else:
            m_gamma = X * (d - pihat)[:, None]
            odds_theta = _odds_from_prob(pihat, clip=ps_clip)
            m_mu1 = (d * (y - mu1))[:, None]
            m_mu0 = ((1.0 - d) * odds_theta * (y - mu0))[:, None]
        return np.column_stack([m_gamma, m_mu1, m_mu0])

    def target_grad(theta):
        g = np.zeros_like(theta)
        g[-2] = 1.0
        g[-1] = -1.0
        return g

    infl = stacked_influence(
        theta_hat, moments_obs, cluster_ids, target_grad=target_grad
    )
    psi = infl["psi"].reshape(-1)
    return {
        "estimate": tau_hat,
        "se": se_from_influence(psi),
        "psi_by_cluster": pd.Series(
            psi, index=pd.Index(infl["cluster_labels"]), dtype=float
        ),
        "overlap_stats": overlap,
        "n_treat": int((d == 1).sum()),
        "n_ctrl": int((d == 0).sum()),
    }


def _compute_dr_generic(
    local_sample: pd.DataFrame,
    unit: str,
    time: str,
    ps_controls: List[str],
    or_controls: List[str],
    ps_clip: Tuple[float, float],
    ps_with_time_fe: bool,
    ps_time_varying_slopes: bool,
    compute_influence: bool,
) -> dict:
    """Doubly robust ATT-h estimator (generic logit + OLS specification).

    The DR correction applies IPW reweighting to the outcome-regression
    residuals, yielding consistency when *either* the propensity score *or*
    the outcome regression is correctly specified.

    .. math::
        \\hat{\\tau}_h^{DR} = \\frac{1}{N_1}\\sum_{i:D_i=1}(Y_i - \\hat{m}_0(X_i))
            - \\frac{\\sum_{i:D_i=0} o_i (Y_i - \\hat{m}_0(X_i))}{\\sum_{i:D_i=0} o_i}
    """
    if ps_time_varying_slopes:
        local_sample = _restrict_to_treated_time_cells(local_sample, time)

    y = np.asarray(local_sample["outcome_local"], dtype=float)
    d = np.asarray(local_sample["D_local"], dtype=float)
    cluster_ids = local_sample[unit].to_numpy()

    ps_controls = list(ps_controls or [])
    or_controls = list(or_controls or [])

    fit_ps, e_hat = _fit_ps_logit(
        local_sample,
        time,
        ps_controls,
        ps_clip,
        add_time_fe=ps_with_time_fe,
        time_varying_slopes=ps_time_varying_slopes,
    )
    fit_or, m0_hat = _fit_or_ols_controls(local_sample, time, or_controls)
    resid = y - m0_hat
    odds = _odds_from_prob(e_hat)
    mu1_hat = float(np.mean(resid[d == 1]))
    lam = odds[d == 0]
    mu0_hat = float(np.dot(lam, resid[d == 0]) / float(lam.sum()))
    tau_hat = mu1_hat - mu0_hat

    overlap = {
        "min": float(np.min(e_hat)),
        "p5": float(np.quantile(e_hat, 0.05)),
        "p50": float(np.quantile(e_hat, 0.5)),
        "p95": float(np.quantile(e_hat, 0.95)),
        "max": float(np.max(e_hat)),
    }

    if not compute_influence:
        return {
            "estimate": tau_hat, "se": np.nan,
            "overlap_stats": overlap,
            "n_treat": int((d == 1).sum()), "n_ctrl": int((d == 0).sum()),
        }

    z_all = _build_exog_from_fit(fit_or, local_sample)
    beta_hat = np.asarray(fit_or.params, dtype=float)
    if fit_ps is None:
        X = np.ones((len(local_sample), 1), dtype=float)
        pbar = np.clip(np.mean(d), 1e-6, 1 - 1e-6)
        gamma_hat = np.array([np.log(pbar / (1.0 - pbar))])
    else:
        X = np.asarray(fit_ps.model.exog, dtype=float)
        gamma_hat = np.asarray(fit_ps.params, dtype=float)
    theta_hat = np.concatenate([beta_hat, gamma_hat, np.array([mu1_hat, mu0_hat])])

    def moments_obs(theta):
        pb, pg = z_all.shape[1], X.shape[1]
        beta, gamma = theta[:pb], theta[pb:pb + pg]
        mu1, mu0 = theta[pb + pg], theta[pb + pg + 1]
        pihat = (np.clip(expit(_clip_linear_index(X @ gamma)), ps_clip[0], ps_clip[1])
                 if ps_clip is not None else expit(_clip_linear_index(X @ gamma)))
        odds_theta = _odds_from_prob(pihat, clip=ps_clip)
        resid_theta = y - z_all @ beta
        m_beta = ((1.0 - d))[:, None] * z_all * resid_theta[:, None]
        m_gamma = X * (d - pihat)[:, None]
        m_mu1 = (d * (resid_theta - mu1))[:, None]
        m_mu0 = ((1.0 - d) * odds_theta * (resid_theta - mu0))[:, None]
        return np.column_stack([m_beta, m_gamma, m_mu1, m_mu0])

    def target_grad(theta):
        g = np.zeros_like(theta)
        g[-2] = 1.0
        g[-1] = -1.0
        return g

    infl = stacked_influence(
        theta_hat, moments_obs, cluster_ids, target_grad=target_grad
    )
    psi = infl["psi"].reshape(-1)
    return {
        "estimate": tau_hat,
        "se": se_from_influence(psi),
        "psi_by_cluster": pd.Series(
            psi, index=pd.Index(infl["cluster_labels"]), dtype=float
        ),
        "overlap_stats": overlap,
        "n_treat": int((d == 1).sum()),
        "n_ctrl": int((d == 0).sum()),
    }


def _compute_dr_improved(
    local_sample: pd.DataFrame,
    unit: str,
    time: str,
    ps_controls: List[str],
    or_controls: List[str],
    ps_clip: Tuple[float, float],
    ps_with_time_fe: bool,
    ps_time_varying_slopes: bool,
    compute_influence: bool,
) -> dict:
    """Improved DR ATT-h estimator in the Sant'Anna--Zhao style.

    The improved specification uses inverse probability tilting (IPT) for the
    local propensity score and IPT-weighted least squares for the clean-control
    untreated-outcome regression. This is the only improved DR specification
    exposed by the package.
    """
    if ps_time_varying_slopes:
        local_sample = _restrict_to_treated_time_cells(local_sample, time)

    y = np.asarray(local_sample["outcome_local"], dtype=float)
    d = np.asarray(local_sample["D_local"], dtype=float)
    cluster_ids = local_sample[unit].to_numpy()
    ps_controls = list(ps_controls or [])
    or_controls = list(or_controls or [])

    # First nuisance: local IPT propensity score.
    fit_ps, e_hat = _fit_ps_ipt(
        local_sample,
        time,
        ps_controls,
        ps_clip,
        add_time_fe=ps_with_time_fe,
        time_varying_slopes=ps_time_varying_slopes,
    )
    odds = _odds_from_prob(e_hat)

    # Second nuisance: IPT-weighted outcome regression on clean controls.
    fit_or, m0_hat = _fit_or_wls_controls(
        local_sample,
        time,
        or_controls,
        weights=odds,
        add_time_fe=True,
    )

    resid = y - m0_hat
    mu1_hat = float(np.mean(resid[d == 1]))
    lam = odds[d == 0]
    lam_sum = float(lam.sum())
    if lam_sum <= 1e-12:
        raise ValueError(
            "Improved DR control residual weight sum is effectively zero; "
            "local overlap is insufficient at this horizon."
        )
    mu0_hat = float(np.dot(lam, resid[d == 0]) / lam_sum)
    tau_hat = mu1_hat - mu0_hat

    overlap = {
        "min": float(np.min(e_hat)),
        "p5": float(np.quantile(e_hat, 0.05)),
        "p50": float(np.quantile(e_hat, 0.5)),
        "p95": float(np.quantile(e_hat, 0.95)),
        "max": float(np.max(e_hat)),
    }

    if not compute_influence:
        return {
            "estimate": tau_hat, "se": np.nan,
            "overlap_stats": overlap,
            "n_treat": int((d == 1).sum()), "n_ctrl": int((d == 0).sum()),
        }

    z_all = _build_exog_from_fit(fit_or, local_sample)
    beta_hat = np.asarray(fit_or.params, dtype=float)
    X = np.asarray(fit_ps.exog, dtype=float)
    gamma_hat = np.asarray(fit_ps.params, dtype=float)
    theta_hat = np.concatenate([beta_hat, gamma_hat, np.array([mu1_hat, mu0_hat])])

    def moments_obs(theta):
        pb, pg = z_all.shape[1], X.shape[1]
        beta, gamma = theta[:pb], theta[pb:pb + pg]
        mu1, mu0 = theta[pb + pg], theta[pb + pg + 1]
        xg = _clip_linear_index(X @ gamma)
        resid_theta = y - z_all @ beta

        # IPT weighted least-squares score on clean controls.
        ipt_odds_score = np.exp(xg)
        odds_theta = _odds_from_prob(expit(xg), clip=ps_clip)
        m_beta = (((1.0 - d) * odds_theta)[:, None]) * z_all * resid_theta[:, None]

        # IPT propensity-score moment.
        m_gamma = X * (d - (1.0 - d) * ipt_odds_score)[:, None]
        m_mu1 = (d * (resid_theta - mu1))[:, None]
        m_mu0 = ((1.0 - d) * odds_theta * (resid_theta - mu0))[:, None]
        return np.column_stack([m_beta, m_gamma, m_mu1, m_mu0])

    def target_grad(theta):
        g = np.zeros_like(theta)
        g[-2] = 1.0
        g[-1] = -1.0
        return g

    infl = stacked_influence(
        theta_hat, moments_obs, cluster_ids, target_grad=target_grad
    )
    psi = infl["psi"].reshape(-1)
    return {
        "estimate": tau_hat,
        "se": se_from_influence(psi),
        "psi_by_cluster": pd.Series(
            psi, index=pd.Index(infl["cluster_labels"]), dtype=float
        ),
        "overlap_stats": overlap,
        "n_treat": int((d == 1).sum()),
        "n_ctrl": int((d == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Main estimator class
# ---------------------------------------------------------------------------

class DRLPDID:
    """Semiparametric DR-LP-DiD estimator.

    Applies regression-adjustment (RA), inverse probability weighting (IPW),
    or doubly robust (DR) corrections to the LP-DiD horizon-h clean local
    comparison stacks, then assembles an event-study path with valid
    simultaneous confidence bands via the multiplier bootstrap.

    Parameters
    ----------
    estimation_method : {'ra', 'ipw', 'dr'}
        Nuisance-model combination:

        * ``'ra'``: outcome regression only (consistent if OR is correct);
        * ``'ipw'``: propensity-score reweighting only (consistent if PS
          is correct);
        * ``'dr'``: doubly robust (consistent if *either* model is correct).
    dr_method : {'generic', 'improved'}
        Specification for IPW and DR estimators. ``'generic'`` uses logistic
        propensity scores plus OLS outcome regression. ``'improved'`` follows
        the Sant'Anna--Zhao improved DR-DiD logic: IPT propensity scores plus
        IPT-weighted least-squares outcome regression on clean controls.
        Ignored when ``estimation_method='ra'``.
    base_period : int, list of int, or 'all_pre'
        Baseline for long differences (see :class:`~lpdid.LPDID`).
    clean_control : {'not_yet_treated', 'never_treated', 'stabilized'}
        Control group restriction.
    effect_stabilization : int or None
        Effect-stabilization horizon L (required for ``'stabilized'``).
    anticipation : int
        Periods before treatment excluded from pre-trend estimation.
    include_lagged_outcome_change : bool
        Include lagged first-differences of the outcome as controls.
    n_lagged_outcome_changes : int
        Number of lagged outcome changes (ignored if
        ``include_lagged_outcome_change=False``).
    ps_with_time_fe : bool
        Include time fixed effects in the propensity-score model (default
        ``True``).  Setting to ``False`` uses a pooled logit/IPT model.
    ps_time_varying_slopes : bool
        If ``True``, interact user-supplied controls with calendar time in the
        propensity-score model, yielding ``C(time) + C(time):X``. This is useful
        for staggered-adoption LP-DiD stacks because the local odds of entry
        may vary with covariates differently across calendar-time cells.
        Default ``False`` to preserve backward compatibility.
    inference : {'cluster', 'cluster_bootstrap', 'multiplier'}
        Inference method:

        * ``'cluster'``: analytic cluster-robust SE via influence functions;
        * ``'multiplier'``: multiplier/wild bootstrap with simultaneous
          sup-t confidence bands (default; recommended for event studies);
        * ``'cluster_bootstrap'``: paired cluster bootstrap.
    n_bootstrap : int
        Bootstrap replications (default 999).
    bootstrap_weights : str
        Weight distribution for the multiplier bootstrap.  One of
        ``'rademacher'`` (default), ``'mammen'``, or ``'webb'``.
    alpha : float
        Significance level (default 0.05).
    seed : int or None
        Random seed for reproducibility.
    ps_clip : tuple of float or None
        Propensity-score trimming bounds, e.g. ``(0.025, 0.975)``.
        Default ``None`` (disabled). When ``None``, IPT uses ``exp(gamma'X)``
        directly as weights — fully consistent with the theoretical IPT
        estimating equations (Graham et al. 2012). Enabling clipping
        regularises extreme scores in low-overlap settings but introduces
        a minor inconsistency between the point estimate and the influence
        function for logit-based methods when scores are clipped.
    max_pre, max_post : int or None
        Maximum pre- and post-treatment horizons.  Inferred from data when
        ``None``.
    lag_covariates : bool
        If ``True`` (recommended when covariates are time-varying), all
        user-supplied covariates are automatically lagged by one period
        (:math:`X_{i,t-1}`) inside each local comparison stack.
        Ensures covariates are predetermined for both the propensity-score
        and outcome-regression models, as required by the conditional
        parallel trends assumption (Dube et al. 2025, Section 4.1).
        Lagged-outcome-change controls (``ldy*``) are **not** re-lagged.
        Default ``False`` to preserve backward compatibility.

    Examples
    --------
    >>> from lpdid import DRLPDID
    >>> res = DRLPDID(
    ...     estimation_method='dr', dr_method='generic',
    ...     inference='multiplier', n_bootstrap=999,
    ... ).fit(data=df, outcome='y', unit='state', time='year',
    ...       first_treat='g', covariates=['x1', 'x2'])
    >>> res.print_summary()
    """

    def __init__(
        self,
        *,
        estimation_method: str = "dr",
        dr_method: str = "improved",
        base_period: BasePeriod = -1,
        clean_control: str = "not_yet_treated",
        effect_stabilization: Optional[int] = None,
        anticipation: int = 0,
        include_lagged_outcome_change: bool = False,
        n_lagged_outcome_changes: int = 0,
        ps_with_time_fe: bool = True,
        ps_time_varying_slopes: bool = False,
        inference: str = "multiplier",
        n_bootstrap: int = 999,
        bootstrap_weights: str = "rademacher",
        alpha: float = 0.05,
        seed: Optional[int] = None,
        ps_clip: Optional[Tuple[float, float]] = None,
        max_pre: Optional[int] = None,
        max_post: Optional[int] = None,
        lag_covariates: bool = False,
    ) -> None:
        self.estimation_method = str(estimation_method).lower()
        if self.estimation_method not in {"ra", "ipw", "dr"}:
            raise ValueError("estimation_method must be one of {'ra', 'ipw', 'dr'}.")
        self.dr_method = str(dr_method).lower()
        if self.dr_method not in {"generic", "improved"}:
            raise ValueError(
                "dr_method must be one of {'generic', 'improved'}."
            )
        self.base_period = coerce_base_period(base_period)
        self.clean_control = str(clean_control).lower()
        if self.clean_control not in {
            "not_yet_treated", "never_treated",
            "stabilized",
        }:
            raise ValueError(
                "clean_control must be one of {'not_yet_treated', 'never_treated', "
                "'stabilized'}."
            )
        self.effect_stabilization = effect_stabilization
        if (
            self.clean_control == "stabilized"
            and self.effect_stabilization is None
        ):
            raise ValueError(
                "clean_control='stabilized' requires "
                "effect_stabilization to be set."
            )
        self.anticipation = int(max(anticipation, 0))
        self.include_lagged_outcome_change = bool(include_lagged_outcome_change)
        self.n_lagged_outcome_changes = int(max(n_lagged_outcome_changes, 0))
        self.ps_with_time_fe = bool(ps_with_time_fe)
        self.ps_time_varying_slopes = bool(ps_time_varying_slopes)
        self.inference = str(inference).lower()
        if self.inference not in {"cluster", "cluster_bootstrap", "multiplier"}:
            raise ValueError(
                "inference must be one of {'cluster', 'cluster_bootstrap', 'multiplier'}."
            )
        self.n_bootstrap = int(n_bootstrap)
        self.bootstrap_weights = str(bootstrap_weights).lower()
        self.alpha = float(alpha)
        self.seed = seed
        self.ps_clip = tuple(ps_clip) if ps_clip is not None else None
        self.max_pre = None if max_pre is None else int(max_pre)
        self.max_post = None if max_post is None else int(max_post)
        self.lag_covariates = bool(lag_covariates)

    # ------------------------------------------------------------------
    # Dispatch to semiparametric estimator
    # ------------------------------------------------------------------

    def _fit_one(
        self,
        local_sample: pd.DataFrame,
        unit: str,
        time: str,
        ps_controls: List[str],
        or_controls: List[str],
        compute_influence: bool,
    ) -> dict:
        """Dispatch to the appropriate semiparametric estimator."""
        if self.estimation_method == "ra":
            return _compute_ra(local_sample, unit, time, or_controls, compute_influence)
        if self.estimation_method == "ipw":
            return _compute_ipw(
                local_sample,
                unit,
                time,
                ps_controls,
                self.dr_method,
                self.ps_clip,
                self.ps_with_time_fe,
                self.ps_time_varying_slopes,
                compute_influence,
            )
        # DR
        if self.dr_method == "generic":
            return _compute_dr_generic(
                local_sample,
                unit,
                time,
                ps_controls,
                or_controls,
                self.ps_clip,
                self.ps_with_time_fe,
                self.ps_time_varying_slopes,
                compute_influence,
            )
        return _compute_dr_improved(
            local_sample,
            unit,
            time,
            ps_controls,
            or_controls,
            self.ps_clip,
            self.ps_with_time_fe,
            self.ps_time_varying_slopes,
            compute_influence,
        )

    # ------------------------------------------------------------------
    # Core estimation loop
    # ------------------------------------------------------------------

    def _fit_core(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: Optional[str],
        treatment: Optional[str],
        covariates: Optional[List[str]],
        ps_covariates: Optional[List[str]] = None,
        or_covariates: Optional[List[str]] = None,
        compute_influence: bool = True,
    ) -> dict:
        base_covariates = list(covariates or [])
        ps_covariates = base_covariates if ps_covariates is None else list(ps_covariates or [])
        or_covariates = base_covariates if or_covariates is None else list(or_covariates or [])

        all_covariates = list(dict.fromkeys(base_covariates + ps_covariates + or_covariates))
        if all_covariates:
            check_columns(data, all_covariates)
        df = prepare_panel(
            data, outcome, unit, time, first_treat, treatment,
            nonabsorbing=(self.clean_control == "stabilized"),
            n_lagged_outcome_changes=(
                self.n_lagged_outcome_changes
                if self.include_lagged_outcome_change
                else 0
            ),
        )
        max_pre, max_post = infer_windows(
            df, time, self.base_period, self.max_pre, self.max_post
        )
        if self.effect_stabilization is not None:
            df = precompute_ccs(
                df, unit, int(self.effect_stabilization), max_pre, max_post
            )

        ps_controls = _control_columns(
            ps_covariates,
            self.include_lagged_outcome_change,
            self.n_lagged_outcome_changes,
        )
        or_controls = _control_columns(
            or_covariates,
            self.include_lagged_outcome_change,
            self.n_lagged_outcome_changes,
        )
        stack_controls = list(dict.fromkeys(ps_controls + or_controls))
        stack_user_covariates = list(dict.fromkeys(ps_covariates + or_covariates))
        z = z_crit(self.alpha)
        cluster_universe = pd.Index(df[unit].dropna().unique())
        psi_by_h: Dict[int, pd.Series] = {}
        rows: list[dict] = []
        overlap_metadata: dict = {}

        pre_horizons = list(range(-max_pre, -self.anticipation))
        post_horizons = list(range(0, max_post + 1))
        if (
            isinstance(self.base_period, int)
            and self.base_period == -1
            and -1 in pre_horizons
        ):
            pre_horizons.remove(-1)

        for h in pre_horizons + post_horizons:
            local = build_local_sample(
                df, outcome, unit, time, h, self.base_period,
                self.clean_control, self.effect_stabilization, stack_controls,
                user_covariates=stack_user_covariates,
                lag_covariates=self.lag_covariates,
            )
            if (
                local.empty
                or local["D_local"].sum() == 0
                or (local["D_local"] == 0).sum() == 0
            ):
                rows.append(_nan_row(h))
                psi_by_h[int(h)] = pd.Series(0.0, index=cluster_universe)
                continue

            try:
                fit = self._fit_one(
                    local, unit, time, ps_controls, or_controls, compute_influence
                )
            except Exception as exc:
                warnings.warn(
                    f"Horizon {h}: estimation failed — {type(exc).__name__}: {exc}",
                    stacklevel=2,
                )
                rows.append(_nan_row(h))
                psi_by_h[int(h)] = pd.Series(0.0, index=cluster_universe)
                continue

            est, se = fit["estimate"], fit["se"]
            psi_by_h[int(h)] = fit.get(
                "psi_by_cluster", pd.Series(0.0, index=cluster_universe)
            ).reindex(cluster_universe, fill_value=0.0)
            if fit.get("overlap_stats"):
                overlap_metadata[int(h)] = fit["overlap_stats"]
            rows.append(_make_row(h, est, se, z))

        # Normalized base-period row
        if isinstance(self.base_period, int) and self.base_period == -1:
            rows.append({
                "horizon": -1, "estimate": 0.0, "se": 0.0,
                "t_stat": np.nan, "p_value": np.nan,
                "ci_lower": 0.0, "ci_upper": 0.0,
            })
            psi_by_h[-1] = pd.Series(0.0, index=cluster_universe)

        event_df = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)

        # Scalar summaries
        scalar_rows: list[dict] = []
        post = event_df.loc[
            (event_df["horizon"] >= 0) & np.isfinite(event_df["estimate"])
        ].copy()
        if not post.empty:
            post_h = [int(h) for h in post["horizon"].tolist()]
            att_avg = float(np.mean(post["estimate"].to_numpy(dtype=float)))
            se_avg = np.nan
            if compute_influence:
                Psi = np.column_stack([
                    psi_by_h[h].reindex(cluster_universe, fill_value=0.0).to_numpy(dtype=float)
                    for h in post_h
                ])
                se_avg = se_from_influence(np.mean(Psi, axis=1))
            t_avg = att_avg / se_avg if np.isfinite(se_avg) and se_avg > 0 else np.nan
            scalar_rows.append({
                "term": "ATT avg",
                "estimate": att_avg,
                "se": se_avg,
                "t_stat": t_avg,
                "p_value": p_value_two_sided(t_avg) if np.isfinite(t_avg) else np.nan,
                "ci_lower": att_avg - z * se_avg if np.isfinite(se_avg) else np.nan,
                "ci_upper": att_avg + z * se_avg if np.isfinite(se_avg) else np.nan,
            })

        notes: list[str] = []
        if self.estimation_method == "dr" and self.dr_method == "improved":
            notes.append(
                "Improved DR follows the Sant'Anna--Zhao style: IPT for the local "
                "propensity score and IPT-weighted least-squares on clean controls "
                "for the untreated-outcome regression. If ps_covariates and "
                "or_covariates differ, the two nuisance bases are estimated separately."
            )
        if self.estimation_method == "ipw" and self.dr_method == "improved":
            notes.append(
                "IPW with dr_method='improved' uses the IPT first step."
            )

        metadata: dict = {
            "overlap": overlap_metadata,
            "ps_time_varying_slopes": self.ps_time_varying_slopes,
            "covariates": base_covariates,
            "ps_covariates": ps_covariates,
            "or_covariates": or_covariates,
            "ps_controls": ps_controls,
            "or_controls": or_controls,
        }
        if notes:
            metadata["notes"] = notes

        return {
            "panel": df,
            "event_study": event_df,
            "scalars": pd.DataFrame(scalar_rows),
            "psi_by_h": psi_by_h,
            "cluster_universe": cluster_universe,
            "n_obs": int(df.shape[0]),
            "n_treated_units": int(df.loc[df["_treated_ever"] == 1, unit].nunique()),
            "n_control_units": int(df.loc[df["_never_treated"] == 1, unit].nunique()),
            "n_cohorts": int(df.loc[df["_first_treat"] > 0, "_first_treat"].nunique()),
            "n_periods": int(df[time].nunique()),
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Public fit method
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: Optional[str] = None,
        treatment: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        ps_covariates: Optional[List[str]] = None,
        or_covariates: Optional[List[str]] = None,
    ) -> DRLPDIDResults:
        """Fit the DR-LP-DiD estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Long-format panel dataset.
        outcome : str
            Outcome variable column name.
        unit : str
            Unit identifier column name.
        time : str
            Time identifier column name.
        first_treat : str, optional
            First-treatment period column (0 for never-treated).
        treatment : str, optional
            Binary treatment indicator column.
        covariates : list of str, optional
            Backward-compatible covariate columns used in both nuisance models
            unless ``ps_covariates`` or ``or_covariates`` are supplied.
        ps_covariates : list of str, optional
            Covariate columns used only in the propensity-score/IPT equation.
            Defaults to ``covariates``.
        or_covariates : list of str, optional
            Covariate columns used only in the untreated-outcome regression.
            Defaults to ``covariates``.

        Returns
        -------
        DRLPDIDResults
            Result object with ``event_study`` (including simultaneous CIs
            when ``inference='multiplier'``), ``scalars``, and
            ``print_summary()``.
        """
        compute_influence = self.inference in {"cluster", "multiplier"}
        core = self._fit_core(
            data, outcome, unit, time, first_treat, treatment, covariates,
            ps_covariates=ps_covariates,
            or_covariates=or_covariates,
            compute_influence=compute_influence,
        )
        event_df = core["event_study"].copy()
        scalars = core["scalars"].copy()

        if self.inference == "multiplier":
            rng = np.random.default_rng(self.seed)
            finite_es = event_df.sort_values("horizon")
            H = finite_es["horizon"].tolist()
            Psi = np.column_stack([
                core["psi_by_h"][int(h)].reindex(
                    core["cluster_universe"], fill_value=0.0
                ).to_numpy(dtype=float)
                for h in H
            ])
            se_h = finite_es["se"].to_numpy(dtype=float)
            mb = run_multiplier_bootstrap(
                finite_es["estimate"].to_numpy(dtype=float),
                Psi, se_h,
                self.n_bootstrap, self.bootstrap_weights, self.alpha, rng,
            )
            event_df = finite_es.copy()
            event_df["ci_lower"] = mb["ci_lower"]
            event_df["ci_upper"] = mb["ci_upper"]
            event_df["p_value"] = mb["p_values"]
            event_df["t_stat"] = event_df["estimate"] / event_df["se"]
            event_df["sim_ci_lower"] = mb["sim_ci_lower"]
            event_df["sim_ci_upper"] = mb["sim_ci_upper"]
            core["metadata"].update({
                "multiplier": {
                    "critical_value": mb["cband_crit"],
                    "weight_type": self.bootstrap_weights,
                    "n_bootstrap": self.n_bootstrap,
                }
            })

        elif self.inference == "cluster_bootstrap":
            rng = np.random.default_rng(self.seed)

            def _refit(boot_df):
                boot_core = self._fit_core(
                    boot_df, outcome, unit, time, first_treat, treatment,
                    covariates,
                    ps_covariates=ps_covariates,
                    or_covariates=or_covariates,
                    compute_influence=False,
                )
                return {
                    "event_study": boot_core["event_study"],
                    "scalars": boot_core["scalars"],
                }

            orig = {
                "event_index": event_df["horizon"].tolist(),
                "event_hat": dict(zip(
                    event_df["horizon"].tolist(), event_df["estimate"].tolist()
                )),
                "scalar_terms": scalars["term"].tolist(),
                "scalar_hat": dict(zip(
                    scalars["term"].tolist(), scalars["estimate"].tolist()
                )),
            }
            boot = run_cluster_bootstrap(
                _refit, data.copy(), unit, self.n_bootstrap, self.alpha, rng, orig
            )
            event_df = event_df.merge(
                boot["event_boot"], on="horizon", how="left", suffixes=("", "_boot")
            )
            for c in ["se", "ci_lower", "ci_upper", "p_value", "t_stat"]:
                if f"{c}_boot" in event_df.columns:
                    event_df[c] = event_df[f"{c}_boot"]
                    event_df.drop(columns=[f"{c}_boot"], inplace=True)
            scalars = scalars.merge(
                boot["scalar_boot"], on="term", how="left", suffixes=("", "_boot")
            )
            for c in ["se", "ci_lower", "ci_upper", "p_value", "t_stat"]:
                if f"{c}_boot" in scalars.columns:
                    scalars[c] = scalars[f"{c}_boot"]
                    scalars.drop(columns=[f"{c}_boot"], inplace=True)

        return DRLPDIDResults(
            estimator_name="DR-LP-DiD",
            n_obs=core["n_obs"],
            n_treated_units=core["n_treated_units"],
            n_control_units=core["n_control_units"],
            n_cohorts=core["n_cohorts"],
            n_periods=core["n_periods"],
            base_period=self.base_period,
            clean_control=self.clean_control,
            effect_stabilization=self.effect_stabilization,
            anticipation=self.anticipation,
            inference=self.inference,
            alpha=self.alpha,
            event_study=event_df,
            scalars=scalars,
            metadata=core["metadata"],
            estimation_method=self.estimation_method,
            dr_method=self.dr_method,
            covariates=core["metadata"].get("or_covariates", list(covariates or [])),
            target_estimand="ATT",
        )


# ---------------------------------------------------------------------------
# Shared row helpers (also used by lpdid.py via direct call)
# ---------------------------------------------------------------------------

def _nan_row(h: int) -> dict:
    return {
        "horizon": int(h),
        "estimate": np.nan, "se": np.nan,
        "t_stat": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
    }


def _make_row(h: int, est: float, se: float, z: float) -> dict:
    t_stat = est / se if np.isfinite(est) and np.isfinite(se) and se > 0 else np.nan
    p_val = p_value_two_sided(t_stat) if np.isfinite(t_stat) else np.nan
    return {
        "horizon": int(h),
        "estimate": est,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_val,
        "ci_lower": est - z * se if np.isfinite(est) and np.isfinite(se) else np.nan,
        "ci_upper": est + z * se if np.isfinite(est) and np.isfinite(se) else np.nan,
    }
