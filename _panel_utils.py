"""Panel data utilities for LP-DiD estimators.

Provides panel preparation (``prepare_panel``), local comparison-stack
construction (``build_local_sample``), clean-comparison-support indicators
(``precompute_ccs``), and assorted helpers used by the estimator classes.

Clean-control conditions implemented
-------------------------------------
``'not_yet_treated'``
    Absorbing treatment. Controls: :math:`D_{i,t+h}=0`.
    (Dube et al. 2025, eq. 8.)
``'never_treated'``
    Absorbing treatment. Controls: units with :math:`p_i = \\infty` only.
``'first_entry'``
    Non-absorbing treatment. Treated: first-time entrants at *t* that stay
    treated through :math:`t+h`. Controls: units untreated at :math:`t+h`.
    (Dube et al. 2025, Section 4.2.2, eq. 12.)
``'stabilized'``
    Non-absorbing treatment with effect-stabilization horizon *L*. Treated:
    entered treatment at *t*, no prior change within *L* periods. Controls:
    no treatment-status change within :math:`[-h, L]`.
    (Dube et al. 2025, Section 4.2.3, eq. 13.)
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import warnings
import math

import numpy as np
import pandas as pd

BasePeriod = Union[int, List[int], Tuple[int, ...], str]

# ---------------------------------------------------------------------------
# Basic panel helpers
# ---------------------------------------------------------------------------

def check_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def sort_panel(df: pd.DataFrame, unit: str, time: str) -> pd.DataFrame:
    return df.sort_values([unit, time]).reset_index(drop=True).copy()


def lag(df: pd.DataFrame, unit: str, col: str, h: int) -> pd.Series:
    return df.groupby(unit, sort=False)[col].shift(h)


def lead(df: pd.DataFrame, unit: str, col: str, h: int) -> pd.Series:
    return df.groupby(unit, sort=False)[col].shift(-h)


def is_binary(series: pd.Series) -> bool:
    vals = set(pd.Series(series.dropna().unique()).tolist())
    return vals.issubset({0, 1})


def assert_absorbing(df: pd.DataFrame, unit: str, treat: str) -> None:
    """Raise ValueError if treatment ever switches from 1 → 0 within a unit."""
    if (df.groupby(unit, sort=False)[treat].diff() < 0).any():
        raise ValueError(
            f"Treatment column '{treat}' is not absorbing (switches 1→0). "
            "Set nonabsorbing=True to allow non-absorbing treatment."
        )


def _norm_cdf(x: float) -> float:
    return float(0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))


def z_crit(alpha: float) -> float:
    from statistics import NormalDist
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def p_value_two_sided(t_stat: float) -> float:
    return float(2.0 * (1.0 - _norm_cdf(abs(float(t_stat)))))


# ---------------------------------------------------------------------------
# base_period coercion
# ---------------------------------------------------------------------------

def coerce_base_period(base_period: BasePeriod) -> BasePeriod:
    """Validate and normalize *base_period*.

    Accepted values
    ---------------
    ``int``
        Negative integer, e.g. ``-1``.
    ``list`` / ``tuple``
        Collection of negative integers; returned as sorted list.
    ``'all_pre'``
        Average of all pre-treatment outcomes.
    """
    if isinstance(base_period, int):
        if base_period >= 0:
            raise ValueError("base_period integer must be negative, e.g. -1.")
        return int(base_period)
    if isinstance(base_period, (list, tuple)):
        vals = [int(v) for v in base_period]
        if not vals or any(v >= 0 for v in vals):
            raise ValueError(
                "base_period list/tuple must contain only negative integers."
            )
        return sorted(set(vals))
    if isinstance(base_period, str):
        if base_period != "all_pre":
            raise ValueError("base_period string must be 'all_pre'.")
        return "all_pre"
    raise ValueError(f"Invalid base_period: {base_period!r}.")


# ---------------------------------------------------------------------------
# Panel preparation
# ---------------------------------------------------------------------------

def _derive_first_treat(
    df: pd.DataFrame, unit: str, time: str, treatment: str
) -> pd.Series:
    first = (
        df.loc[df[treatment] == 1, [unit, time]]
        .groupby(unit, sort=False)[time]
        .min()
    )
    mapped = df[unit].map(first).fillna(0)
    return pd.to_numeric(mapped, errors="coerce").fillna(0)


def prepare_panel(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    first_treat: Optional[str],
    treatment: Optional[str],
    nonabsorbing: bool = False,
    n_lagged_outcome_changes: int = 0,
) -> pd.DataFrame:
    """Validate, sort, and augment a long panel for LP-DiD estimation.

    Parameters
    ----------
    data : pd.DataFrame
    outcome, unit, time : str
    first_treat : str or None
        Column with first treatment period (0 = never treated).
    treatment : str or None
        Binary treatment indicator. Either ``first_treat`` or ``treatment``
        must be provided.
    nonabsorbing : bool
        If ``True``, skip the absorbing-treatment check. Required for
        ``clean_control='first_entry'`` or ``'stabilized'``.
    n_lagged_outcome_changes : int
        Number of lagged first-differences of *outcome* to add as controls.

    Returns
    -------
    pd.DataFrame
        Augmented panel with columns ``_first_treat``, ``_treat``,
        ``_never_treated``, ``_treated_ever``, ``_first_obs_time``,
        ``_left_censored``, ``_exposure_age``, ``D_treat``, ``dy``,
        ``ldy1``, ``ldy2``, … .
    """
    if first_treat is None and treatment is None:
        raise ValueError("Provide either `first_treat` or `treatment`.")
    req = [outcome, unit, time]
    if first_treat is not None:
        req.append(first_treat)
    if treatment is not None:
        req.append(treatment)
    check_columns(data, req)

    df = sort_panel(data.copy(), unit, time)
    df[time] = pd.to_numeric(df[time], errors="coerce")

    if first_treat is None:
        if not is_binary(df[treatment]):
            raise ValueError("treatment must be binary 0/1.")
        df["_first_treat_internal"] = _derive_first_treat(df, unit, time, treatment)
        first_treat = "_first_treat_internal"

    df["_first_treat"] = pd.to_numeric(df[first_treat], errors="coerce").fillna(0)
    if treatment is None:
        df["_treat"] = (
            (df["_first_treat"] > 0) & (df[time] >= df["_first_treat"])
        ).astype(int)
    else:
        df["_treat"] = (
            pd.to_numeric(df[treatment], errors="coerce").fillna(0).astype(int)
        )

    if not is_binary(df["_treat"]):
        raise ValueError("Derived treatment path is not binary 0/1.")

    if not nonabsorbing:
        assert_absorbing(df, unit, "_treat")

    df["_first_obs_time"] = df.groupby(unit, sort=False)[time].transform("min")
    df["_left_censored"] = (
        (df["_treat"] == 1)
        & (df["_first_treat"] > 0)
        & (df["_first_treat"] < df["_first_obs_time"])
    ).astype(int)

    left_censored_units = df.loc[df["_left_censored"] == 1, unit].drop_duplicates().tolist()
    if left_censored_units:
        warnings.warn(
            f"Dropping {len(left_censored_units)} left-censored unit(s) already "
            "treated in the first observed period.",
            stacklevel=2,
        )
        df = df.loc[~df[unit].isin(left_censored_units)].copy()
        df = sort_panel(df, unit, time)
        df["_first_obs_time"] = df.groupby(unit, sort=False)[time].transform("min")
        df["_left_censored"] = (
            (df["_treat"] == 1)
            & (df["_first_treat"] > 0)
            & (df["_first_treat"] < df["_first_obs_time"])
        ).astype(int)

    df["D_treat"] = df.groupby(unit, sort=False)["_treat"].diff()
    first_obs = df.groupby(unit, sort=False).cumcount() == 0
    df.loc[first_obs, "D_treat"] = df.loc[first_obs, "_treat"]
    df.loc[first_obs & (df["_left_censored"] == 1), "D_treat"] = 0

    df["dy"] = df.groupby(unit, sort=False)[outcome].diff()
    for k in range(1, int(max(n_lagged_outcome_changes, 0)) + 1):
        df[f"ldy{k}"] = df.groupby(unit, sort=False)["dy"].shift(k)

    df["_never_treated"] = (df["_first_treat"] == 0).astype(int)
    df["_treated_ever"] = (df["_first_treat"] > 0).astype(int)
    df["_exposure_age"] = np.where(
        df["_first_treat"] > 0, df[time] - df["_first_treat"], np.nan
    )
    df["rel_time"] = df["_exposure_age"]
    return df


# ---------------------------------------------------------------------------
# base_period helpers
# ---------------------------------------------------------------------------

def _all_pre_base(df: pd.DataFrame, outcome: str, unit: str) -> pd.Series:
    obsnum = df.groupby(unit, sort=False).cumcount() + 1
    cumy = df.groupby(unit, sort=False)[outcome].cumsum()
    out = lag(pd.DataFrame({"_cumy": cumy, unit: df[unit]}), unit, "_cumy", 1) / (obsnum - 1)
    out = pd.Series(out, index=df.index)
    out.loc[obsnum <= 1] = np.nan
    return out


def compute_base_series(
    df: pd.DataFrame, outcome: str, unit: str, base_period: BasePeriod
) -> pd.Series:
    if isinstance(base_period, int):
        return lag(df, unit, outcome, abs(base_period))
    if isinstance(base_period, list):
        mats = np.column_stack([lag(df, unit, outcome, abs(v)) for v in base_period])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return pd.Series(np.nanmean(mats, axis=1), index=df.index)
    return _all_pre_base(df, outcome, unit)


def compute_long_difference(
    df: pd.DataFrame, outcome: str, unit: str, h: int, base_period: BasePeriod
) -> pd.Series:
    base = compute_base_series(df, outcome, unit, base_period)
    if h >= 0:
        return lead(df, unit, outcome, h) - base
    return lag(df, unit, outcome, abs(h)) - base


# ---------------------------------------------------------------------------
# Window inference
# ---------------------------------------------------------------------------

def infer_windows(
    df: pd.DataFrame,
    time: str,
    base_period: BasePeriod,
    max_pre: Optional[int] = None,
    max_post: Optional[int] = None,
) -> Tuple[int, int]:
    rel = pd.to_numeric(df["rel_time"], errors="coerce")
    max_post_data = int(rel.max()) if np.isfinite(rel.max()) else 0
    neg_rel = rel.loc[np.isfinite(rel) & (rel < 0)]
    max_pre_data = int(abs(neg_rel.min())) if not neg_rel.empty else 0

    if isinstance(base_period, int):
        max_pre_data = max(max_pre_data, abs(base_period))
    elif isinstance(base_period, list):
        max_pre_data = max(max_pre_data, max(abs(v) for v in base_period))
    else:
        max_pre_data = max(max_pre_data, 1)

    final_pre = max_pre_data if max_pre is None else min(int(max_pre), max_pre_data)
    final_post = max_post_data if max_post is None else min(int(max_post), max_post_data)
    return int(max(final_pre, 0)), int(max(final_post, 0))


# ---------------------------------------------------------------------------
# CCS indicators for the stabilized condition (Section 4.2.3)
# ---------------------------------------------------------------------------

def precompute_ccs(
    df: pd.DataFrame, unit: str, effect_stabilization: int,
    max_pre: int, max_post: int,
) -> pd.DataFrame:
    """Precompute clean-comparison-support indicators horizon by horizon.

    Used by ``clean_control='stabilized'`` to implement eq. (13) of
    Dube et al. (2025, Section 4.2.3).
    """
    out = df.copy()
    ccs0 = pd.Series(True, index=out.index)
    for k in range(1, effect_stabilization + 1):
        ccs0 = ccs0 & (lag(out, unit, "D_treat", k).abs() != 1)
    out["CCS_0"] = ccs0.astype(int)

    for h in range(1, max_post + 1):
        prev = out[f"CCS_{h-1}"] == 1
        no_future = lead(out, unit, "D_treat", h).abs() != 1
        out[f"CCS_{h}"] = (prev & no_future).astype(int)

    out["CCS_m1"] = out["CCS_0"]
    for h in range(2, max_pre + 1):
        prev = out[f"CCS_m{h-1}"] == 1
        lag_prev = lag(out, unit, f"CCS_m{h-1}", 1) == 1
        out[f"CCS_m{h}"] = (prev & lag_prev).astype(int)
    return out


# ---------------------------------------------------------------------------
# Local comparison stack builder
# ---------------------------------------------------------------------------

def build_local_sample(
    df: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    h: int,
    base_period: BasePeriod,
    clean_control: str,
    effect_stabilization: Optional[int],
    extra_cols: List[str],
    user_covariates: Optional[List[str]] = None,
    lag_covariates: bool = False,
) -> pd.DataFrame:
    """Build the horizon-h clean local comparison stack :math:`S_h`.

    Implements the sample restrictions for all four supported clean-control
    conditions (Dube et al. 2025, Sections 3.1, 4.2.2, 4.2.3).

    Parameters
    ----------
    df : pd.DataFrame
        Prepared panel (output of :func:`prepare_panel`).
    outcome, unit, time : str
    h : int
        Event-time horizon (negative = pre-treatment).
    base_period : BasePeriod
    clean_control : {'not_yet_treated', 'never_treated', 'first_entry', \
'stabilized'}
    effect_stabilization : int or None
        Required when ``clean_control='stabilized'``.
    extra_cols : list of str
        Additional columns that must be non-missing.

    Returns
    -------
    pd.DataFrame
        Stack with columns ``D_local`` (1 = treated, 0 = control) and
        ``outcome_local`` (long-differenced outcome).
    """
    s = df.copy()
    s["outcome_local"] = compute_long_difference(s, outcome, unit, h, base_period)

    # --- treated mask --------------------------------------------------
    if clean_control in {"not_yet_treated", "never_treated"}:
        # Absorbing: newly treated at t (ΔD = 1)
        treat_mask = s["D_treat"] == 1

    elif clean_control == "first_entry":
        # Non-absorbing, Section 4.2.2 (eq. 12):
        # ΔD_it = 1  AND  D_{i,t-j} = 0 for j ≥ 1 (first-time entry)
        # AND (for h≥0) stays treated through t+h
        newly_treated = s["D_treat"] == 1
        first_time = s["_first_treat"] == s[time]  # never treated before t
        if h >= 0:
            stays_treated = lead(s, unit, "_treat", h).fillna(0) == 1
        else:
            stays_treated = pd.Series(True, index=s.index)
        treat_mask = newly_treated & first_time & stays_treated

    elif clean_control == "stabilized":
        # Non-absorbing, Section 4.2.3 (eq. 13):
        # Treated: ΔD_it = 1 AND no treatment change in [t-L, t-1]
        # (the CCS pre-indicators capture this)
        treat_mask = s["D_treat"] == 1

    else:
        raise ValueError(
            f"Unknown clean_control='{clean_control}'. "
            "Must be one of {{'not_yet_treated', 'never_treated', "
            "'first_entry', 'stabilized'}}."
        )

    # --- control mask --------------------------------------------------
    if clean_control == "never_treated":
        ctrl_mask = s["_never_treated"] == 1

    elif clean_control in {"not_yet_treated", "first_entry"}:
        # Controls: not yet treated at t+h (or at t-|h| for pre-periods)
        if h >= 0:
            ctrl_mask = lead(s, unit, "_treat", h).fillna(1.0) == 0
        else:
            ctrl_mask = s["_treat"] == 0

    elif clean_control == "stabilized":
        if effect_stabilization is None:
            raise ValueError(
                "clean_control='stabilized' requires effect_stabilization."
            )
        col = f"CCS_{h}" if h >= 0 else ("CCS_m1" if h == -1 else f"CCS_m{abs(h)}")
        stable_mask = s[col] == 1
        # Control = any unit with no treatment ONSET at t and clean CCS history.
        # Matches the Stata condition exactly: D.treat==0 & CCS_j==1
        # (Dube et al. 2025, eq. 13; appendix_sim_LPDiD_estimation.do lines 31-35).
        # The CCS indicator already guarantees the required stability window;
        # an additional exposure-age filter is not needed and not in the paper.
        ctrl_mask = (s["D_treat"] == 0) & stable_mask

    # --- lag user covariates to t-1 (predetermined) --------------------
    # Computed on the full panel BEFORE any subsetting.
    # ldy* columns are already lags and must NOT be lagged again.
    # We overwrite the column in-place so downstream formulas using the
    # original column name automatically receive X_{i,t-1}.
    if lag_covariates and user_covariates:
        for c in user_covariates:
            if c in s.columns:
                s[c] = lag(s, unit, c, 1)

    # --- assemble stack ------------------------------------------------
    s = s.loc[treat_mask | ctrl_mask].copy()
    s["D_local"] = treat_mask.loc[s.index].astype(int)

    needed = [unit, time, "D_local", "outcome_local"] + list(extra_cols)
    s = s.dropna(subset=[c for c in needed if c in s.columns]).copy()
    s = s.loc[np.isfinite(s["outcome_local"])].copy()
    if s.empty:
        return s.reset_index(drop=True)

    # Drop time periods with no control observations
    ctrl_by_time = s.loc[s["D_local"] == 0].groupby(time).size()
    good_times = ctrl_by_time.loc[ctrl_by_time > 0].index
    s = s.loc[s[time].isin(good_times)].copy()
    return s.reset_index(drop=True)
