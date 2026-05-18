"""Result containers for LP-DiD estimators.

Each estimator returns a dataclass that inherits from :class:`EventStudyResults`
and exposes two tidy DataFrames:

* ``event_study`` — horizon-by-horizon ATT estimates with standard errors,
  confidence intervals, and p-values.
* ``scalars`` — pooled summaries (``ATT avg`` and ``ATT pooled``).

The ``summary()`` / ``print_summary()`` methods produce a formatted table
suitable for console inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sig_code(p: float) -> str:
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "."
    return ""


def _fmt(x: float, width: int = 12, digits: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return f"{'nan':>{width}}"
    return f"{x:>{width}.{digits}f}"


# ---------------------------------------------------------------------------
# Base result class
# ---------------------------------------------------------------------------

@dataclass
class EventStudyResults:
    """Shared base for all LP-DiD result objects.

    Attributes
    ----------
    estimator_name : str
    n_obs : int
    n_treated_units : int
    n_control_units : int
    n_cohorts : int
    n_periods : int
    base_period : int, list of int, or 'all_pre'
    clean_control : str
    effect_stabilization : int or None
    anticipation : int
    inference : str
    alpha : float
    event_study : pd.DataFrame
        Columns: horizon, estimate, se, t_stat, p_value, ci_lower, ci_upper
        (optionally sim_ci_lower, sim_ci_upper for multiplier bootstrap).
    scalars : pd.DataFrame
        Columns: term, estimate, se, t_stat, p_value, ci_lower, ci_upper.
    metadata : dict
    """

    estimator_name: str
    n_obs: int
    n_treated_units: int
    n_control_units: int
    n_cohorts: int
    n_periods: int
    base_period: object
    clean_control: str
    effect_stabilization: Optional[int]
    anticipation: int
    inference: str
    alpha: float
    event_study: pd.DataFrame = field(default_factory=pd.DataFrame)
    scalars: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict = field(default_factory=dict)

    @property
    def event_study_effects(self) -> pd.DataFrame:
        return self.event_study

    @property
    def scalar_summaries(self) -> pd.DataFrame:
        return self.scalars

    def summary(self) -> str:
        W = 105
        lines: list[str] = []
        lines.append("=" * W)
        lines.append(f"{self.estimator_name:^{W}}")
        lines.append("=" * W)
        lines.append("")
        lines.append(f"Total observations:       {self.n_obs:>28}")
        lines.append(f"Treated units:            {self.n_treated_units:>28}")
        lines.append(f"Control units:            {self.n_control_units:>28}")
        lines.append(f"Treatment cohorts:        {self.n_cohorts:>28}")
        lines.append(f"Time periods:             {self.n_periods:>28}")
        lines.append(f"Base period:              {str(self.base_period):>28}")
        lines.append(f"Clean control:            {self.clean_control:>28}")
        lines.append(f"Effect stabilization L:   {str(self.effect_stabilization):>28}")
        lines.append(f"Anticipation:             {self.anticipation:>28}")
        lines.append(f"Inference:                {self.inference:>28}")

        notes = self.metadata.get("notes") if isinstance(self.metadata, dict) else None
        if notes:
            lines.append("")
            lines.append("Notes:")
            for note in notes:
                lines.append(f"  - {note}")

        id_w = 38
        col_header = (
            f"{'Horizon':<{id_w}}"
            f"{'Estimate':>12}{'Std. Err.':>14}{'t-stat':>14}{'P>|t|':>14}{'Sig.':>8}"
        )

        # ── Scalar summaries ───────────────────────────────────────────────
        if not self.scalars.empty:
            lines.append("")
            lines.append("-" * W)
            lines.append(f"{'Scalar summaries':^{W}}")
            lines.append("-" * W)
            lines.append(
                f"{'Term':<{id_w}}"
                f"{'Estimate':>12}{'Std. Err.':>14}{'t-stat':>14}{'P>|t|':>14}{'Sig.':>8}"
            )
            lines.append("-" * W)
            for _, row in self.scalars.iterrows():
                lines.append(
                    f"{str(row['term']):<{id_w}}"
                    f"{_fmt(row.get('estimate', np.nan), 12)}"
                    f"{_fmt(row.get('se', np.nan), 14)}"
                    f"{_fmt(row.get('t_stat', np.nan), 14)}"
                    f"{_fmt(row.get('p_value', np.nan), 14)}"
                    f"{_sig_code(row.get('p_value', np.nan)):>8}"
                )
            lines.append("-" * W)

        # ── Event-study path ───────────────────────────────────────────────
        if not self.event_study.empty:
            lines.append("")
            lines.append("-" * W)
            lines.append(f"{'Event-study path':^{W}}")
            lines.append("-" * W)
            lines.append(col_header)
            lines.append("-" * W)
            for _, row in self.event_study.sort_values("horizon").iterrows():
                lines.append(
                    f"{int(row['horizon']):<{id_w}}"
                    f"{_fmt(row.get('estimate', np.nan), 12)}"
                    f"{_fmt(row.get('se', np.nan), 14)}"
                    f"{_fmt(row.get('t_stat', np.nan), 14)}"
                    f"{_fmt(row.get('p_value', np.nan), 14)}"
                    f"{_sig_code(row.get('p_value', np.nan)):>8}"
                )
            lines.append("-" * W)

        lines.append("")
        lines.append("Signif. codes:  '***' 0.001  '**' 0.01  '*' 0.05  '.' 0.1")
        lines.append("=" * W)
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())


# ---------------------------------------------------------------------------
# Concrete result types
# ---------------------------------------------------------------------------

@dataclass
class LPDIDResults(EventStudyResults):
    """Results for the LP-DiD estimator (VW, RW, and RA target estimands).

    Additional attributes
    ---------------------
    target_estimand : str
        ``'VW'``, ``'RW'``, or ``'RA'``.
    nonabsorbing : bool
        Whether non-absorbing treatment was enabled.
    covariates : list of str
    """
    target_estimand: str = "VW"
    nonabsorbing: bool = False
    covariates: list = field(default_factory=list)


@dataclass
class DRLPDIDResults(EventStudyResults):
    """Results for the semiparametric DR-LP-DiD estimator.

    Additional attributes
    ---------------------
    estimation_method : str
        ``'ra'``, ``'ipw'``, or ``'dr'``.
    dr_method : str or None
        ``'generic'``, ``'improved'``, or ``'ipt_wls'``.
    covariates : list of str
    target_estimand : str
        Always ``'ATT'``.
    """
    estimation_method: str = "dr"
    dr_method: Optional[str] = "generic"
    covariates: list = field(default_factory=list)
    target_estimand: str = "ATT"
