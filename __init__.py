"""lpdid — Local Projections Difference-in-Differences for Python.

Implements the LP-DiD family of estimators introduced by Dube, Girardi,
Jordà & Taylor (2025) and a semiparametric doubly-robust extension.

Estimators
----------
LPDID
    Benchmark LP-DiD estimator covering all three target estimands:

    * ``target_estimand='vw'`` — variance-weighted ATT (Section 3.1).
    * ``target_estimand='rw'`` — equally-weighted ATT via FWL reweighting
      (Section 3.3; equivalent to Callaway & Sant'Anna 2020).
    * ``target_estimand='ra'`` — equally-weighted ATT via regression-adjustment
      imputation (Section 3.3; equivalent to BJS 2024 with ``all_pre``).

    Supports both absorbing and non-absorbing treatment:

    * ``clean_control='not_yet_treated'`` / ``'never_treated'`` — absorbing.
    * ``clean_control='first_entry'`` — non-absorbing, first-time entry
      (Section 4.2.2, eq. 12); requires ``nonabsorbing=True``.
    * ``clean_control='stabilized'`` — non-absorbing, effect-stabilization
      (Section 4.2.3, eq. 13); requires ``nonabsorbing=True`` and
      ``effect_stabilization=L``.

DRLPDID
    Semiparametric doubly-robust estimator (RA / IPW / DR variants).

Utilities
---------
plot_event_study
    Quick event-study plot from any result object.

Common interface::

    result = LPDID(target_estimand='vw').fit(
        data, outcome='y', unit='id', time='t', first_treat='g'
    )
    result.print_summary()

References
----------
Dube, A., Girardi, D., Jordà, Ò., & Taylor, A. M. (2025).
    A local projections approach to difference-in-differences.
    *Journal of Applied Econometrics*, 40, 741–758.
    https://doi.org/10.1002/jae.70000
"""

from .lpdid import LPDID
from .drlpdid import DRLPDID
from ._plotting import plot_event_study
from ._results import LPDIDResults, DRLPDIDResults

__all__ = [
    "LPDID",
    "DRLPDID",
    "LPDIDResults",
    "DRLPDIDResults",
    "plot_event_study",
]
