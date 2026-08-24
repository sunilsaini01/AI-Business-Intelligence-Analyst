"""Deterministic sklearn/XGBoost/SHAP helpers for the ML Agent (Sec 2, Sec 5).
Phase 8. No LLM calls anywhere in this module (Sec 5 "0 LLM calls" rule).

Sec 2 JUDGMENT CALL: `baseline_forecast` (linear-trend/seasonal-naive) should
be tried, backtested, and reported before `xgboost_forecast` is ever reached
for — 18-24 monthly points is not where gradient boosting earns its
complexity.
"""

from __future__ import annotations

import numpy as np


def baseline_forecast(monthly_values: np.ndarray, periods_ahead: int = 1) -> np.ndarray:
    """Linear-trend baseline via numpy.polyfit — the first thing to try, not XGBoost."""
    x = np.arange(len(monthly_values))
    coeffs = np.polyfit(x, monthly_values, deg=1)
    future_x = np.arange(len(monthly_values), len(monthly_values) + periods_ahead)
    return np.polyval(coeffs, future_x)
