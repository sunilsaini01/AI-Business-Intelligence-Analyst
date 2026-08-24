import numpy as np

from app.tools.ml_tools import baseline_forecast


def test_baseline_forecast_extends_a_perfect_linear_trend():
    values = np.array([10.0, 20.0, 30.0, 40.0])
    forecast = baseline_forecast(values, periods_ahead=2)
    np.testing.assert_allclose(forecast, [50.0, 60.0], atol=1e-6)
