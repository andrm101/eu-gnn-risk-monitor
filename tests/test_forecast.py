import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from export_dashboard_data import _forecast_region


def test_forecast_linear_series():
    """For a perfectly linear series the point estimates must match exact extrapolation."""
    years = list(range(2010, 2024))   # 14 years — matches real data length
    slope = 0.01
    intercept = 0.05
    scores = [slope * y + intercept for y in years]

    result = _forecast_region(years, scores)

    assert len(result) == 3
    assert {pt['year'] for pt in result} == {2024, 2025, 2026}
    for pt in result:
        expected = slope * pt['year'] + intercept
        assert abs(pt['point'] - expected) < 1e-3, (
            f"Year {pt['year']}: expected ≈{round(expected, 4)}, got {pt['point']}"
        )
        assert pt['lower'] <= pt['point'] <= pt['upper'], (
            f"Prediction interval ordering violated at year {pt['year']}"
        )
        assert pt['lower'] >= 0.0, f"Lower bound is negative at year {pt['year']}"
        assert pt['year'] in (2024, 2025, 2026)
