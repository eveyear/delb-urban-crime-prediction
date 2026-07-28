import pandas as pd

from entropy_crime_bike.e11_manuscript_synthesis import (
    _data_summary_table,
    latex_escape,
)


def test_latex_escape_handles_table_special_characters():
    assert latex_escape("A&B_1%") == r"A\&B\_1\%"


def test_data_summary_table_uses_observed_values():
    panel = pd.DataFrame(
        {
            "city": ["DC", "NY", "VAN"],
            "active_grids": [1, 2, 3],
            "panel_rows": [10, 20, 30],
            "crime_events_in_primary_domain": [4, 5, 6],
            "bike_trips_any_endpoint_in_primary_domain": [7, 8, 9],
        }
    )
    variables = pd.DataFrame(
        {
            "city": ["DC", "NY", "VAN"] * 2,
            "variable": ["crime_count_all"] * 3 + ["bike_total_flow"] * 3,
            "mean": [0.1, 0.2, 0.3, 1.0, 2.0, 3.0],
            "zero_share": [0.9, 0.8, 0.7, 0.0, 0.0, 0.0],
        }
    )
    result = _data_summary_table(panel, variables)
    assert "Washington, DC" in result
    assert "New York City" in result
    assert "30" in result
    assert "70.0\\%" in result
