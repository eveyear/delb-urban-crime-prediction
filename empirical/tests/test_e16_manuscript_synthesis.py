from __future__ import annotations

from pathlib import Path

import pandas as pd

from entropy_crime_bike.e16_manuscript_synthesis import _randomization_table


def test_randomization_table_preserves_zero_discoveries(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "null_design": ["temporal_block", "spatial_series", "circular_shift"],
            "randomization_p_value": [1.0, 0.5, 0.1],
            "observed_minus_null_mean_bits": [-0.1, 0.1, 0.2],
            "separates_global_fdr": [False, False, False],
        }
    )
    output = tmp_path / "table.tex"
    _randomization_table(output, frame)
    text = output.read_text(encoding="utf-8")
    assert "FDR discoveries" in text
    assert text.count("& 0\\\\") == 3
