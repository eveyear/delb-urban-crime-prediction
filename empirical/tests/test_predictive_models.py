from __future__ import annotations

import numpy as np
import pandas as pd

from entropy_crime_bike.predictive_models import (
    StateMeanRegressor,
    crime_lag_state,
    fit_negative_binomial,
    nonnegative_half_up,
)


def test_nonnegative_half_up_rounding() -> None:
    values = nonnegative_half_up([-0.2, 0.49, 0.5, 1.5, 2.499])
    assert values.tolist() == [0, 0, 1, 2, 2]


def test_crime_lag_state() -> None:
    states = crime_lag_state(pd.Series([0, 1, 2, 3, 8]))
    assert states.astype(str).tolist() == [
        "zero",
        "one",
        "two",
        "three_plus",
        "three_plus",
    ]


def test_state_mean_unseen_state_backs_off_to_grid() -> None:
    train = pd.DataFrame(
        {
            "grid_id": ["a", "a", "b", "b"],
            "state": ["x", "x", "x", "y"],
        }
    )
    model = StateMeanRegressor(["grid_id", "state"], smoothing=0.0)
    model.fit(train, [0, 2, 4, 6])
    predicted = model.predict(
        pd.DataFrame({"grid_id": ["a", "b"], "state": ["z", "x"]})
    )
    assert predicted.tolist() == [1.0, 4.0]


def test_negative_binomial_fit_predicts_finite_positive_means() -> None:
    frame = pd.DataFrame(
        {
            "grid_id": ["a"] * 20 + ["b"] * 20,
            "state": ["x", "y"] * 20,
        }
    )
    target = np.array([0, 1] * 10 + [1, 3] * 10)
    fitted = fit_negative_binomial(
        frame,
        target,
        features=["grid_id", "state"],
        dispersion=0.5,
        l2_penalty=1.0e-4,
        max_iter=100,
        tolerance=1.0e-8,
    )
    prediction = fitted.predict(frame)
    assert len(prediction) == len(frame)
    assert np.isfinite(prediction).all()
    assert (prediction > 0).all()
