from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def nonnegative_half_up(values: Iterable[float]) -> np.ndarray:
    """Round conditional means to the nonnegative integer lattice."""
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Predictions must be finite before integer rounding.")
    return np.floor(np.maximum(array, 0.0) + 0.5).astype(np.int64)


def crime_lag_state(values: pd.Series) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="raise")
    labels = np.select(
        [
            numeric.eq(0),
            numeric.eq(1),
            numeric.eq(2),
            numeric.ge(3),
        ],
        ["zero", "one", "two", "three_plus"],
        default=None,
    )
    return pd.Categorical(
        labels,
        categories=["zero", "one", "two", "three_plus"],
        ordered=True,
    )


def _string_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.loc[:, columns].copy()
    for column in columns:
        result[column] = result[column].astype(str)
    return result


@dataclass
class StateMeanRegressor:
    features: list[str]
    smoothing: float
    grid_column: str = "grid_id"

    def fit(self, frame: pd.DataFrame, target: Iterable[float]) -> "StateMeanRegressor":
        if self.smoothing < 0:
            raise ValueError("Smoothing must be nonnegative.")
        working = _string_frame(frame, self.features)
        working["_target"] = np.asarray(target, dtype=float)
        self.global_mean_ = float(working["_target"].mean())
        self.grid_means_ = (
            working.groupby(self.grid_column, observed=True)["_target"]
            .mean()
            .to_dict()
        )
        grouped = (
            working.groupby(self.features, observed=True, dropna=False)["_target"]
            .agg(["sum", "count"])
            .reset_index()
        )
        self.state_sum_ = {}
        self.state_count_ = {}
        for row in grouped.itertuples(index=False):
            key = tuple(str(getattr(row, column)) for column in self.features)
            self.state_sum_[key] = float(row.sum)
            self.state_count_[key] = int(row.count)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        working = _string_frame(frame, self.features)
        predictions = np.empty(len(working), dtype=float)
        grid_index = self.features.index(self.grid_column)
        for index, key_values in enumerate(
            working.itertuples(index=False, name=None)
        ):
            key = tuple(str(value) for value in key_values)
            grid = key[grid_index]
            prior = float(self.grid_means_.get(grid, self.global_mean_))
            count = int(self.state_count_.get(key, 0))
            if count == 0:
                predictions[index] = prior
            else:
                predictions[index] = (
                    float(self.state_sum_[key]) + self.smoothing * prior
                ) / (count + self.smoothing)
        return np.maximum(predictions, 0.0)


@dataclass
class NegativeBinomialResult:
    encoder: OneHotEncoder
    coefficients: np.ndarray
    dispersion: float
    l2_penalty: float
    converged: bool
    iterations: int
    objective: float
    features: list[str]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        encoded = self.encoder.transform(_string_frame(frame, self.features))
        design = sparse.hstack(
            [
                sparse.csr_matrix(np.ones((len(frame), 1), dtype=float)),
                encoded,
            ],
            format="csr",
        )
        linear = np.asarray(design @ self.coefficients).ravel()
        return np.exp(np.clip(linear, -20.0, 20.0))


def fit_negative_binomial(
    frame: pd.DataFrame,
    target: Iterable[float],
    *,
    features: list[str],
    dispersion: float,
    l2_penalty: float,
    max_iter: int,
    tolerance: float,
    initial_coefficients: np.ndarray | None = None,
) -> NegativeBinomialResult:
    """Fit an NB2 log-link model with sparse categorical fixed effects."""
    if dispersion <= 0:
        raise ValueError("NB2 dispersion must be positive.")
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float64,
    )
    encoded = encoder.fit_transform(_string_frame(frame, features))
    design = sparse.hstack(
        [
            sparse.csr_matrix(np.ones((len(frame), 1), dtype=float)),
            encoded,
        ],
        format="csr",
    )
    y = np.asarray(target, dtype=float)
    if np.any(y < 0) or not np.isfinite(y).all():
        raise ValueError("NB2 targets must be finite nonnegative counts.")
    r = 1.0 / float(dispersion)
    if initial_coefficients is None or len(initial_coefficients) != design.shape[1]:
        coefficients = np.zeros(design.shape[1], dtype=float)
        coefficients[0] = math.log(max(float(y.mean()), 1.0e-6))
    else:
        coefficients = np.asarray(initial_coefficients, dtype=float).copy()
    penalty_mask = np.ones(design.shape[1], dtype=float)
    penalty_mask[0] = 0.0
    sample_size = float(len(y))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.asarray(design @ beta).ravel()
        mu = np.exp(np.clip(eta, -20.0, 20.0))
        log_likelihood = (
            gammaln(y + r)
            - gammaln(r)
            - gammaln(y + 1.0)
            + r * (math.log(r) - np.log(r + mu))
            + y * (np.log(mu) - np.log(r + mu))
        )
        penalty = 0.5 * l2_penalty * np.dot(
            penalty_mask * beta, beta
        )
        value = -float(log_likelihood.sum()) / sample_size + penalty
        score_eta = r * (y - mu) / (r + mu)
        gradient = -np.asarray(design.T @ score_eta).ravel() / sample_size
        gradient += l2_penalty * penalty_mask * beta
        return value, gradient

    fitted = minimize(
        objective,
        coefficients,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(max_iter),
            "ftol": float(tolerance),
            "gtol": float(tolerance),
            "maxls": 40,
        },
    )
    return NegativeBinomialResult(
        encoder=encoder,
        coefficients=np.asarray(fitted.x, dtype=float),
        dispersion=float(dispersion),
        l2_penalty=float(l2_penalty),
        converged=bool(fitted.success),
        iterations=int(fitted.nit),
        objective=float(fitted.fun),
        features=list(features),
    )


def build_poisson_pipeline(
    features: list[str],
    *,
    alpha: float,
    max_iter: int,
    tolerance: float,
) -> Pipeline:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float64,
    )
    regressor = PoissonRegressor(
        alpha=float(alpha),
        max_iter=int(max_iter),
        tol=float(tolerance),
        fit_intercept=True,
    )
    return Pipeline([("encoder", encoder), ("regressor", regressor)])


def build_histogram_poisson_pipeline(
    features: list[str],
    *,
    learning_rate: float,
    max_leaf_nodes: int,
    l2_regularization: float,
    max_iter: int,
    min_samples_leaf: int,
    random_seed: int,
) -> Pipeline:
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.float64,
    )
    regressor = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=float(learning_rate),
        max_leaf_nodes=int(max_leaf_nodes),
        l2_regularization=float(l2_regularization),
        max_iter=int(max_iter),
        min_samples_leaf=int(min_samples_leaf),
        categorical_features=np.ones(len(features), dtype=bool),
        early_stopping=False,
        random_state=int(random_seed),
    )
    return Pipeline([("encoder", encoder), ("regressor", regressor)])


def model_input(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return _string_frame(frame, features)
