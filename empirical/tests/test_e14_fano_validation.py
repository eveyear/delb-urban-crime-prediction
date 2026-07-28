from __future__ import annotations

import math

import numpy as np
import pytest

from entropy_crime_bike.e14_fano_validation import (
    conditional_entropy_pair,
    fano_inverse,
    fano_phi,
    modal_classifier_error,
    target_codes,
)


def test_binary_inverse_uses_increasing_branch() -> None:
    for probability in [0.0, 0.05, 0.2, 0.49]:
        entropy = fano_phi(probability, 2)
        assert fano_inverse(entropy, 2) == pytest.approx(probability)


def test_multiclass_inverse_uses_correct_branch() -> None:
    for classes in [3, 5]:
        for probability in [0.0, 0.1, 0.4]:
            entropy = fano_phi(probability, classes)
            assert fano_inverse(entropy, classes) == pytest.approx(
                probability
            )
        assert fano_inverse(math.log2(classes), classes) == pytest.approx(
            1 - 1 / classes
        )


def test_fixed_target_definitions() -> None:
    counts = np.array([0, 1, 2, 5])
    assert np.array_equal(
        target_codes(counts, "binary_occurrence"), [0, 1, 1, 1]
    )
    assert np.array_equal(
        target_codes(counts, "three_level_count"), [0, 1, 2, 2]
    )


def test_modal_classifier_and_fano_floor() -> None:
    train_y = np.array([0, 0, 1, 1, 1, 0])
    train_s = np.array([0, 0, 1, 1, 1, 2])
    test_y = np.array([0, 1, 0, 1])
    test_s = np.array([0, 1, 2, 3])
    plugin, mm = conditional_entropy_pair(train_y, train_s, 2)
    train_error, test_error, unseen = modal_classifier_error(
        train_y, train_s, test_y, test_s, 2
    )
    assert plugin >= 0
    assert mm >= plugin
    assert train_error == 0
    assert test_error == pytest.approx(0.25)
    assert unseen == 1


def test_invalid_class_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        fano_inverse(0.0, 1)
    with pytest.raises(ValueError):
        fano_phi(0.8, 2)
