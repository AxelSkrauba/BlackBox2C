"""Tests for conversion transparency warnings and FLASH budget validation."""

import warnings
from unittest.mock import patch

import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from blackbox2c import ConversionConfig, Converter


@pytest.fixture
def iris_models():
    X, y = load_iris(return_X_y=True)
    X_train, X_test = X[:100], X[100:]
    y_train = y[:100]
    forest = RandomForestClassifier(n_estimators=5, random_state=42).fit(
        X_train, y_train
    )
    tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
    return X_train, X_test, forest, tree


def test_surrogate_without_test_set_warns_and_records_train_source(iris_models):
    X_train, _, forest, _ = iris_models
    converter = Converter(ConversionConfig(n_samples=200))

    with patch(
        "blackbox2c.converter.SurrogateExtractor.get_fidelity", return_value=0.99
    ):
        with pytest.warns(UserWarning, match="X_train"):
            converter.convert(forest, X_train)

    assert converter.metrics_["fidelity_evaluation_set"] == "train"
    assert converter.metrics_["fidelity_warning_threshold"] == 0.95


def test_low_surrogate_fidelity_warns_against_configured_threshold(iris_models):
    X_train, X_test, forest, _ = iris_models
    converter = Converter(
        ConversionConfig(n_samples=200, fidelity_warning_threshold=0.8)
    )

    with patch(
        "blackbox2c.converter.SurrogateExtractor.get_fidelity", return_value=0.5
    ):
        with pytest.warns(UserWarning, match="below fidelity_warning_threshold=0.8000"):
            converter.convert(forest, X_train, X_test=X_test)

    assert converter.metrics_["fidelity_evaluation_set"] == "test"
    assert converter.metrics_["fidelity"] == 0.5


def test_direct_tree_has_explicit_fidelity_source_without_warnings(iris_models):
    X_train, _, _, tree = iris_models
    converter = Converter(ConversionConfig(fidelity_warning_threshold=1.0))

    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        converter.convert(tree, X_train)

    assert not warnings_record
    assert converter.metrics_["fidelity"] == 1.0
    assert converter.metrics_["fidelity_evaluation_set"] == "direct_tree"


def test_flash_budget_exceedance_warns_and_records_metrics(iris_models):
    X_train, _, _, tree = iris_models
    converter = Converter(ConversionConfig(memory_budget_kb=0.001))

    with pytest.warns(UserWarning, match="estimated FLASH"):
        converter.convert(tree, X_train)

    assert converter.metrics_["memory_budget_met"] is False
    assert converter.metrics_["memory_budget_flash_bytes"] == pytest.approx(1.024)


def test_flash_budget_success_is_recorded_without_warning(iris_models):
    X_train, _, _, tree = iris_models
    converter = Converter(ConversionConfig(memory_budget_kb=100.0))

    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        converter.convert(tree, X_train)

    assert not warnings_record
    assert converter.metrics_["memory_budget_met"] is True
    assert converter.metrics_["memory_budget_flash_bytes"] == pytest.approx(102400.0)
