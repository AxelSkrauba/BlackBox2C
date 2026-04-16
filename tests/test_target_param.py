"""
Tests for the unified 'target' parameter in convert() and Converter.convert().
Also covers: include_probabilities warning, real-dataset integration tests,
and coverage gaps in converter.py / config.py.
"""

import warnings

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from blackbox2c import Converter, ConversionConfig, convert


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def iris_data():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target
    )
    return X_train, X_test, y_train, y_test, list(iris.feature_names), list(iris.target_names)


@pytest.fixture(scope="module")
def iris_dt(iris_data):
    X_train, X_test, y_train, y_test, feat, cls = iris_data
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="module")
def iris_rf(iris_data):
    X_train, X_test, y_train, y_test, feat, cls = iris_data
    model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    return model


# ── target parameter ───────────────────────────────────────────────────────

class TestTargetParameter:
    def test_default_is_c(self, iris_dt, iris_data):
        X_train = iris_data[0]
        code = convert(iris_dt, X_train)
        assert "#include <stdint.h>" in code
        assert "uint8_t predict" in code

    def test_target_c_explicit(self, iris_dt, iris_data):
        X_train = iris_data[0]
        code = convert(iris_dt, X_train, target="c")
        assert "uint8_t predict" in code

    def test_target_cpp(self, iris_dt, iris_data):
        X_train, X_test = iris_data[0], iris_data[1]
        code = convert(iris_dt, X_train, X_test=X_test, target="cpp")
        assert "class Predictor" in code
        assert "#include <cstdint>" in code

    def test_target_arduino(self, iris_dt, iris_data):
        X_train = iris_data[0]
        code = convert(iris_dt, X_train, target="arduino")
        assert "#include <Arduino.h>" in code
        assert "uint8_t predict" in code

    def test_target_micropython(self, iris_dt, iris_data):
        X_train = iris_data[0]
        code = convert(iris_dt, X_train, target="micropython")
        assert "class Predictor:" in code
        assert "def predict" in code

    def test_converter_target_cpp(self, iris_rf, iris_data):
        X_train, X_test = iris_data[0], iris_data[1]
        feat, cls = iris_data[4], iris_data[5]
        converter = Converter(ConversionConfig(max_depth=4, n_samples=2000))
        code = converter.convert(
            iris_rf, X_train, X_test=X_test,
            feature_names=feat,
            class_names=cls,
            target="cpp",
        )
        assert "class Predictor" in code

    def test_converter_target_arduino(self, iris_dt, iris_data):
        X_train = iris_data[0]
        converter = Converter()
        code = converter.convert(iris_dt, X_train, target="arduino")
        assert "#include <Arduino.h>" in code

    def test_converter_target_micropython(self, iris_dt, iris_data):
        X_train = iris_data[0]
        converter = Converter()
        code = converter.convert(iris_dt, X_train, target="micropython")
        assert "class Predictor:" in code

    def test_all_targets_produce_output(self, iris_dt, iris_data):
        X_train = iris_data[0]
        for tgt in ["c", "cpp", "arduino", "micropython"]:
            code = convert(iris_dt, X_train, target=tgt)
            assert len(code) > 100, f"Empty or tiny output for target={tgt}"
            assert "predict" in code.lower(), f"No predict in target={tgt}"

    def test_target_cpp_case_insensitive(self, iris_dt, iris_data):
        X_train = iris_data[0]
        code = convert(iris_dt, X_train, target="CPP")
        assert "class Predictor" in code

    def test_metrics_preserved_for_all_targets(self, iris_rf, iris_data):
        X_train, X_test = iris_data[0], iris_data[1]
        for tgt in ["c", "cpp", "arduino", "micropython"]:
            converter = Converter(ConversionConfig(n_samples=1000))
            converter.convert(iris_rf, X_train, X_test=X_test, target=tgt)
            m = converter.get_metrics()
            assert "fidelity" in m
            assert "complexity" in m
            assert "size_estimate" in m


# ── include_probabilities warning ──────────────────────────────────────────

class TestIncludeProbabilitiesWarning:
    def test_warning_issued(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ConversionConfig(include_probabilities=True)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "not yet implemented" in str(w[0].message).lower()

    def test_no_warning_when_false(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ConversionConfig(include_probabilities=False)
        assert len(w) == 0


# ── Real dataset integration tests ─────────────────────────────────────────

class TestRealDatasetIntegration:
    def test_iris_end_to_end_c(self, iris_dt, iris_data):
        X_train, X_test, _, _, feat, cls = iris_data
        code = convert(
            iris_dt, X_train, X_test=X_test,
            feature_names=feat,
            class_names=cls,
            target="c",
            max_depth=4,
        )
        assert "uint8_t predict(float features[4])" in code
        assert "#define SETOSA 0" in code
        assert "if" in code and "return" in code

    def test_wine_random_forest(self):
        wine = load_wine()
        X_train, X_test, y_train, _ = train_test_split(
            wine.data, wine.target, test_size=0.25, random_state=42
        )
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        code = convert(
            model, X_train, X_test=X_test,
            class_names=list(wine.target_names),
            target="c",
            max_depth=5,
            n_samples=3000,
        )
        assert "uint8_t predict" in code
        assert len(code) > 200

    def test_iris_arduino_has_progmem(self, iris_dt, iris_data):
        X_train = iris_data[0]
        code = convert(iris_dt, X_train, target="arduino")
        assert "PROGMEM" in code

    def test_iris_micropython_valid_syntax(self, iris_dt, iris_data):
        X_train = iris_data[0]
        code = convert(iris_dt, X_train, target="micropython")
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated MicroPython code has syntax errors: {e}")

    def test_feature_threshold_flow(self):
        """Tests the feature_threshold branch in converter.py (lines 72-109)."""
        from sklearn.ensemble import RandomForestClassifier
        iris = load_iris()
        X_train, _, y_train, _ = train_test_split(
            iris.data, iris.target, test_size=0.25, random_state=42
        )
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)
        config = ConversionConfig(
            feature_threshold=2,
            max_depth=3,
            n_samples=500,
        )
        converter = Converter(config)
        code = converter.convert(model, X_train)
        assert "uint8_t predict" in code

    def test_classes_inferred_without_classes_attr(self):
        """Tests n_classes inference for models without .classes_ (lines 130-131).
        Uses a minimal stub model that has predict() but no classes_ attribute.
        """
        import numpy as np
        from sklearn.datasets import make_classification

        class _NoClassesModel:
            """Stub model without .classes_ to exercise the fallback branch."""
            _estimator_type = 'classifier'

            def __init__(self, inner):
                self._inner = inner

            def predict(self, X):
                return self._inner.predict(X)

        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42)
        inner = DecisionTreeClassifier(max_depth=3, random_state=42)
        inner.fit(X_train, y_train)
        model = _NoClassesModel(inner)
        code = convert(model, X_train, max_depth=3, n_samples=500)
        assert "uint8_t predict" in code

    def test_x_train_list_input_converted(self):
        """Tests that X_train as list (not ndarray) is handled properly."""
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(iris.data, iris.target)
        X_list = iris.data.tolist()
        code = convert(model, X_list)
        assert "uint8_t predict" in code
