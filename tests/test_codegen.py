"""
Tests for C code generation module.

Critical regression tests for the feature-name-as-C-identifier bug (fixed in
v0.1.1): the generator must always emit ``features[i]`` array indexing, never
the raw feature name string, which may contain spaces and special characters
that are invalid as C identifiers.
"""

import re
import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression, load_iris

from blackbox2c.codegen import CCodeGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_raw_feature_name(c_code: str, feature_names: list) -> bool:
    """Return True if any raw feature name appears bare in an if-condition."""
    for name in feature_names:
        # Look for the name outside of a comment or string literal
        if re.search(r'if\s*\(' + re.escape(name), c_code):
            return True
    return False


def _all_conditions_use_array(c_code: str) -> bool:
    """Return True if every if-condition uses features[<int>] syntax."""
    conditions = re.findall(r'if\s*\(([^)]+)\)', c_code)
    for cond in conditions:
        if not re.match(r'\s*features\[\d+\]', cond):
            return False
    return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def iris_tree():
    """Iris DecisionTreeClassifier with real feature names (contain spaces)."""
    iris = load_iris()
    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree.fit(iris.data, iris.target)
    return tree, list(iris.feature_names), list(iris.target_names)


@pytest.fixture
def synthetic_clf_tree():
    """Synthetic 3-class classification tree."""
    X, y = make_classification(
        n_samples=200, n_features=4, n_informative=3,
        n_redundant=0, n_classes=3, random_state=42
    )
    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree.fit(X, y)
    feature_names = [f"sensor reading {i} (V)" for i in range(4)]
    return tree, feature_names, ['class_0', 'class_1', 'class_2']


@pytest.fixture
def regression_tree():
    """Simple regression tree."""
    X, y = make_regression(n_samples=200, n_features=3, random_state=42)
    tree = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree.fit(X, y)
    feature_names = ["temperature (°C)", "humidity (%)", "pressure (hPa)"]
    return tree, feature_names


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestCCodeGeneratorInit:
    def test_defaults(self):
        gen = CCodeGenerator()
        assert gen.function_name == 'predict'
        assert gen.use_fixed_point is False
        assert gen.precision == 8
        assert gen.task_type is None

    def test_custom_params(self):
        gen = CCodeGenerator(function_name='my_predict', use_fixed_point=True, precision=16)
        assert gen.function_name == 'my_predict'
        assert gen.use_fixed_point is True
        assert gen.precision == 16


# ---------------------------------------------------------------------------
# Bug regression: feature names must NOT appear as C identifiers
# ---------------------------------------------------------------------------

class TestFeatureNameBugRegression:
    """
    Regression tests for the bug where feature names with spaces/special chars
    were emitted verbatim in if-conditions, producing invalid C code.
    Fixed in v0.1.1.
    """

    def test_iris_feature_names_not_in_conditions(self, iris_tree):
        """Iris feature names contain spaces — must not appear bare in if()."""
        tree, feature_names, class_names = iris_tree
        gen = CCodeGenerator()
        code = gen.generate(tree, feature_names, class_names)
        assert not _has_raw_feature_name(code, feature_names), (
            "Raw feature name found in if-condition — bug not fixed.\n"
            f"Names: {feature_names}"
        )

    def test_iris_all_conditions_use_array_indexing(self, iris_tree):
        """Every if-condition in Iris output must use features[i] syntax."""
        tree, feature_names, class_names = iris_tree
        gen = CCodeGenerator()
        code = gen.generate(tree, feature_names, class_names)
        assert _all_conditions_use_array(code), (
            "Found if-condition not using features[i] array indexing.\n"
            + code[:500]
        )

    def test_special_char_feature_names(self, synthetic_clf_tree):
        """Feature names with spaces, parentheses, units — still valid C."""
        tree, feature_names, class_names = synthetic_clf_tree
        gen = CCodeGenerator()
        code = gen.generate(tree, feature_names, class_names)
        assert not _has_raw_feature_name(code, feature_names)
        assert _all_conditions_use_array(code)

    def test_regression_feature_names_not_in_conditions(self, regression_tree):
        """Same bug check for regression output."""
        tree, feature_names = regression_tree
        gen = CCodeGenerator()
        code = gen.generate(tree, feature_names)
        assert not _has_raw_feature_name(code, feature_names)
        assert _all_conditions_use_array(code)

    def test_single_feature_tree(self):
        """Edge case: single feature with a long weird name."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTreeClassifier(max_depth=2, random_state=0)
        tree.fit(X, y)
        feature_names = ["CO2 concentration (ppm)"]
        gen = CCodeGenerator()
        code = gen.generate(tree, feature_names, ["low", "high"])
        assert not _has_raw_feature_name(code, feature_names)
        assert _all_conditions_use_array(code)


# ---------------------------------------------------------------------------
# Output structure validation
# ---------------------------------------------------------------------------

class TestGeneratedCodeStructure:
    def test_classification_includes_and_signature(self, iris_tree):
        tree, feature_names, class_names = iris_tree
        gen = CCodeGenerator()
        code = gen.generate(tree, feature_names, class_names)
        assert '#include <stdint.h>' in code
        assert 'uint8_t predict(' in code
        assert 'return' in code
        assert 'if' in code

    def test_classification_defines_use_sanitized_names(self, iris_tree):
        """#define macros must use valid C identifiers (no spaces)."""
        tree, feature_names, class_names = iris_tree
        gen = CCodeGenerator()
        code = gen.generate(tree, feature_names, class_names)
        # Iris target names: setosa, versicolor, virginica — already valid
        assert '#define SETOSA 0' in code
        assert '#define VERSICOLOR 1' in code
        assert '#define VIRGINICA 2' in code

    def test_defines_with_special_class_names(self):
        """Class names with spaces/hyphens must be sanitized in #define lines."""
        X, y = make_classification(n_samples=50, n_features=2, n_classes=2,
                                    n_informative=2, n_redundant=0, random_state=0)
        tree = DecisionTreeClassifier(max_depth=2, random_state=0)
        tree.fit(X, y)
        gen = CCodeGenerator(task_type='classification')
        code = gen.generate(tree, ["f0", "f1"], ["class-A (ok)", "class-B (ok)"])
        # #define lines must use sanitized identifiers — no raw name with hyphens
        define_lines = [ln for ln in code.splitlines() if ln.startswith('#define')]
        for ln in define_lines:
            assert 'class-A' not in ln, f"Raw class name in #define: {ln}"
            assert 'class-B' not in ln, f"Raw class name in #define: {ln}"
        # Must contain sanitized version as a #define
        assert any('CLASS_A' in ln for ln in define_lines)

    def test_regression_float_return(self, regression_tree):
        tree, feature_names = regression_tree
        gen = CCodeGenerator()
        code = gen.generate(tree, feature_names)
        assert 'float predict(' in code
        assert '#include <stdint.h>' in code
        # Regression leaves return float literals
        assert re.search(r'return -?\d+\.\d+f;', code)

    def test_custom_function_name(self, iris_tree):
        tree, feature_names, class_names = iris_tree
        gen = CCodeGenerator(function_name='classify_iris')
        code = gen.generate(tree, feature_names, class_names)
        assert 'uint8_t classify_iris(' in code

    def test_fixed_point_8bit(self, iris_tree):
        tree, feature_names, class_names = iris_tree
        gen = CCodeGenerator(use_fixed_point=True, precision=8)
        code = gen.generate(tree, feature_names, class_names)
        assert 'int8_t features[' in code

    def test_fixed_point_16bit(self, iris_tree):
        tree, feature_names, class_names = iris_tree
        gen = CCodeGenerator(use_fixed_point=True, precision=16)
        code = gen.generate(tree, feature_names, class_names)
        assert 'int16_t features[' in code

    def test_float_mode(self, iris_tree):
        tree, feature_names, class_names = iris_tree
        gen = CCodeGenerator(use_fixed_point=False)
        code = gen.generate(tree, feature_names, class_names)
        assert 'float features[' in code


# ---------------------------------------------------------------------------
# Sanitizer helper
# ---------------------------------------------------------------------------

class TestSanitizeCIdentifier:
    def test_simple_name(self):
        assert CCodeGenerator._sanitize_c_identifier('setosa') == 'SETOSA'

    def test_name_with_spaces(self):
        result = CCodeGenerator._sanitize_c_identifier('petal length (cm)')
        assert re.match(r'^[A-Z][A-Z0-9_]*$', result), f"Not a valid C identifier: {result}"

    def test_name_starting_with_digit(self):
        result = CCodeGenerator._sanitize_c_identifier('1class')
        assert result.startswith('CLS_')

    def test_hyphen_and_slash(self):
        result = CCodeGenerator._sanitize_c_identifier('class-A/B')
        assert re.match(r'^[A-Z][A-Z0-9_]*$', result)

    def test_already_valid(self):
        assert CCodeGenerator._sanitize_c_identifier('MY_CLASS') == 'MY_CLASS'


# ---------------------------------------------------------------------------
# Size estimation
# ---------------------------------------------------------------------------

class TestCodeSizeEstimation:
    def test_estimate_fields(self, iris_tree):
        tree, _, _ = iris_tree
        gen = CCodeGenerator()
        est = gen.estimate_code_size(tree)
        assert {'flash_bytes', 'ram_bytes', 'n_nodes', 'n_conditions', 'n_leaves'} <= est.keys()
        assert est['flash_bytes'] > 0
        assert est['ram_bytes'] > 0
        assert est['n_nodes'] > 0
        assert est['n_leaves'] > 0
        assert est['n_conditions'] >= 0

    def test_deeper_tree_larger_estimate(self):
        iris = load_iris()
        gen = CCodeGenerator()
        tree_shallow = DecisionTreeClassifier(max_depth=2, random_state=0)
        tree_shallow.fit(iris.data, iris.target)
        tree_deep = DecisionTreeClassifier(max_depth=6, random_state=0)
        tree_deep.fit(iris.data, iris.target)
        est_s = gen.estimate_code_size(tree_shallow)
        est_d = gen.estimate_code_size(tree_deep)
        assert est_d['flash_bytes'] > est_s['flash_bytes']


# ---------------------------------------------------------------------------
# Header and comment generation
# ---------------------------------------------------------------------------

class TestHeaderAndComments:
    def test_classification_header(self):
        gen = CCodeGenerator(precision=16, task_type='classification')
        header = gen._generate_header(4, 3)
        assert '* Auto-generated C code by BlackBox2C' in header
        assert '*   - Input features: 4' in header
        assert '*   - Output classes: 3' in header
        assert '*   - Precision: 16-bit' in header
        assert '#include <stdint.h>' in header

    def test_regression_header(self):
        gen = CCodeGenerator(precision=8, task_type='regression')
        header = gen._generate_header(4, None)
        assert '*   - Task: Regression' in header

    def test_usage_comment_contains_feature_names(self):
        gen = CCodeGenerator(function_name='my_predict', task_type='classification')
        comment = gen._generate_usage_comment(['feat_a', 'feat_b'], ['A', 'B'])
        assert 'my_predict' in comment
        assert 'feat_a' in comment
