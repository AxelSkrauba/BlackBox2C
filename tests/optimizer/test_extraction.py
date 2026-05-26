"""
Tests for ``optimizer.extraction.from_sklearn_tree``.

The cornerstone is the *equivalence oracle*: for any sklearn tree the
extracted RuleSet must produce identical predictions on a large pool
of samples.  Property-based tests (Hypothesis) randomise the tree
hyper-parameters and the input distribution to surface edge cases that
are easy to miss with hand-written examples.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from sklearn.datasets import (
    load_diabetes,
    load_iris,
    load_wine,
    make_classification,
    make_regression,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from blackbox2c.optimizer.extraction import from_sklearn_tree


_RNG = np.random.default_rng(0)


# ─────────────────────────── basic shape ────────────────────────────
class TestExtractionBasics:
    def test_rejects_non_tree_estimator(self):
        with pytest.raises(TypeError, match="tree_"):
            from_sklearn_tree(object())

    def test_classifier_metadata_preserved(self):
        X, y = load_iris(return_X_y=True)
        tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
        rs = from_sklearn_tree(tree, feature_names=load_iris().feature_names)

        assert rs.n_features == 4
        assert rs.n_classes == 3
        assert rs.feature_names is not None and len(rs.feature_names) == 4
        # Every conjunction must end at a leaf, hence have a non-negative
        # integer prediction within range.
        for conj in rs.rules:
            assert isinstance(conj.prediction, int)
            assert 0 <= conj.prediction < 3

    def test_regressor_metadata_preserved(self):
        X, y = load_diabetes(return_X_y=True)
        tree = DecisionTreeRegressor(max_depth=4, random_state=42).fit(X, y)
        rs = from_sklearn_tree(tree)
        assert rs.n_classes is None
        assert rs.n_features == X.shape[1]
        for conj in rs.rules:
            assert isinstance(conj.prediction, float)

    def test_single_leaf_tree(self):
        # Constant target → root is the only leaf.
        X = np.zeros((10, 2))
        y = np.zeros(10, dtype=int)
        tree = DecisionTreeClassifier().fit(X, y)
        rs = from_sklearn_tree(tree)
        assert len(rs.rules) == 1
        assert rs.rules[0].literals == ()  # always-true conjunction
        assert int(rs.predict(np.array([[1.0, 2.0]]))[0]) == 0

    def test_rejects_mismatched_feature_names(self):
        X, y = load_iris(return_X_y=True)
        tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
        with pytest.raises(ValueError, match="feature_names"):
            from_sklearn_tree(tree, feature_names=["a", "b"])  # 2 != 4


# ────────────────── deterministic equivalence oracle ────────────────
@pytest.mark.parametrize(
    "loader, depth",
    [
        (load_iris, 3),
        (load_iris, 6),
        (load_wine, 4),
        (load_wine, 8),
    ],
)
def test_extracted_classifier_matches_sklearn(loader, depth):
    X, y = loader(return_X_y=True)
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X, y)
    rs = from_sklearn_tree(tree)

    # 10k uniform samples within the feature ranges -> hard equivalence.
    rng = np.random.default_rng(42)
    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = 0.05 * (hi - lo)
    X_eval = rng.uniform(lo - pad, hi + pad, size=(10_000, X.shape[1]))

    assert np.array_equal(rs.predict(X_eval), tree.predict(X_eval))


@pytest.mark.parametrize("depth", [2, 5, 8])
def test_extracted_regressor_matches_sklearn(depth):
    X, y = load_diabetes(return_X_y=True)
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(X, y)
    rs = from_sklearn_tree(tree)

    rng = np.random.default_rng(7)
    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = 0.05 * (hi - lo)
    X_eval = rng.uniform(lo - pad, hi + pad, size=(5_000, X.shape[1]))

    np.testing.assert_allclose(rs.predict(X_eval), tree.predict(X_eval),
                               rtol=0, atol=1e-9)


# ─────────────────── property-based equivalence ─────────────────────
# Hypothesis explores a wide space of tree configurations.  Each example
# trains a tiny classifier and checks the extraction oracle on 1k random
# samples.  We disable the function-scoped-fixture warning since we use
# composite strategies to build our own data inside the test.
@settings(
    deadline=None,
    max_examples=25,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    n_features=st.integers(min_value=2, max_value=8),
    n_classes=st.integers(min_value=2, max_value=4),
    max_depth=st.integers(min_value=1, max_value=6),
    seed=st.integers(min_value=0, max_value=2**16 - 1),
)
def test_property_extraction_predicts_like_sklearn(
    n_features, n_classes, max_depth, seed
):
    # ``make_classification`` requires n_informative <= n_features and
    # n_informative >= log2(n_classes * n_clusters_per_class).
    n_informative = min(n_features, max(2, n_classes))
    n_samples = max(60, 4 * n_classes)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed).fit(X, y)
    rs = from_sklearn_tree(tree)

    rng = np.random.default_rng(seed + 1)
    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = 0.1 * (hi - lo + 1e-9)
    X_eval = rng.uniform(lo - pad, hi + pad, size=(1_000, n_features))

    assert np.array_equal(rs.predict(X_eval), tree.predict(X_eval))


@settings(
    deadline=None,
    max_examples=15,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    n_features=st.integers(min_value=1, max_value=6),
    max_depth=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=2**16 - 1),
)
def test_property_regression_extraction(n_features, max_depth, seed):
    X, y = make_regression(
        n_samples=80, n_features=n_features, noise=0.5, random_state=seed
    )
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=seed).fit(X, y)
    rs = from_sklearn_tree(tree)

    rng = np.random.default_rng(seed + 1)
    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = 0.1 * (hi - lo + 1e-9)
    X_eval = rng.uniform(lo - pad, hi + pad, size=(500, n_features))

    np.testing.assert_allclose(rs.predict(X_eval), tree.predict(X_eval),
                               rtol=0, atol=1e-9)


# ─────────────────── coverage / no-overlap invariants ───────────────
def test_ruleset_partitions_input_space():
    """For tree-derived RuleSets, conjunctions must partition the input
    domain: every sample matches *exactly* one rule."""
    X, y = load_iris(return_X_y=True)
    tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    rs = from_sklearn_tree(tree)

    rng = np.random.default_rng(0)
    X_eval = rng.uniform(X.min(axis=0), X.max(axis=0), size=(2_000, 4))

    matches = np.zeros(len(X_eval), dtype=int)
    for conj in rs.rules:
        matches += conj.evaluate(X_eval).astype(int)
    assert (matches == 1).all(), (
        f"Some rows matched 0 or >1 rules: {np.unique(matches, return_counts=True)}"
    )
