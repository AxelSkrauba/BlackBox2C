"""
Regression tests for correctness defects found in the v0.2.x audit.

Each class pins one behaviour that produced silently wrong output in a
released version, so the defect cannot come back unnoticed:

* ``TestRegressionRuleOptimization`` - rule pruning must not collapse
  sibling leaves that predict different values (v0.2.0 - v0.2.3).
* ``TestFidelityReporting`` - the reported fidelity must describe the
  tree that is actually emitted, not a pre-optimisation one.
* ``TestFeatureThresholdGuard`` - ``feature_threshold`` must never emit
  code that indexes outside the generated function signature.
* ``TestEstimatorImmutability`` - ``convert()`` must not modify the
  estimator owned by the caller.
"""

import copy

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from blackbox2c import ConversionConfig, Converter
from blackbox2c.optimizer import RuleOptimizer
from blackbox2c.tree_constants import TREE_LEAF


def _regression_tree(max_depth=4, random_state=7):
    """Fit a regression tree on synthetic data and return it with its data."""
    X, y = make_regression(
        n_samples=300,
        n_features=4,
        n_informative=3,
        noise=5.0,
        random_state=random_state,
    )
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    tree.fit(X, y)
    return tree, X


def _classification_tree(max_depth=6, random_state=42):
    """Fit a classification tree on synthetic data and return it with its data."""
    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        random_state=random_state,
    )
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    tree.fit(X, y)
    return tree, X


def _leaf_counts(tree):
    return int(np.sum(tree.tree_.feature == TREE_LEAF))


def _same_prediction_sibling_parents(tree):
    """
    Nodes whose two children are leaves predicting the same class.

    This restates the documented ``medium`` pruning rule independently of
    the implementation, so the test can pin exactly which nodes collapse.
    """
    tree_struct = tree.tree_
    parents = []
    for node_id in range(tree_struct.node_count):
        if tree_struct.feature[node_id] == TREE_LEAF:
            continue
        left = tree_struct.children_left[node_id]
        right = tree_struct.children_right[node_id]
        if (
            tree_struct.feature[left] == TREE_LEAF
            and tree_struct.feature[right] == TREE_LEAF
            and np.argmax(tree_struct.value[left][0])
            == np.argmax(tree_struct.value[right][0])
        ):
            parents.append(node_id)
    return parents


class TestRegressionRuleOptimization:
    """
    Regression trees must keep their predictions through rule optimisation.

    ``_prune_redundant_branches`` compared ``np.argmax(value[child][0])``.
    Regressors store a single scalar per node, so that ``argmax`` is always
    ``0`` and every node with two leaf children was collapsed into the
    average of both leaves. ``optimize_rules='medium'`` is the default, so
    every regression conversion silently lost accuracy.
    """

    def test_medium_pruning_preserves_regression_predictions(self):
        tree, X = _regression_tree()
        expected = tree.predict(X)
        optimized = RuleOptimizer('medium').optimize(tree)

        assert np.array_equal(optimized.predict(X), expected)
        assert r2_score(expected, optimized.predict(X)) == pytest.approx(1.0)

    def test_high_merging_preserves_regression_function(self):
        tree, X = _regression_tree()
        expected = tree.predict(X)
        optimized = RuleOptimizer('high').optimize(tree)

        # 'high' may merge sibling leaves with nearly identical means, so the
        # function must stay close instead of being flattened into one leaf.
        assert r2_score(expected, optimized.predict(X)) > 0.99
        assert _leaf_counts(optimized) < optimized.tree_.node_count

    @pytest.mark.parametrize('level', ['medium', 'high'])
    def test_regression_tree_is_not_collapsed_into_a_stump(self, level):
        tree, X = _regression_tree()
        optimized = RuleOptimizer(level).optimize(tree)

        predictions = optimized.predict(X)
        assert np.unique(predictions).size > 1, (
            "regression tree collapsed into a constant prediction"
        )

    def test_medium_pruning_still_collapses_same_class_siblings(self):
        """
        Classification behaviour is unchanged.

        Every parent whose two leaf children predict the same class must still
        become a leaf, and predictions must stay identical.
        """
        tree, X = _classification_tree()
        expected_collapses = _same_prediction_sibling_parents(tree)
        assert expected_collapses, "fixture must contain collapsible branches"

        expected_predictions = tree.predict(X)
        optimized = RuleOptimizer('medium').optimize(tree)
        tree_struct = optimized.tree_

        assert np.array_equal(optimized.predict(X), expected_predictions)
        for node_id in expected_collapses:
            assert tree_struct.feature[node_id] == TREE_LEAF
            assert tree_struct.children_left[node_id] == tree_struct.children_right[node_id]
