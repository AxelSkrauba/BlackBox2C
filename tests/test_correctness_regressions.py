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

import re

import numpy as np
import pytest
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


class TestEstimatorImmutability:
    """
    ``convert()`` must treat the caller's estimator as immutable input.

    The optimizer mutates the tree in place, and ``Converter.convert`` stored
    the caller's ``DecisionTree*`` in ``surrogate_tree_`` directly, so a
    conversion pruned the user's fitted model: its predictions changed and a
    second conversion started from an already-damaged tree.
    """

    _SNAPSHOT_FIELDS = (
        'feature',
        'threshold',
        'children_left',
        'children_right',
        'value',
        'n_node_samples',
    )

    @pytest.mark.parametrize('kind', ['classifier', 'regressor'])
    def test_convert_does_not_mutate_caller_estimator(self, kind):
        if kind == 'classifier':
            model, X = _classification_tree()
        else:
            model, X = _regression_tree()

        snapshot = {
            field: np.array(getattr(model.tree_, field), copy=True)
            for field in self._SNAPSHOT_FIELDS
        }
        node_count = model.tree_.node_count
        expected_predictions = model.predict(X)

        Converter().convert(model, X)

        assert model.tree_.node_count == node_count
        for field, before in snapshot.items():
            assert np.array_equal(getattr(model.tree_, field), before), (
                f"convert() mutated tree_.{field}"
            )
        assert np.array_equal(model.predict(X), expected_predictions)

    def test_converting_the_same_model_twice_is_idempotent(self):
        model, X = _regression_tree()

        first = Converter().convert(model, X)
        second = Converter().convert(model, X)

        assert second == first

    def test_surrogate_tree_is_not_the_caller_estimator(self):
        model, X = _regression_tree()
        converter = Converter()
        converter.convert(model, X)

        assert converter.surrogate_tree_ is not model


class TestFidelityReporting:
    """
    The reported fidelity must describe the tree that is actually emitted.

    ``Converter.convert`` measured fidelity right after surrogate extraction
    and *before* ``optimizer.optimize``, and hardcoded ``1.0`` for direct tree
    inputs. The metric therefore described a tree that was no longer the one
    being exported, which is why the regression corruption went unnoticed.
    """

    @staticmethod
    def _damage_tree(self, tree):
        """Stand-in for ``RuleOptimizer.optimize`` that destroys the tree."""
        tree.tree_.value[:] = 0.0
        return tree

    def test_fidelity_is_measured_after_optimization(self, monkeypatch):
        X, y = make_regression(
            n_samples=300, n_features=4, n_informative=3, noise=5.0, random_state=3
        )
        model = GradientBoostingRegressor(random_state=3).fit(X, y)
        X_eval = X[:120]
        assert np.unique(model.predict(X_eval)).size > 3

        converter = Converter(
            ConversionConfig(
                max_depth=3, n_samples=400, fidelity_warning_threshold=None
            )
        )
        monkeypatch.setattr(RuleOptimizer, 'optimize', self._damage_tree)
        converter.convert(model, X, X_test=X_eval)

        assert converter.metrics_['fidelity'] < 0.1, (
            "fidelity described the pre-optimization surrogate instead of the "
            "emitted tree"
        )
        assert converter.metrics_['fidelity_evaluation_set'] == 'test'

    def test_direct_tree_fidelity_is_measured_not_assumed(self, monkeypatch):
        model, X = _regression_tree()
        X_eval = X[:120]

        converter = Converter(ConversionConfig(fidelity_warning_threshold=None))
        monkeypatch.setattr(RuleOptimizer, 'optimize', self._damage_tree)
        converter.convert(model, X, X_test=X_eval)

        assert converter.metrics_['fidelity'] < 0.1, (
            "direct tree conversion still reports a hardcoded fidelity of 1.0"
        )
        assert converter.metrics_['fidelity_evaluation_set'] == 'test'

    def test_direct_tree_without_test_set_keeps_its_source_label(self):
        model, X = _regression_tree()

        converter = Converter(ConversionConfig(fidelity_warning_threshold=None))
        converter.convert(model, X)

        assert converter.metrics_['fidelity_evaluation_set'] == 'direct_tree'
        assert converter.metrics_['fidelity'] == pytest.approx(1.0)

    def test_reported_fidelity_matches_the_emitted_regression_tree(self):
        model, X = _regression_tree()
        X_eval = X[:120]

        converter = Converter(ConversionConfig(optimize_rules='high'))
        converter.convert(model, X, X_test=X_eval)

        expected = r2_score(
            model.predict(X_eval), converter.surrogate_tree_.predict(X_eval)
        )
        assert converter.metrics_['fidelity'] == pytest.approx(expected, abs=1e-12)

    def test_reported_fidelity_matches_the_emitted_classifier(self):
        X, y = make_classification(
            n_samples=400,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            n_classes=3,
            random_state=5,
        )
        model = RandomForestClassifier(n_estimators=20, random_state=5).fit(X, y)
        X_eval = X[:150]

        converter = Converter(
            ConversionConfig(
                max_depth=4,
                n_samples=500,
                optimize_rules='high',
                fidelity_warning_threshold=None,
            )
        )
        converter.convert(model, X, X_test=X_eval)

        expected = np.mean(
            model.predict(X_eval) == converter.surrogate_tree_.predict(X_eval)
        )
        assert converter.metrics_['fidelity'] == pytest.approx(expected, abs=1e-12)


class TestFeatureThresholdGuard:
    """
    ``feature_threshold`` must never emit code that indexes outside the
    generated signature.

    Feature selection slices ``X_train`` down to the selected columns and then
    reuses a direct ``DecisionTree*`` input unchanged, so a 4-feature tree
    trained on all columns was emitted under a ``predict(float features[2])``
    signature while still reading ``features[3]`` - an out-of-bounds read in the
    generated C/C++.
    """

    @pytest.mark.parametrize('kind', ['classifier', 'regressor'])
    def test_direct_tree_with_feature_threshold_raises(self, kind):
        model, X = (
            _classification_tree() if kind == 'classifier' else _regression_tree()
        )

        with pytest.raises(ValueError, match='feature_threshold=2 is not supported'):
            Converter(ConversionConfig(feature_threshold=2)).convert(model, X)

    def test_error_names_the_model_type_and_a_workaround(self):
        model, X = _classification_tree()

        with pytest.raises(ValueError) as excinfo:
            Converter(ConversionConfig(feature_threshold=2)).convert(model, X)

        message = str(excinfo.value)
        assert 'DecisionTreeClassifier' in message
        assert 'Pipeline' in message

    def test_direct_tree_without_feature_threshold_still_converts(self):
        model, X = _classification_tree()
        code = Converter().convert(model, X)

        assert 'predict(float features[4])' in code

    @pytest.mark.parametrize('scenario', ['direct_tree', 'feature_selection'])
    def test_emitted_feature_indices_fit_the_signature(self, scenario):
        """
        Memory-safety invariant for every emission path.

        Whatever the generated signature declares, the body must only index
        features inside it. The direct tree + ``feature_threshold`` combination
        violated this (signature ``features[2]``, body reading ``features[3]``).
        """
        if scenario == 'direct_tree':
            model, X = _classification_tree()
            config = ConversionConfig()
        else:
            X, y = make_classification(
                n_samples=300,
                n_features=5,
                n_informative=3,
                n_redundant=0,
                n_classes=2,
                random_state=11,
            )
            model = RandomForestClassifier(n_estimators=10, random_state=11).fit(X, y)
            config = ConversionConfig(
                feature_threshold=3,
                max_depth=3,
                n_samples=300,
                fidelity_warning_threshold=None,
            )

        code = Converter(config).convert(model, X)

        # Ignore comments (which document a usage example such as
        # ``float input[4] = {...};``) and the signature itself, whose array
        # size is not an index.
        body = re.sub(r'/\*.*?\*/', '', code, flags=re.S)
        signature = re.search(r'predict\(float features\[(\d+)\]\)', body)
        assert signature is not None
        n_features = int(signature.group(1))
        body = body.replace(signature.group(0), '', 1)
        used_indices = {int(i) for i in re.findall(r'features\[(\d+)\]', body)}
        assert used_indices <= set(range(n_features)), (
            f"emitted indices {sorted(used_indices)} do not fit a "
            f"features[{n_features}] signature"
        )
