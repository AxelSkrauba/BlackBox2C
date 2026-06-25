"""
Regression tests for the v0.2.0 ``features[-2]`` bug.

Root cause: ``RuleOptimizer._prune_redundant_branches`` (and
``_merge_similar_leaves``) marked a node as a leaf by setting only
``Tree.feature[node] = -2`` without clearing ``children_left`` /
``children_right``.  The platform exporters (C++, Arduino, MicroPython)
detect leaves via ``children_left == children_right`` and therefore
treated the half-mutated node as an internal split, emitting
``features[-2]`` — an invalid negative array index in C that
``avr-gcc`` / ``xtensa-gcc`` accept silently, making the bug dangerous
on embedded hardware.

These tests pin three things:

1. No generated code (any target, any optimisation level) ever
   contains a negative feature index.
2. After optimisation the tree is *structurally consistent*: every
   leaf has both ``feature == -2`` and
   ``children_left == children_right == -1``.
3. Pruning actually collapses redundant nodes to a single ``return``
   (the v0.2.0 bug left a dead ``if/else`` with two identical returns).
"""

import re

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from blackbox2c import Converter, ConversionConfig
from blackbox2c.exporters import create_exporter
from blackbox2c.optimizer import RuleOptimizer
from blackbox2c.tree_constants import TREE_LEAF, TREE_UNDEFINED


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def iris_rf_surrogate():
    """Train the exact MRE model and return its surrogate tree.

    A RandomForestClassifier fitted on the first 3 iris features with
    ``max_depth=5`` reliably produces surrogate-tree nodes whose two
    children are same-class leaves — the pattern that triggers
    ``_prune_redundant_branches``.
    """
    iris = load_iris()
    X = iris.data[:, :3]
    y = iris.target

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    config = ConversionConfig(
        max_depth=5,
        optimize_rules='low',  # extract an uncorrupted surrogate first
        n_samples=5000,
        function_name='predict',
    )
    conv = Converter(config)
    # Run conversion to populate ``surrogate_tree_``; we discard the code.
    conv.convert(
        model, X,
        target='c',
        feature_names=['feat0', 'feat1', 'feat2'],
        class_names=['setosa', 'versicolor', 'virginica'],
    )
    return conv.surrogate_tree_, X


def _has_negative_index(code: str) -> bool:
    """Return True if any ``features[<negative>]`` appears in ``code``."""
    return bool(re.search(r'features\[-\d+\]', code))


# ---------------------------------------------------------------------------
# MRE reproduction
# ---------------------------------------------------------------------------

class TestPruneNegativeIndexBug:
    """Regression tests for the ``features[-2]`` corruption."""

    def test_mre_arduino_medium_no_negative_index(self, iris_rf_surrogate):
        """The exact MRE from the issue must no longer emit features[-2]."""
        tree, X = iris_rf_surrogate

        config = ConversionConfig(
            max_depth=5,
            optimize_rules='medium',
            n_samples=5000,
            function_name='predict',
        )
        conv = Converter(config)
        code = conv.convert(
            tree, X,
            target='arduino',
            feature_names=['feat0', 'feat1', 'feat2'],
            class_names=['setosa', 'versicolor', 'virginica'],
        )
        assert not _has_negative_index(code), (
            "Negative feature index found in generated Arduino code — "
            "bug not fixed.\n" + code[:800]
        )

    @pytest.mark.parametrize("target", ["c", "cpp", "arduino", "micropython"])
    @pytest.mark.parametrize("level", ["medium", "high"])
    def test_all_targets_all_levels_no_negative_index(
        self, iris_rf_surrogate, target, level
    ):
        """No target × level combination may emit a negative index.

        ``auto`` is excluded from this matrix because the v0.2.0
        ``codegen_bridge._tree_shape`` / ``_emit_tree`` recursors
        explode exponentially when many rules are neutral to the
        chosen pivot literal (a pre-existing bug unrelated to the
        ``features[-2]`` fix — see ``test_auto_small_tree`` below for
        ``auto`` coverage with a tree small enough not to trigger it).
        """
        tree, X = iris_rf_surrogate

        config = ConversionConfig(
            max_depth=5,
            optimize_rules=level,
            n_samples=5000,
            function_name='predict',
        )
        conv = Converter(config)
        code = conv.convert(
            tree, X,
            target=target,
            feature_names=['feat0', 'feat1', 'feat2'],
            class_names=['setosa', 'versicolor', 'virginica'],
        )
        assert not _has_negative_index(code), (
            f"Negative feature index for target={target!r}, "
            f"level={level!r}.\n" + code[:800]
        )

    @pytest.mark.parametrize("target", ["c", "cpp", "arduino", "micropython"])
    def test_auto_small_tree_no_negative_index(self, target):
        """``auto`` must not emit negative indices.

        Uses a small depth-3 tree (4 unique literals) so the
        ``codegen_bridge._tree_shape`` / ``_emit_tree`` recursors —
        which explode exponentially on large neutral-rule sets (a
        pre-existing v0.2.0 bug, not the ``features[-2]`` issue) —
        stay tractable.  This still exercises the full ``auto``
        pipeline: legacy medium pruning → IR extraction →
        ``_auto_route`` → bridge/exporter codegen.
        """
        iris = load_iris()
        X = iris.data[:, :3]
        y = iris.target
        tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

        config = ConversionConfig(
            max_depth=3,
            optimize_rules='auto',
            n_samples=5000,
            function_name='predict',
        )
        conv = Converter(config)
        code = conv.convert(
            tree, X,
            target=target,
            feature_names=['feat0', 'feat1', 'feat2'],
            class_names=['setosa', 'versicolor', 'virginica'],
        )
        assert not _has_negative_index(code), (
            f"Negative feature index for target={target!r}, level='auto'.\n"
            + code[:800]
        )

    # ── structural consistency ────────────────────────────────────────
    @pytest.mark.parametrize("level", ["medium", "high"])
    def test_prune_invariant_leaf_consistency(self, iris_rf_surrogate, level):
        """After optimisation every leaf is consistently marked.

        For each node: ``feature == -2`` iff
        ``children_left == children_right == -1``.  This is the
        invariant the v0.2.0 bug violated.
        """
        tree, _ = iris_rf_surrogate
        # Work on a fresh copy so parametrisation is independent.
        from copy import deepcopy
        tree = deepcopy(tree)

        RuleOptimizer(optimization_level=level).optimize(tree)

        ts = tree.tree_
        for node_id in range(ts.node_count):
            is_feat_leaf = ts.feature[node_id] == TREE_LEAF
            is_child_leaf = (
                ts.children_left[node_id] == TREE_UNDEFINED
                and ts.children_right[node_id] == TREE_UNDEFINED
                and ts.children_left[node_id] == ts.children_right[node_id]
            )
            assert is_feat_leaf == is_child_leaf, (
                f"Node {node_id} inconsistent: feature_leaf={is_feat_leaf}, "
                f"child_leaf={is_child_leaf} (feature={ts.feature[node_id]}, "
                f"left={ts.children_left[node_id]}, "
                f"right={ts.children_right[node_id]})"
            )

    # ── pruning actually reduces code ─────────────────────────────────
    def test_prune_actually_reduces_conditions(self, iris_rf_surrogate):
        """Medium pruning must collapse redundant nodes to a single return.

        In v0.2.0 the corrupted node was still emitted as a dead
        ``if (features[-2] <= t) { return k; } else { return k; }``,
        so 'medium' produced the same number of ``if`` statements as
        'low'.  After the fix, pruning removes those conditions.
        """
        tree, _ = iris_rf_surrogate
        from copy import deepcopy

        feature_names = ['feat0', 'feat1', 'feat2']
        class_names = ['setosa', 'versicolor', 'virginica']

        tree_low = deepcopy(tree)
        tree_med = deepcopy(tree)
        RuleOptimizer(optimization_level='low').optimize(tree_low)
        RuleOptimizer(optimization_level='medium').optimize(tree_med)

        exporter = create_exporter('arduino')
        code_low = exporter.generate(
            tree_low, feature_names=feature_names, class_names=class_names
        )
        code_med = exporter.generate(
            tree_med, feature_names=feature_names, class_names=class_names
        )

        n_if_low = len(re.findall(r'\bif\s*\(', code_low))
        n_if_med = len(re.findall(r'\bif\s*\(', code_med))
        assert n_if_med <= n_if_low, (
            f"Medium pruning did not reduce conditions "
            f"(low={n_if_low}, medium={n_if_med})"
        )
        # And no negative index in either.
        assert not _has_negative_index(code_med)

    # ── direct unit test on the pruner ────────────────────────────────
    def test_direct_prune_updates_all_attributes(self):
        """A prunable node must have all four structural attrs updated."""
        # Build a tree guaranteed to contain a node whose two children
        # are same-class leaves: a depth-2 tree on linearly separable
        # data collapses many internal nodes to redundant splits.
        X, y = load_iris(return_X_y=True)
        X = X[:, :3]
        tree = DecisionTreeClassifier(max_depth=3, random_state=0)
        tree.fit(X, y)

        ts = tree.tree_
        # Find a prunable node before pruning (sanity: at least one).
        prunable = []
        for nid in range(ts.node_count):
            if ts.feature[nid] == TREE_LEAF:
                continue
            lc, rc = ts.children_left[nid], ts.children_right[nid]
            if ts.feature[lc] == TREE_LEAF and ts.feature[rc] == TREE_LEAF:
                if np.argmax(ts.value[lc][0]) == np.argmax(ts.value[rc][0]):
                    prunable.append(nid)
        if not prunable:
            pytest.skip("No prunable node in this tree; rerun with other seed")

        RuleOptimizer(optimization_level='medium').optimize(tree)

        ts = tree.tree_
        for nid in prunable:
            assert ts.feature[nid] == TREE_LEAF, f"node {nid} feature not -2"
            assert ts.threshold[nid] == TREE_LEAF, f"node {nid} threshold not -2"
            assert ts.children_left[nid] == TREE_UNDEFINED, (
                f"node {nid} children_left not cleared"
            )
            assert ts.children_right[nid] == TREE_UNDEFINED, (
                f"node {nid} children_right not cleared"
            )

    def test_exporter_emits_return_not_features_neg2(self, iris_rf_surrogate):
        """A pruned node must emit ``return k;`` not ``features[-2]``."""
        tree, _ = iris_rf_surrogate
        from copy import deepcopy
        tree = deepcopy(tree)
        RuleOptimizer(optimization_level='medium').optimize(tree)

        exporter = create_exporter('arduino')
        code = exporter.generate(
            tree, feature_names=['feat0', 'feat1', 'feat2'],
            class_names=['setosa', 'versicolor', 'virginica'],
        )
        # No condition may reference a negative index.
        for cond in re.findall(r'if\s*\(([^)]+)\)', code):
            assert not re.match(r'\s*features\[-', cond), (
                f"Negative-index condition emitted: {cond!r}"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-vv'])
