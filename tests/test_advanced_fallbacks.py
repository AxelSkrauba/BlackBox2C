"""Regression tests for safe advanced-optimizer fallbacks and code-size metrics."""

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from blackbox2c import ConversionConfig, Converter
from blackbox2c.codegen import CCodeGenerator
from blackbox2c.codegen_bridge import (
    RuleSetCodeGenerator,
    RuleSetExpansionLimitError,
)
from blackbox2c.optimizer import routing
from blackbox2c.optimizer.extraction import from_sklearn_tree
from blackbox2c.optimizer.ir import Conjunction, Literal, RuleSet
from blackbox2c.tree_constants import TREE_LEAF, TREE_UNDEFINED


def _tree_ruleset(max_depth=3):
    X, y = load_iris(return_X_y=True)
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42).fit(X, y)
    return tree, X, from_sklearn_tree(tree)


def _expanding_ruleset():
    rules = []
    for feature_idx in range(8):
        rules.append(
            Conjunction(
                (Literal(feature_idx % 2, float(feature_idx), "<="),),
                feature_idx % 2,
            )
        )
    return RuleSet(tuple(rules), n_features=2, n_classes=2)


def test_auto_above_both_caps_skips_bridge_estimation(monkeypatch):
    _, _, ruleset = _tree_ruleset()

    def fail_if_called(_):
        raise AssertionError("_estimate_size must not run when both caps reject")

    monkeypatch.setattr(routing, "_estimate_size", fail_if_called)

    with pytest.warns(UserWarning, match="could not pick"):
        result = routing.optimize_ruleset(
            ruleset,
            level="auto",
            qm_max_literals=0,
            bdd_max_literals=0,
        )

    assert result is ruleset


def test_ruleset_bridge_stops_before_unbounded_expansion():
    generator = RuleSetCodeGenerator(max_bridge_nodes=4)

    with pytest.raises(RuleSetExpansionLimitError):
        generator.estimate_code_size_from_ruleset(_expanding_ruleset())


@pytest.mark.parametrize("level", ["qm", "bdd", "auto"])
def test_advanced_noop_uses_legacy_codegen_and_metrics(level):
    tree, X, _ = _tree_ruleset()
    config = ConversionConfig(
        max_depth=3,
        optimize_rules=level,
        qm_max_literals=0,
        bdd_max_literals=0,
    )
    converter = Converter(config)

    with pytest.warns(UserWarning):
        code = converter.convert(
            tree,
            X,
            feature_names=[f"f{i}" for i in range(X.shape[1])],
            class_names=["A", "B", "C"],
        )

    assert converter.optimized_ruleset_ is None
    assert converter.metrics_["advanced_optimizer_applied"] is False
    assert converter.metrics_["advanced_optimizer_fallback"] is True
    assert converter.metrics_["size_estimate"] == CCodeGenerator().estimate_code_size(
        converter.surrogate_tree_
    )
    assert "uint8_t predict(" in code


def test_legacy_size_estimate_counts_only_reachable_nodes():
    tree = SimpleNamespace(
        tree_=SimpleNamespace(
            node_count=3,
            feature=np.array([TREE_LEAF, TREE_LEAF, TREE_LEAF]),
            threshold=np.array([TREE_LEAF, TREE_LEAF, TREE_LEAF]),
            children_left=np.array(
                [TREE_UNDEFINED, TREE_UNDEFINED, TREE_UNDEFINED]
            ),
            children_right=np.array(
                [TREE_UNDEFINED, TREE_UNDEFINED, TREE_UNDEFINED]
            ),
        )
    )

    estimate = CCodeGenerator().estimate_code_size(tree)

    assert estimate == {
        "flash_bytes": 54,
        "ram_bytes": 32,
        "n_nodes": 1,
        "n_conditions": 0,
        "n_leaves": 1,
    }
