"""
Tests for the ROBDD-based optimizer.

Coverage:
- Internal ``_BDD`` data-structure: unique-table identity, child
  collapse rule, true-path enumeration.
- ``_build_var_order`` deterministic frequency order.
- ``BDDOptimizer.minimize`` equivalence on Boolean truth tables and on
  real sklearn trees, with a property-based suite for breadth.
- Fallback paths (regression, max_literals) emit a warning and return
  the input unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.tree import DecisionTreeClassifier

from blackbox2c.optimizer.bdd import (
    BDDOptimizer,
    _BDD,
    _TERMINAL_FALSE,
    _TERMINAL_TRUE,
    _build_var_order,
)
from blackbox2c.optimizer.extraction import from_sklearn_tree
from blackbox2c.optimizer.ir import Conjunction, Literal, RuleSet


# ───────────────────────── _BDD primitives ────────────────────────────
class TestBDDPrimitives:
    def test_collapse_when_children_equal(self):
        bdd = _BDD(var_order=((0, 0.0),))
        node = bdd._make(0, _TERMINAL_TRUE, _TERMINAL_TRUE)
        assert node == _TERMINAL_TRUE
        assert bdd.n_internal_nodes == 0

    def test_unique_table_returns_same_id(self):
        bdd = _BDD(var_order=((0, 0.0),))
        a = bdd._make(0, _TERMINAL_FALSE, _TERMINAL_TRUE)
        b = bdd._make(0, _TERMINAL_FALSE, _TERMINAL_TRUE)
        assert a == b
        assert bdd.n_internal_nodes == 1

    def test_build_constant_true(self):
        bdd = _BDD(var_order=((0, 0.0), (1, 0.0)))
        root = bdd.build(lambda assignment: True)
        assert root == _TERMINAL_TRUE
        assert bdd.n_internal_nodes == 0

    def test_build_constant_false(self):
        bdd = _BDD(var_order=((0, 0.0),))
        root = bdd.build(lambda assignment: False)
        assert root == _TERMINAL_FALSE

    def test_build_xor_two_vars_canonical_size(self):
        var_order = ((0, 0.0), (1, 0.0))
        bdd = _BDD(var_order=var_order)
        root = bdd.build(
            lambda a: a[(0, 0.0)] ^ a[(1, 0.0)]
        )
        # Canonical ROBDD of XOR(a,b) has exactly 3 internal nodes:
        # the root on var ``a``, plus two distinct ``b`` nodes (b vs
        # ~b) — they cannot share storage because their children are
        # swapped.  This 3-node count is the textbook fingerprint.
        assert bdd.n_internal_nodes == 3
        # And exactly two true-paths.
        paths = bdd.true_paths(root)
        assert len(paths) == 2

    def test_true_paths_constant(self):
        bdd = _BDD(var_order=())
        assert bdd.true_paths(_TERMINAL_TRUE) == [{}]
        assert bdd.true_paths(_TERMINAL_FALSE) == []


# ─────────────────────────── var ordering ─────────────────────────────
class TestVarOrder:
    def test_frequency_first_then_feature_then_threshold(self):
        rs = RuleSet(
            rules=(
                Conjunction(
                    (Literal(0, 1.0, "<="), Literal(1, 2.0, "<=")), 0
                ),
                Conjunction((Literal(0, 1.0, ">"),), 1),
                Conjunction((Literal(0, 1.0, "<="), Literal(2, 3.0, ">")), 0),
            ),
            n_features=3,
            n_classes=2,
        )
        order = _build_var_order(rs)
        # var (0, 1.0) appears 3 times (counted via canonicalisation),
        # (1, 2.0) once, (2, 3.0) once.  Tie between (1, 2.0) and
        # (2, 3.0) is resolved by feature_idx.
        assert order[0] == (0, 1.0)
        assert order[1] == (1, 2.0)
        assert order[2] == (2, 3.0)


# ─────────────────────────── boolean tables ───────────────────────────
def _ruleset_from_truth_table(truth: dict, n_features: int) -> RuleSet:
    """Boolean truth-table → RuleSet using ``x > 0`` as the True signal."""
    rules = []
    for bits, label in truth.items():
        lits = [
            Literal(i, 0.0, "<=" if b == 0 else ">")
            for i, b in enumerate(bits)
        ]
        rules.append(Conjunction(tuple(lits), int(label)))
    classes = max(truth.values()) + 1
    return RuleSet(tuple(rules), n_features=n_features, n_classes=classes)


class TestBooleanReductions:
    def test_xor_equivalence(self):
        truth = {
            (0, 0): 0, (0, 1): 1,
            (1, 0): 1, (1, 1): 0,
        }
        rs = _ruleset_from_truth_table(truth, 2)
        opt = BDDOptimizer().minimize(rs)
        rng = np.random.default_rng(0)
        X = rng.uniform(-2.0, 2.0, size=(4_000, 2))
        assert np.array_equal(opt.predict(X), rs.predict(X))

    def test_majority3_equivalence(self):
        truth = {}
        for bits in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]:
            truth[bits] = int(sum(bits) >= 2)
        rs = _ruleset_from_truth_table(truth, 3)
        opt = BDDOptimizer().minimize(rs)
        rng = np.random.default_rng(1)
        X = rng.uniform(-2.0, 2.0, size=(4_000, 3))
        assert np.array_equal(opt.predict(X), rs.predict(X))


# ─────────────── equivalence on real sklearn trees ────────────────────
@pytest.mark.parametrize(
    "loader, depth",
    [
        (load_iris, 2),
        (load_iris, 3),
        (load_wine, 3),
    ],
)
def test_bdd_equivalence_on_sklearn_trees(loader, depth):
    X, y = loader(return_X_y=True)
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X, y)
    rs = from_sklearn_tree(tree)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        opt = BDDOptimizer(max_literals=24).minimize(rs)

    rng = np.random.default_rng(42)
    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = 0.05 * (hi - lo)
    X_eval = rng.uniform(lo - pad, hi + pad, size=(5_000, X.shape[1]))

    assert np.array_equal(opt.predict(X_eval), tree.predict(X_eval))


# ─────────────────────────── property-based ───────────────────────────
@settings(
    deadline=None,
    max_examples=12,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    n_features=st.integers(min_value=2, max_value=4),
    n_classes=st.integers(min_value=2, max_value=3),
    max_depth=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=2**16 - 1),
)
def test_property_bdd_preserves_predictions(
    n_features, n_classes, max_depth, seed
):
    n_informative = min(n_features, max(2, n_classes))
    X, y = make_classification(
        n_samples=80,
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        opt = BDDOptimizer(max_literals=20).minimize(rs)

    rng = np.random.default_rng(seed + 1)
    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = 0.1 * (hi - lo + 1e-9)
    X_eval = rng.uniform(lo - pad, hi + pad, size=(1_000, n_features))

    assert np.array_equal(opt.predict(X_eval), tree.predict(X_eval))


# ─────────────────────────── fallback behaviour ───────────────────────
class TestFallback:
    def test_regression_warns_and_returns_unchanged(self):
        rs = RuleSet(
            rules=(Conjunction((), prediction=1.5),),
            n_features=1,
            n_classes=None,
        )
        with pytest.warns(UserWarning, match="classification"):
            out = BDDOptimizer().minimize(rs)
        assert out is rs

    def test_max_literals_fallback(self):
        rules = []
        for i in range(8):
            lits = (
                Literal(0, float(i), "<="),
                Literal(1, float(i), "<="),
            )
            rules.append(Conjunction(lits, 0))
        rules.append(Conjunction((), 1))
        rs = RuleSet(tuple(rules), n_features=2, n_classes=2)
        with pytest.warns(UserWarning, match="literals"):
            opt = BDDOptimizer(max_literals=4).minimize(rs)
        assert opt is rs

    def test_single_leaf_is_noop(self):
        rs = RuleSet(
            rules=(Conjunction((), prediction=0),),
            n_features=2,
            n_classes=2,
        )
        opt = BDDOptimizer().minimize(rs)
        assert opt is rs
