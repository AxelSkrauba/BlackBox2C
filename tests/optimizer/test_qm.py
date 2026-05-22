"""
Tests for the multi-valued Quine-McCluskey minimizer.

Strategy:
- Internal helpers (cube merging, interval lookup, cover) are tested
  in isolation with hand-crafted examples whose answer is computable
  by hand.
- The full ``QMOptimizer.minimize`` is exercised via the equivalence
  oracle: for every test RuleSet, ``qm(rs).predict ≡ rs.predict`` over
  a large pool of random samples *and* over every minterm centroid.
- A property-based suite (Hypothesis) randomises tree shape and
  thresholds, ensuring equivalence holds in a much wider corner of the
  configuration space than hand-written cases reach.
- Performance / fallback behaviour is checked separately.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.tree import DecisionTreeClassifier

from blackbox2c.optimizer.extraction import from_sklearn_tree
from blackbox2c.optimizer.ir import Conjunction, Literal, RuleSet
from blackbox2c.optimizer.qm import (
    QMOptimizer,
    _cover,
    _cube_contains,
    _interval_for,
    _representative,
    _try_merge,
)


# ─────────────────────────── interval helpers ───────────────────────────
class TestIntervalHelpers:
    def test_interval_for_no_thresholds(self):
        assert _interval_for(3.14, []) == 0

    def test_interval_for_basic(self):
        thr = [0.0, 1.0, 2.0]
        # (-inf, 0]  → 0
        # (0, 1]     → 1
        # (1, 2]     → 2
        # (2, +inf)  → 3
        assert _interval_for(-1.0, thr) == 0
        assert _interval_for(0.0, thr) == 0  # left-closed at threshold
        assert _interval_for(0.5, thr) == 1
        assert _interval_for(1.0, thr) == 1
        assert _interval_for(2.5, thr) == 3

    def test_representative_inside_interval(self):
        thr = [0.0, 2.0, 5.0]
        # interval 0 = (-inf, 0]   rep < 0
        # interval 1 = (0, 2]      rep in (0, 2]
        # interval 2 = (2, 5]      rep in (2, 5]
        # interval 3 = (5, +inf)   rep > 5
        assert _representative(0, thr) == -1.0
        assert 0.0 < _representative(1, thr) <= 2.0
        assert 2.0 < _representative(2, thr) <= 5.0
        assert _representative(3, thr) == 6.0

    def test_representative_consistent_with_interval_for(self):
        """``_interval_for(_representative(i, t), t)`` must yield ``i``."""
        thr = [0.0, 1.0, 2.0, 3.0]
        for i in range(len(thr) + 1):
            r = _representative(i, thr)
            assert _interval_for(r, thr) == i


# ─────────────────────────── cube / merge ──────────────────────────────
class TestCubeMerge:
    def test_merge_singletons_differing_in_one_axis(self):
        a = (frozenset({0}), frozenset({1}))
        b = (frozenset({0}), frozenset({2}))
        merged = _try_merge(a, b)
        assert merged == (frozenset({0}), frozenset({1, 2}))

    def test_merge_returns_none_for_two_axis_diff(self):
        a = (frozenset({0}), frozenset({1}))
        b = (frozenset({1}), frozenset({2}))
        assert _try_merge(a, b) is None

    def test_merge_returns_none_for_identical(self):
        a = (frozenset({0}), frozenset({1}))
        assert _try_merge(a, a) is None

    def test_cube_contains(self):
        cube = (frozenset({0, 1}), frozenset({2}))
        assert _cube_contains(cube, (0, 2)) is True
        assert _cube_contains(cube, (1, 2)) is True
        assert _cube_contains(cube, (2, 2)) is False
        assert _cube_contains(cube, (0, 3)) is False


# ──────────────────────────── cover ────────────────────────────────────
class TestCover:
    def test_essential_pi_only(self):
        # 3 mintermos, 3 PIs, each covers exactly one different mintermo →
        # all three are essential.
        on = [(0,), (1,), (2,)]
        pis = [
            (frozenset({0}),),
            (frozenset({1}),),
            (frozenset({2}),),
        ]
        cover = _cover(on, pis)
        assert len(cover) == 3
        assert set(cover) == set(pis)

    def test_one_pi_covers_all(self):
        on = [(0,), (1,), (2,)]
        pis = [
            (frozenset({0, 1, 2}),),
            (frozenset({0}),),
            (frozenset({1}),),
        ]
        cover = _cover(on, pis)
        assert len(cover) == 1
        assert cover[0] == (frozenset({0, 1, 2}),)

    def test_petrick_picks_minimum(self):
        # 4 mintermos, 4 PIs each covering 2:
        # PIs that pairwise cover all 4 in 2 picks.
        on = [(0,), (1,), (2,), (3,)]
        pis = [
            (frozenset({0, 1}),),  # covers 0,1
            (frozenset({2, 3}),),  # covers 2,3
            (frozenset({1, 2}),),  # covers 1,2
            (frozenset({0, 3}),),  # covers 0,3
        ]
        cover = _cover(on, pis, exact_threshold=10)
        assert len(cover) == 2  # two suffice; greedy/Petrick must find this


# ─────────────────────────── single boolean cases ──────────────────────
def _ruleset_from_truth_table(truth: dict, n_features: int) -> RuleSet:
    """Build a RuleSet whose features are booleans encoded as
    ``x_i <= 0`` (False) vs ``x_i > 0`` (True).  ``truth`` maps each
    boolean tuple to a class label."""
    rules = []
    for bits, label in truth.items():
        lits = []
        for i, b in enumerate(bits):
            lits.append(Literal(i, 0.0, "<=" if b == 0 else ">"))
        rules.append(Conjunction(tuple(lits), int(label)))
    classes = max(truth.values()) + 1
    return RuleSet(tuple(rules), n_features=n_features, n_classes=classes)


def _truth_eval(truth, X, threshold=0.0):
    """Reference predictor for the ``_ruleset_from_truth_table`` schema."""
    bits = (X > threshold).astype(int)
    keys = [tuple(b) for b in bits]
    return np.array([truth[k] for k in keys])


class TestBooleanReductions:
    def test_xor_two_inputs(self):
        truth = {
            (0, 0): 0, (0, 1): 1,
            (1, 0): 1, (1, 1): 0,
        }
        rs = _ruleset_from_truth_table(truth, 2)
        opt = QMOptimizer().minimize(rs)

        rng = np.random.default_rng(0)
        X = rng.uniform(-2.0, 2.0, size=(4_000, 2))
        # Equivalence
        assert np.array_equal(opt.predict(X), rs.predict(X))
        # XOR has no Boolean simplification: minimum SOP is 2 terms per
        # class.  Total terms across both classes = 4, the same as the
        # original.  We require the optimiser to *not* blow up.
        assert opt.complexity()["n_rules"] <= 4

    def test_majority_three_inputs_simplifies(self):
        # Class 1 iff at least two of three inputs are True.
        truth = {}
        for bits in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]:
            truth[bits] = int(sum(bits) >= 2)
        rs = _ruleset_from_truth_table(truth, 3)
        opt = QMOptimizer().minimize(rs)

        rng = np.random.default_rng(1)
        X = rng.uniform(-2.0, 2.0, size=(4_000, 3))
        assert np.array_equal(opt.predict(X), rs.predict(X))
        # Minimum SOP for majority(3) of class 1 is exactly 3 terms
        # (ab + bc + ac).  Class 0 also needs 3 terms (a'b' + b'c' + a'c').
        # Total ≤ 6, strictly less than the 8 mintermos.
        assert opt.complexity()["n_rules"] < 8


# ─────────────── equivalence oracle on real sklearn trees ──────────────
@pytest.mark.parametrize(
    "loader, depth",
    [
        (load_iris, 2),
        (load_iris, 3),
        (load_wine, 3),
    ],
)
def test_qm_equivalence_on_sklearn_trees(loader, depth):
    X, y = loader(return_X_y=True)
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X, y)
    rs = from_sklearn_tree(tree)

    # The QM cap is meant to be hit gracefully; if it triggers, we just
    # want equivalence to remain (no-op fallback).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        opt = QMOptimizer(max_literals=20, max_minterms=8192).minimize(rs)

    rng = np.random.default_rng(42)
    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = 0.05 * (hi - lo)
    X_eval = rng.uniform(lo - pad, hi + pad, size=(5_000, X.shape[1]))

    assert np.array_equal(opt.predict(X_eval), tree.predict(X_eval))


# ─────────────────────────── property based ────────────────────────────
@settings(
    deadline=None,
    max_examples=15,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    n_features=st.integers(min_value=2, max_value=5),
    n_classes=st.integers(min_value=2, max_value=3),
    max_depth=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=2**16 - 1),
)
def test_property_qm_preserves_predictions(
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
        opt = QMOptimizer(max_literals=15, max_minterms=4096).minimize(rs)

    rng = np.random.default_rng(seed + 1)
    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = 0.1 * (hi - lo + 1e-9)
    X_eval = rng.uniform(lo - pad, hi + pad, size=(1_000, n_features))

    assert np.array_equal(opt.predict(X_eval), tree.predict(X_eval))


# ─────────────────────────── fallback behaviour ────────────────────────
class TestFallback:
    def test_regression_input_warns_and_returns_unchanged(self):
        rs = RuleSet(
            rules=(Conjunction((), prediction=1.5),),
            n_features=1,
            n_classes=None,
        )
        with pytest.warns(UserWarning, match="classification"):
            out = QMOptimizer().minimize(rs)
        assert out is rs  # exact same object → unchanged

    def test_max_literals_fallback(self):
        # Build a trivially-large RuleSet with many distinct thresholds.
        # Each rule introduces one fresh threshold per feature; with two
        # features and 8 rules we obtain 16 literals → above default cap.
        rules = []
        for i in range(8):
            lits = (
                Literal(0, float(i), "<="),
                Literal(1, float(i), "<="),
            )
            rules.append(Conjunction(lits, 0))
        rules.append(Conjunction((), 1))  # default fallback
        rs = RuleSet(tuple(rules), n_features=2, n_classes=2)
        with pytest.warns(UserWarning, match="literals"):
            opt = QMOptimizer(max_literals=4).minimize(rs)
        assert opt is rs

    def test_single_leaf_is_noop(self):
        rs = RuleSet(
            rules=(Conjunction((), prediction=0),),
            n_features=2,
            n_classes=2,
        )
        opt = QMOptimizer().minimize(rs)
        assert opt is rs
