"""
Unit tests for the optimizer IR (Literal, Conjunction, RuleSet).

These tests do not depend on scikit-learn; the IR is exercised in
isolation so any breakage is localised to ``blackbox2c.optimizer.ir``.
"""

import numpy as np
import pytest

from blackbox2c.optimizer.ir import Conjunction, Literal, RuleSet


# ───────────────────────────── Literal ──────────────────────────────
class TestLiteral:
    def test_evaluate_le(self):
        lit = Literal(0, 1.5, "<=")
        X = np.array([[1.0], [1.5], [2.0]])
        assert lit.evaluate(X).tolist() == [True, True, False]

    def test_evaluate_gt(self):
        lit = Literal(0, 1.5, ">")
        X = np.array([[1.0], [1.5], [2.0]])
        assert lit.evaluate(X).tolist() == [False, False, True]

    def test_negate_roundtrip(self):
        lit = Literal(2, 3.14, "<=")
        assert lit.negate().negate() == lit

    def test_invalid_op_rejected(self):
        with pytest.raises(ValueError, match="Literal.op"):
            Literal(0, 0.0, "<")

    def test_negative_feature_rejected(self):
        with pytest.raises(ValueError):
            Literal(-1, 0.0, "<=")

    def test_is_hashable_and_immutable(self):
        lit = Literal(0, 1.0, "<=")
        # Frozen dataclasses are hashable by default and refuse mutation.
        {lit}  # noqa: B018 — exercising hashability is the assertion
        with pytest.raises(Exception):
            lit.threshold = 2.0  # type: ignore[misc]


# ─────────────────────────── Conjunction ────────────────────────────
class TestConjunction:
    def test_empty_conjunction_always_true(self):
        conj = Conjunction(literals=(), prediction=7)
        X = np.random.default_rng(0).standard_normal((20, 3))
        assert conj.evaluate(X).all()

    def test_and_semantics(self):
        a = Literal(0, 0.0, ">")
        b = Literal(1, 1.0, "<=")
        conj = Conjunction((a, b), prediction=1)
        X = np.array([
            [+0.5, +0.5],   # a True, b True  -> True
            [-0.5, +0.5],   # a False         -> False
            [+0.5, +1.5],   # b False         -> False
        ])
        assert conj.evaluate(X).tolist() == [True, False, False]


# ───────────────────────────── RuleSet ──────────────────────────────
def _toy_clf_ruleset():
    """A 2-feature, 2-class RuleSet equivalent to::

        if x0 <= 0.5: 0
        elif x1 <= 0.5: 1
        else:          0
    """
    rules = (
        Conjunction((Literal(0, 0.5, "<="),), prediction=0),
        Conjunction((Literal(0, 0.5, ">"), Literal(1, 0.5, "<=")), prediction=1),
        Conjunction((Literal(0, 0.5, ">"), Literal(1, 0.5, ">")), prediction=0),
    )
    return RuleSet(rules, n_features=2, n_classes=2)


class TestRuleSet:
    def test_predict_classification(self):
        rs = _toy_clf_ruleset()
        X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        assert rs.predict(X).tolist() == [0, 1, 0]

    def test_predict_returns_int_for_classification(self):
        rs = _toy_clf_ruleset()
        out = rs.predict(np.array([[0.0, 0.0]]))
        assert np.issubdtype(out.dtype, np.integer)

    def test_predict_returns_float_for_regression(self):
        rs = RuleSet(
            rules=(Conjunction((), prediction=3.14),),
            n_features=1,
            n_classes=None,
        )
        out = rs.predict(np.array([[0.0]]))
        assert np.issubdtype(out.dtype, np.floating)
        assert out[0] == pytest.approx(3.14)

    def test_predict_1d_input_promoted_to_2d(self):
        rs = _toy_clf_ruleset()
        out = rs.predict(np.array([0.0, 0.0]))
        assert out.shape == (1,)
        assert int(out[0]) == 0

    def test_incomplete_ruleset_raises(self):
        # Single rule that only matches x0<=0; nothing else -> incomplete.
        rs = RuleSet(
            rules=(Conjunction((Literal(0, 0.0, "<="),), prediction=1),),
            n_features=1,
            n_classes=2,
        )
        with pytest.raises(ValueError, match="incomplete"):
            rs.predict(np.array([[1.0]]))

    def test_feature_count_mismatch_raises(self):
        rs = _toy_clf_ruleset()
        with pytest.raises(ValueError, match="Expected 2 features"):
            rs.predict(np.array([[1.0]]))

    def test_feature_names_length_validated(self):
        with pytest.raises(ValueError, match="feature_names"):
            RuleSet(
                rules=(Conjunction((), prediction=0),),
                n_features=2,
                feature_names=("only_one",),
            )

    def test_complexity_summary(self):
        rs = _toy_clf_ruleset()
        c = rs.complexity()
        assert c["n_rules"] == 3
        assert c["total_literals"] == 1 + 2 + 2
        assert c["max_depth"] == 2

    def test_unique_literals_canonicalises_negations(self):
        # > and <= on the same (feature, threshold) collapse to one.
        rs = RuleSet(
            rules=(
                Conjunction((Literal(0, 0.5, "<="),), prediction=0),
                Conjunction((Literal(0, 0.5, ">"),), prediction=1),
            ),
            n_features=1,
            n_classes=2,
        )
        uniq = rs.unique_literals()
        assert len(uniq) == 1
        assert uniq[0].op == "<="
        assert uniq[0].threshold == 0.5
