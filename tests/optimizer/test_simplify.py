"""
Tests for :meth:`Conjunction.simplify` and :meth:`RuleSet.simplify`.

Two invariants are pinned by every test in this module:

1. **Semantic equivalence** — for any input ``X``, the simplified
   conjunction must produce the same boolean mask as the original
   (or the all-False mask if the original was unsatisfiable).
2. **Literal minimality** — the simplified conjunction must contain
   at most one ``<=`` and one ``>`` literal per feature index, and
   must drop redundant copies.
"""

from __future__ import annotations

import numpy as np
import pytest

from blackbox2c.optimizer.ir import Conjunction, Literal, RuleSet


# ─────────────────────────── Conjunction.simplify ───────────────────────
class TestConjunctionSimplify:
    def test_collapses_multiple_le_on_same_feature(self):
        c = Conjunction(
            literals=(
                Literal(0, 4.89, "<="),
                Literal(0, 1.44, "<="),
                Literal(0, 2.50, "<="),
            ),
            prediction=1,
        )
        s = c.simplify()
        assert s is not None
        assert s.literals == (Literal(0, 1.44, "<="),)
        assert s.prediction == 1

    def test_collapses_multiple_gt_on_same_feature(self):
        c = Conjunction(
            literals=(
                Literal(2, 1.0, ">"),
                Literal(2, 5.0, ">"),
                Literal(2, 3.0, ">"),
            ),
            prediction=0,
        )
        s = c.simplify()
        assert s is not None
        assert s.literals == (Literal(2, 5.0, ">"),)

    def test_keeps_one_le_and_one_gt_as_interval(self):
        c = Conjunction(
            literals=(
                Literal(1, 3.0, ">"),
                Literal(1, 5.0, "<="),
            ),
            prediction=0,
        )
        s = c.simplify()
        assert s is not None
        # Conventionally we emit '>' before '<=' for the same feature.
        assert s.literals == (
            Literal(1, 3.0, ">"),
            Literal(1, 5.0, "<="),
        )

    def test_empty_interval_returns_none(self):
        # x > 5 AND x <= 5  →  empty interval (left-open, right-closed
        # semantics: the two constraints share no point).
        c = Conjunction(
            literals=(
                Literal(0, 5.0, ">"),
                Literal(0, 5.0, "<="),
            ),
            prediction=0,
        )
        assert c.simplify() is None

    def test_inverted_interval_returns_none(self):
        # x > 7 AND x <= 3  →  trivially empty.
        c = Conjunction(
            literals=(
                Literal(0, 7.0, ">"),
                Literal(0, 3.0, "<="),
            ),
            prediction=1,
        )
        assert c.simplify() is None

    def test_no_op_when_already_minimal(self):
        c = Conjunction(
            literals=(
                Literal(0, 1.0, "<="),
                Literal(2, 3.5, ">"),
            ),
            prediction=2,
        )
        s = c.simplify()
        assert s is not None
        assert set(s.literals) == set(c.literals)

    def test_empty_conjunction_simplifies_to_empty(self):
        c = Conjunction(literals=(), prediction=0)
        s = c.simplify()
        assert s is not None
        assert s.literals == ()
        assert s.prediction == 0

    def test_semantic_equivalence_random(self):
        """For non-empty intervals the mask must agree on a dense grid."""
        c = Conjunction(
            literals=(
                Literal(0, 3.0, ">"),
                Literal(0, 5.0, "<="),
                Literal(0, 4.5, "<="),  # redundant — tighter than 5.0
                Literal(1, 0.0, ">"),
                Literal(0, 2.0, ">"),   # redundant — looser than 3.0
            ),
            prediction=1,
        )
        s = c.simplify()
        assert s is not None

        rng = np.random.default_rng(42)
        X = rng.uniform(-5, 10, size=(2_000, 2))
        np.testing.assert_array_equal(c.evaluate(X), s.evaluate(X))


# ─────────────────────────── RuleSet.simplify ──────────────────────────
class TestRuleSetSimplify:
    def test_drops_unsat_rules_and_keeps_others(self):
        rules = (
            Conjunction(
                literals=(
                    Literal(0, 0.0, ">"),
                    Literal(0, -1.0, "<="),  # unsatisfiable
                ),
                prediction=0,
            ),
            Conjunction(
                literals=(
                    Literal(0, 0.0, "<="),
                    Literal(0, 5.0, "<="),  # redundant
                ),
                prediction=1,
            ),
            Conjunction(literals=(), prediction=0),
        )
        rs = RuleSet(rules=rules, n_features=1, n_classes=2)
        out = rs.simplify()
        assert len(out.rules) == 2
        # First rule's redundant '<= 5.0' should be gone.
        assert out.rules[0].literals == (Literal(0, 0.0, "<="),)
        # The unconditional fallback is preserved.
        assert out.rules[1].literals == ()

    def test_simplify_preserves_predictions_on_random_inputs(self):
        rules = (
            Conjunction(
                literals=(
                    Literal(0, 0.0, ">"),
                    Literal(0, 5.0, "<="),
                    Literal(0, 4.0, "<="),  # redundant
                    Literal(1, -1.0, ">"),  # always true on rng range
                ),
                prediction=1,
            ),
            Conjunction(literals=(Literal(0, 0.0, "<="),), prediction=0),
            Conjunction(literals=(), prediction=2),
        )
        rs = RuleSet(rules=rules, n_features=2, n_classes=3)
        out = rs.simplify()

        rng = np.random.default_rng(0)
        X = rng.uniform(-2, 8, size=(2_000, 2))
        np.testing.assert_array_equal(rs.predict(X), out.predict(X))

    def test_total_literals_can_only_decrease(self):
        rules = (
            Conjunction(
                literals=(
                    Literal(0, 0.0, ">"),
                    Literal(0, 5.0, "<="),
                    Literal(0, 4.0, "<="),
                    Literal(0, 3.0, "<="),
                    Literal(2, 1.0, "<="),
                ),
                prediction=0,
            ),
            Conjunction(literals=(Literal(0, 0.0, "<="),), prediction=1),
        )
        rs = RuleSet(rules=rules, n_features=3, n_classes=2)
        before = sum(len(r.literals) for r in rs.rules)
        after = sum(len(r.literals) for r in rs.simplify().rules)
        assert after < before
