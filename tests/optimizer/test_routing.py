"""
Tests for the optimizer-level router.

Routing is the only place where the public ``optimize_rules`` string
becomes a concrete algorithm choice, so the tests here pin the
contract: which level dispatches to which optimizer, and how
fallbacks (regression / over-cap) are signalled.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from blackbox2c.optimizer import routing
from blackbox2c.optimizer.extraction import from_sklearn_tree
from blackbox2c.optimizer.ir import Conjunction, Literal, RuleSet


# ───────────────────────── small fixtures ──────────────────────────
def _real_clf_rs(max_depth: int = 2):
    """Build a real, *complete* RuleSet extracted from a sklearn tree on Iris.

    Returns ``(rs, X_train)``; ``X_train`` is used to derive sampling
    ranges for the equivalence assertions.
    """
    X, y = load_iris(return_X_y=True)
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42).fit(X, y)
    return from_sklearn_tree(tree), X


def _eval_grid(X: np.ndarray, n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(0)
    lo, hi = X.min(axis=0), X.max(axis=0)
    return rng.uniform(lo, hi, size=(n, X.shape[1]))


def _toy_reg_rs() -> RuleSet:
    return RuleSet(
        rules=(Conjunction((), prediction=1.5),),
        n_features=1,
        n_classes=None,
    )


# ─────────────────────────── public surface ────────────────────────
class TestPublicSurface:
    def test_invalid_level_raises_with_message(self):
        rs, _ = _real_clf_rs()
        with pytest.raises(ValueError, match="Unknown optimize_rules"):
            routing.optimize_ruleset(rs, level="bogus")

    def test_legacy_levels_rejected_at_router(self):
        # The Converter is responsible for not routing legacy levels
        # through the IR pipeline; calling them here is a programmer
        # error and must be loud.
        rs, _ = _real_clf_rs()
        for lvl in ("medium", "high"):
            with pytest.raises(ValueError, match="legacy"):
                routing.optimize_ruleset(rs, level=lvl)

    def test_low_is_passthrough(self):
        rs, _ = _real_clf_rs()
        out = routing.optimize_ruleset(rs, level="low")
        assert out is rs

    def test_is_advanced_level(self):
        assert routing.is_advanced_level("qm")
        assert routing.is_advanced_level("bdd")
        assert routing.is_advanced_level("auto")
        assert not routing.is_advanced_level("low")
        assert not routing.is_advanced_level("medium")
        assert not routing.is_advanced_level("high")


# ─────────────────────────── direct dispatch ───────────────────────
class TestDirectDispatch:
    def test_qm_runs_when_under_cap(self):
        rs, X = _real_clf_rs(max_depth=2)
        out = routing.optimize_ruleset(rs, level="qm", qm_max_literals=12)
        X_eval = _eval_grid(X)
        assert np.array_equal(out.predict(X_eval), rs.predict(X_eval))

    def test_bdd_runs_when_under_cap(self):
        rs, X = _real_clf_rs(max_depth=2)
        out = routing.optimize_ruleset(rs, level="bdd", bdd_max_literals=24)
        X_eval = _eval_grid(X)
        assert np.array_equal(out.predict(X_eval), rs.predict(X_eval))


# ─────────────────────────── auto routing ──────────────────────────
class TestAutoRouting:
    def test_auto_picks_qm_under_qm_cap(self):
        rs, X = _real_clf_rs(max_depth=2)
        out = routing.optimize_ruleset(
            rs, level="auto", qm_max_literals=12, bdd_max_literals=24,
        )
        X_eval = _eval_grid(X)
        assert np.array_equal(out.predict(X_eval), rs.predict(X_eval))

    def test_auto_falls_back_to_bdd_above_qm_cap(self):
        rs, X = _real_clf_rs(max_depth=2)
        n_lits = len(rs.unique_literals())
        # qm_max_literals just below n_lits → QM rejected, BDD accepts.
        out = routing.optimize_ruleset(
            rs,
            level="auto",
            qm_max_literals=max(0, n_lits - 1),
            bdd_max_literals=n_lits + 4,
        )
        X_eval = _eval_grid(X)
        assert np.array_equal(out.predict(X_eval), rs.predict(X_eval))

    def test_auto_warns_when_above_both_caps(self):
        rs, _ = _real_clf_rs(max_depth=2)
        with pytest.warns(UserWarning, match="could not pick"):
            out = routing.optimize_ruleset(
                rs, level="auto", qm_max_literals=0, bdd_max_literals=0,
            )
        assert out is rs


# ─────────────────────── regression fallback ───────────────────────
class TestRegressionFallback:
    @pytest.mark.parametrize("level", ["qm", "bdd", "auto"])
    def test_regression_warns_and_returns_unchanged(self, level):
        rs = _toy_reg_rs()
        with pytest.warns(UserWarning, match="classification"):
            out = routing.optimize_ruleset(rs, level=level)
        assert out is rs


# ─────────────────────── interaction with caps ─────────────────────
class TestCapsForwarding:
    def test_qm_max_literals_is_forwarded(self):
        # qm_max_literals=0 forces every non-trivial RuleSet to fall back.
        rs, _ = _real_clf_rs(max_depth=2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = routing.optimize_ruleset(rs, level="qm", qm_max_literals=0)
        assert any("literals" in str(rec.message).lower() for rec in w)
        assert out is rs

    def test_bdd_max_literals_is_forwarded(self):
        rs, _ = _real_clf_rs(max_depth=2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = routing.optimize_ruleset(rs, level="bdd", bdd_max_literals=0)
        assert any("literals" in str(rec.message).lower() for rec in w)
        assert out is rs
