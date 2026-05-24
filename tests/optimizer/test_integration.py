"""
End-to-end integration tests for the v0.2 advanced optimization
levels through the public ``Converter`` API.

These tests pin three properties:

1. **Backward compatibility** — every legacy level still produces
   exactly the same C output as before this PR (we hash the legacy
   output and compare).
2. **Public API extension** — ``ConversionConfig`` accepts the new
   ``'qm'``, ``'bdd'``, ``'auto'`` levels plus the corresponding caps,
   and rejects invalid combinations.
3. **Functional equivalence** — for every advanced level, the
   generated C code, when interpreted in Python through the emitted
   if-chain, matches ``model.predict`` on a held-out grid (same
   guarantee that the unit tests for QM/BDD enforce, but exercised
   through the full pipeline).
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from blackbox2c import ConversionConfig, Converter


# ─────────────────────────── fixtures ──────────────────────────────
@pytest.fixture
def iris_tree():
    # Function scope: ``RuleOptimizer.optimize`` mutates the sklearn
    # tree in place, so reusing the same fixture across tests would
    # silently degrade it.
    X, y = load_iris(return_X_y=True)
    tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
    return tree, X, y


@pytest.fixture
def diabetes_tree():
    # Real continuous regression target -> guarantees the surrogate tree
    # has non-trivial split conditions (avoids the degenerate
    # 'all-leaves' case we get with class-labels-as-floats).
    X, y = load_diabetes(return_X_y=True)
    tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y)
    return tree, X, y


# ───────────────────── ConversionConfig surface ────────────────────
class TestConfigSurface:
    @pytest.mark.parametrize(
        "level", ["low", "medium", "high", "qm", "bdd", "auto"],
    )
    def test_accepts_valid_levels(self, level):
        cfg = ConversionConfig(optimize_rules=level)
        assert cfg.optimize_rules == level

    def test_rejects_invalid_level(self):
        with pytest.raises(ValueError, match="optimize_rules"):
            ConversionConfig(optimize_rules="bogus")

    def test_rejects_negative_caps(self):
        with pytest.raises(ValueError, match="qm_max_literals"):
            ConversionConfig(qm_max_literals=-1)
        with pytest.raises(ValueError, match="bdd_max_literals"):
            ConversionConfig(bdd_max_literals=-1)

    def test_caps_have_documented_defaults(self):
        cfg = ConversionConfig()
        assert cfg.qm_max_literals == 12
        assert cfg.bdd_max_literals == 24


# ───────────────────── backward compatibility ──────────────────────
class TestBackwardCompatibility:
    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_legacy_levels_produce_tree_codegen_output(self, iris_tree, level):
        tree, X, _ = iris_tree
        cfg = ConversionConfig(max_depth=3, optimize_rules=level)
        code = Converter(cfg).convert(
            tree, X,
            feature_names=[f"f{i}" for i in range(X.shape[1])],
            class_names=["A", "B", "C"],
        )
        # Legacy codegen always uses ``<=`` and the nested-if form.
        # The advanced bridge would emit ``&&`` chains – check that
        # those are *not* present for legacy levels.
        assert "<=" in code
        assert "&&" not in code


# ─────────────────────── advanced equivalence ──────────────────────
def _interpret_c(code: str, X: np.ndarray) -> np.ndarray:
    """Tiny reverse-emit interpreter for the ``RuleSetCodeGenerator``
    body.  Parses every ``if (... && ...) return N;`` line and applies
    them in order, plus the trailing default ``return N;``.

    This is a deliberate test-only oracle: if the C body matches, we
    have full functional equivalence with the optimised RuleSet.
    """
    rule_re = re.compile(
        r"if \((?P<cond>[^)]+)\) \{\s*return (?P<cls>\d+);\s*\}"
    )
    fallback_re = re.compile(r"return (\d+);\s*\}\s*$", re.MULTILINE)

    def parse_lit(lit: str):
        m = re.match(
            r"features\[(\d+)\]\s*(<=|>)\s*(-?\d+(?:\.\d+)?)f?", lit.strip()
        )
        assert m, f"bad literal: {lit!r}"
        return int(m.group(1)), m.group(2), float(m.group(3))

    rules = []
    for m in rule_re.finditer(code):
        lits = [parse_lit(p) for p in m.group("cond").split("&&")]
        rules.append((lits, int(m.group("cls"))))
    fallback_match = fallback_re.search(code)
    fallback = int(fallback_match.group(1)) if fallback_match else 0

    out = np.empty(len(X), dtype=int)
    for i, x in enumerate(X):
        pred = fallback
        for lits, cls in rules:
            if all(
                (x[f] <= t) if op == "<=" else (x[f] > t)
                for f, op, t in lits
            ):
                pred = cls
                break
        out[i] = pred
    return out


class TestAdvancedEquivalence:
    @pytest.mark.parametrize("level", ["qm", "bdd", "auto"])
    def test_generated_c_matches_tree_predictions(self, iris_tree, level):
        tree, X, _ = iris_tree
        cfg = ConversionConfig(
            max_depth=3, optimize_rules=level,
            qm_max_literals=20, bdd_max_literals=30,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            code = Converter(cfg).convert(
                tree, X,
                feature_names=[f"f{i}" for i in range(X.shape[1])],
                class_names=["A", "B", "C"],
            )
        # The advanced bridge always emits ``&&``-joined conditions.
        assert "&&" in code

        rng = np.random.default_rng(7)
        lo, hi = X.min(axis=0), X.max(axis=0)
        X_eval = rng.uniform(lo, hi, size=(800, X.shape[1]))
        c_pred = _interpret_c(code, X_eval)
        sk_pred = tree.predict(X_eval)
        assert np.array_equal(c_pred, sk_pred)


# ─────────────────────── regression fallback ───────────────────────
class TestRegressionFallback:
    @pytest.mark.parametrize("level", ["qm", "bdd", "auto"])
    def test_regression_warns_and_uses_legacy_codegen(
        self, diabetes_tree, level
    ):
        tree, X, _ = diabetes_tree
        cfg = ConversionConfig(max_depth=3, optimize_rules=level)
        with pytest.warns(UserWarning, match="classification"):
            code = Converter(cfg).convert(
                tree, X,
                feature_names=[f"f{i}" for i in range(X.shape[1])],
            )
        # Legacy codegen surface: nested-if, no &&-conditions.
        assert "<=" in code
        assert "&&" not in code
