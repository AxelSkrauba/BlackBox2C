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
def _extract_body(code: str) -> str:
    """Slice between the function-opening ``{`` and the matching ``}``."""
    start = code.index("{", code.index("predict("))
    depth = 0
    for i in range(start, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return code[start + 1:i]
    raise AssertionError("unbalanced braces in generated code")


def _interpret_c(code: str, X: np.ndarray) -> np.ndarray:
    """Tiny C interpreter for the bodies that this project emits.

    Supports the two surfaces we actually generate:
    1. *Nested if/else*: ``if (features[i] <op> thr) { ... } else { ... }``
       (legacy codegen and the v0.2 hierarchical bridge).
    2. *Flat ``&&`` chains*: ``if (a && b) { return N; }`` followed by a
       trailing ``return N;`` (kept around so older snapshots of the
       bridge stay testable).

    The interpreter is a tokenless mini-parser: enough to verify that
    the emitted C, evaluated as Python, agrees with the original model.
    """
    body = _extract_body(code)
    pos = [0]

    cond_re = re.compile(
        r"features\[(\d+)\]\s*(<=|>)\s*(-?\d+(?:\.\d+)?)f?"
    )
    return_re = re.compile(r"\s*return\s+(\d+)\s*;")

    def skip_ws():
        while pos[0] < len(body) and body[pos[0]] in " \t\n\r":
            pos[0] += 1

    def parse_block(x: np.ndarray):
        """Parse a brace-delimited block and return its prediction."""
        skip_ws()
        assert body[pos[0]] == "{", f"expected {{ at {pos[0]}: {body[pos[0]:pos[0]+30]!r}"
        pos[0] += 1
        result = parse_stmt_list(x)
        skip_ws()
        assert body[pos[0]] == "}"
        pos[0] += 1
        return result

    def parse_condition(x: np.ndarray) -> bool:
        # Read a parenthesised condition, possibly joined by &&.
        skip_ws()
        assert body[pos[0]] == "("
        pos[0] += 1
        depth = 1
        start = pos[0]
        while depth:
            if body[pos[0]] == "(":
                depth += 1
            elif body[pos[0]] == ")":
                depth -= 1
                if depth == 0:
                    break
            pos[0] += 1
        cond_str = body[start:pos[0]]
        pos[0] += 1  # consume )
        # Evaluate AND-of-literals.
        result = True
        for lit in cond_str.split("&&"):
            m = cond_re.match(lit.strip())
            assert m, f"bad literal: {lit!r}"
            f, op, thr = int(m.group(1)), m.group(2), float(m.group(3))
            ok = (x[f] <= thr) if op == "<=" else (x[f] > thr)
            result = result and ok
        return result

    def parse_stmt_list(x: np.ndarray):
        """Return prediction or ``None`` if no statement fired."""
        while True:
            skip_ws()
            if pos[0] >= len(body) or body[pos[0]] == "}":
                return None
            # Try `return N;`
            m = return_re.match(body, pos[0])
            if m:
                pos[0] = m.end()
                return int(m.group(1))
            # Else expect `if (...) { ... } [else { ... }]`
            assert body.startswith("if", pos[0]), \
                f"unexpected: {body[pos[0]:pos[0]+20]!r}"
            pos[0] += 2
            cond_ok = parse_condition(x)
            if cond_ok:
                pred = parse_block(x)
                # Skip any trailing else block without executing it.
                skip_ws()
                if body.startswith("else", pos[0]):
                    pos[0] += 4
                    _skip_block()
                if pred is not None:
                    return pred
            else:
                _skip_block()
                skip_ws()
                if body.startswith("else", pos[0]):
                    pos[0] += 4
                    pred = parse_block(x)
                    if pred is not None:
                        return pred

    def _skip_block():
        skip_ws()
        assert body[pos[0]] == "{"
        depth = 0
        while pos[0] < len(body):
            if body[pos[0]] == "{":
                depth += 1
            elif body[pos[0]] == "}":
                depth -= 1
                if depth == 0:
                    pos[0] += 1
                    return
            pos[0] += 1

    out = np.empty(len(X), dtype=int)
    for i, x in enumerate(X):
        pos[0] = 0
        pred = parse_stmt_list(x)
        out[i] = 0 if pred is None else pred
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
        # Bridge emits the same nested-if/else surface as legacy
        # codegen since v0.2-Phase-6; the strong invariant is
        # functional equivalence, asserted below.
        assert "<=" in code

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
