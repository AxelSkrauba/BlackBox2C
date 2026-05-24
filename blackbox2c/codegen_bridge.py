"""
RuleSet → C bridge.

The advanced (``'qm'``, ``'bdd'``, ``'auto'``) optimization levels
operate on the immutable IR (:class:`blackbox2c.optimizer.ir.RuleSet`)
rather than on a sklearn tree.  Their output therefore cannot reuse the
tree-walking codegen path verbatim; this module supplies a thin
generator that emits the same C surface (header, defines, function
signature, usage comments) as :class:`CCodeGenerator`, but with a body
made of *flat* ``if (lit && lit && ...) return cls;`` chains – one per
optimised rule.

Design notes
------------
* We subclass :class:`CCodeGenerator` strictly to inherit its already
  battle-tested formatting helpers (``_generate_header``,
  ``_generate_defines``, ``_generate_function_signature``,
  ``_generate_usage_comment``).  We do **not** depend on its tree-aware
  internals.
* The default (no rule matched) branch must never fire on RuleSets
  derived from total decision-tree partitions, but we emit one anyway
  to keep the generated function total at the C level.  The fallback
  prediction is the most frequent class in the RuleSet.
* Code-size estimation reuses the same heuristic as the tree-based
  generator (cost per literal ≈ cost per condition; cost per rule ≈
  cost per return).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .codegen import CCodeGenerator
from .optimizer.ir import Conjunction, Literal, RuleSet


class RuleSetCodeGenerator(CCodeGenerator):
    """Emit embedded C code directly from a :class:`RuleSet`."""

    # ── public API ────────────────────────────────────────────────
    def generate_from_ruleset(
        self,
        rs: RuleSet,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
    ) -> str:
        """Generate C code for ``rs``.

        Parameters
        ----------
        rs : RuleSet
            RuleSet (already optimised).
        feature_names : list[str]
            Feature names matching ``rs.n_features``.
        class_names : list[str] or None
            Class labels for classification; ignored for regression.
        """
        if len(feature_names) != rs.n_features:
            raise ValueError(
                f"feature_names has length {len(feature_names)} but "
                f"RuleSet has n_features={rs.n_features}"
            )

        if rs.n_classes is None:
            self.task_type = "regression"
            n_classes: Optional[int] = None
            class_names = None
        else:
            self.task_type = "classification"
            n_classes = rs.n_classes
            if class_names is None:
                class_names = [f"CLASS_{i}" for i in range(n_classes)]

        n_features = rs.n_features

        parts: List[str] = []
        parts.append(self._generate_header(n_features, n_classes))
        if self.task_type == "classification":
            parts.append(self._generate_defines(class_names))
        parts.append(self._generate_function_signature(n_features))

        body = self._generate_ruleset_body(rs)
        parts.append(body)

        parts.append("}\n")

        if self.task_type == "classification":
            parts.append(self._generate_usage_comment(feature_names, class_names))
        else:
            parts.append(self._generate_usage_comment_regression(feature_names))

        return "\n".join(parts)

    def estimate_code_size_from_ruleset(self, rs: RuleSet) -> dict:
        """Heuristic size estimate parallel to ``estimate_code_size``."""
        n_rules = len(rs.rules)
        n_literals = sum(len(rule.literals) for rule in rs.rules)

        bytes_per_condition = 12
        bytes_per_return = 4
        overhead = 50

        estimated_flash = (
            overhead
            + n_literals * bytes_per_condition
            + n_rules * bytes_per_return
        )
        return {
            "flash_bytes": estimated_flash,
            "ram_bytes": 32,
            "n_nodes": n_rules + n_literals,
            "n_conditions": n_literals,
            "n_leaves": n_rules,
        }

    # ── private helpers ───────────────────────────────────────────
    def _generate_ruleset_body(self, rs: RuleSet) -> str:
        indent = " " * self.indent_size
        lines: List[str] = []
        for rule in rs.rules:
            lines.append(self._format_rule(rule, indent))

        # Default fallback: emit the most-frequent prediction so the
        # function is total at the C level.  For total partitions
        # extracted from sklearn trees this branch is unreachable.
        lines.append(indent + self._format_fallback(rs))
        return "\n".join(lines)

    def _format_rule(self, rule: Conjunction, indent: str) -> str:
        if not rule.literals:
            # Unconditional rule.
            return indent + self._format_return(rule.prediction)
        cond = " && ".join(self._format_literal(lit) for lit in rule.literals)
        return (
            f"{indent}if ({cond}) {{\n"
            f"{indent}{' ' * self.indent_size}"
            f"{self._format_return(rule.prediction)}\n"
            f"{indent}}}"
        )

    def _format_literal(self, lit: Literal) -> str:
        thr = self._format_threshold(lit.threshold)
        return f"features[{lit.feature_idx}] {lit.op} {thr}"

    def _format_threshold(self, threshold: float) -> str:
        if self.use_fixed_point:
            return str(int(threshold * (2 ** (self.precision - 1))))
        return f"{float(threshold):.6f}f"

    def _format_return(self, prediction) -> str:
        if self.task_type == "classification":
            return f"return {int(prediction)};"
        return f"return {float(prediction):.6f}f;"

    def _format_fallback(self, rs: RuleSet) -> str:
        if self.task_type == "classification":
            preds = [int(rule.prediction) for rule in rs.rules]
            fallback = int(np.bincount(preds).argmax()) if preds else 0
            return f"return {fallback};"
        preds = [float(rule.prediction) for rule in rs.rules]
        fallback = float(np.mean(preds)) if preds else 0.0
        return f"return {fallback:.6f}f;"
