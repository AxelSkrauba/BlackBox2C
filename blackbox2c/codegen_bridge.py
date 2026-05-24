"""
RuleSet → C bridge.

The advanced (``'qm'``, ``'bdd'``, ``'auto'``) optimization levels
operate on the immutable IR (:class:`blackbox2c.optimizer.ir.RuleSet`)
rather than on a sklearn tree.  This module rebuilds a *hierarchical*
decision tree from the optimized RuleSet and emits nested if/else C
code, matching the surface of the legacy :class:`CCodeGenerator` so
prefix sharing is preserved across rules (a flat ``if (a&&b) ...; if
(a&&c) ...;`` chain re-evaluates ``a`` once per rule – the hierarchical
form evaluates it once for the entire subtree).

Algorithm sketch (split-on-most-frequent-literal):

1. If every remaining rule shares the same prediction, emit a single
   ``return``.
2. Otherwise pick the most frequent ``(feature, threshold)`` pair
   across the remaining rule literals.  This is a deterministic greedy
   heuristic that approximates the optimum in linear time per call.
3. Partition the rules into the ``<=`` branch, the ``>`` branch and a
   "neutral" set (rules whose literal set does not mention this
   particular pair) – the latter is replicated into both branches.
4. Strip the pivot literal from each branch's rules and recurse.

This relies on the input RuleSet being **mutually exclusive and
total** over the feature space, which is always the case for RuleSets
that come out of the QM and BDD optimizers in this repository.  For an
arbitrary user-supplied RuleSet with overlapping rules the
hierarchical reconstruction may not respect decision-list semantics;
the documented entry point of the bridge therefore expects RuleSets
produced by the optimizer pipeline.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

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
        """Heuristic size estimate parallel to ``estimate_code_size``.

        We measure the *reconstructed* tree (number of internal split
        nodes and number of leaves) rather than the literal count of
        the input RuleSet, so the estimate matches what the
        hierarchical codegen actually emits.
        """
        n_internal, n_leaves = self._tree_shape(list(rs.rules))

        bytes_per_condition = 12
        bytes_per_return = 4
        overhead = 50

        estimated_flash = (
            overhead
            + n_internal * bytes_per_condition
            + n_leaves * bytes_per_return
        )
        return {
            "flash_bytes": estimated_flash,
            "ram_bytes": 32,
            "n_nodes": n_internal + n_leaves,
            "n_conditions": n_internal,
            "n_leaves": n_leaves,
        }

    # ── private helpers ───────────────────────────────────────────
    def _generate_ruleset_body(self, rs: RuleSet) -> str:
        return self._emit_tree(list(rs.rules), depth=1, fallback=self._fallback_prediction(rs))

    def _fallback_prediction(self, rs: RuleSet):
        if self.task_type == "classification":
            preds = [int(rule.prediction) for rule in rs.rules]
            return int(np.bincount(preds).argmax()) if preds else 0
        preds = [float(rule.prediction) for rule in rs.rules]
        return float(np.mean(preds)) if preds else 0.0

    def _emit_tree(
        self, rules: List[Conjunction], depth: int, fallback
    ) -> str:
        indent = " " * (self.indent_size * depth)

        # ── base cases ────────────────────────────────────────────
        if not rules:
            return f"{indent}{self._format_return(fallback)}"

        # First unconditional rule shadows everything that follows.
        for r in rules:
            if not r.literals:
                return f"{indent}{self._format_return(r.prediction)}"

        # Same-prediction collapse.
        preds = {r.prediction for r in rules}
        if len(preds) == 1:
            return f"{indent}{self._format_return(rules[0].prediction)}"

        pivot = self._pick_pivot(rules)
        if pivot is None:
            # No literal left to split on but predictions still
            # disagree — emit the fallback (should be unreachable for
            # total RuleSets coming from QM/BDD).
            return f"{indent}{self._format_return(fallback)}"

        f_idx, threshold = pivot
        yes_rules, no_rules = self._partition(rules, f_idx, threshold)

        thr_str = self._format_threshold(threshold)
        yes_code = self._emit_tree(yes_rules, depth + 1, fallback)
        no_code = self._emit_tree(no_rules, depth + 1, fallback)
        return (
            f"{indent}if (features[{f_idx}] <= {thr_str}) {{\n"
            f"{yes_code}\n"
            f"{indent}}} else {{\n"
            f"{no_code}\n"
            f"{indent}}}"
        )

    @staticmethod
    def _pick_pivot(rules: List[Conjunction]) -> Optional[Tuple[int, float]]:
        """Return the most common ``(feature_idx, threshold)`` pair
        across the rules' literals, or ``None`` if no rule has any
        literal left.  Ties are broken deterministically (sorted)."""
        counts: Counter = Counter()
        for r in rules:
            for lit in r.literals:
                counts[(lit.feature_idx, float(lit.threshold))] += 1
        if not counts:
            return None
        max_count = max(counts.values())
        return min(k for k, v in counts.items() if v == max_count)

    @staticmethod
    def _partition(
        rules: List[Conjunction], f_idx: int, threshold: float
    ) -> Tuple[List[Conjunction], List[Conjunction]]:
        """Split ``rules`` into the ``<=`` and ``>`` branches.

        For each rule we look for a literal on ``(f_idx, threshold)``:
        a ``<=`` literal sends the rule to the left branch, a ``>``
        literal sends it to the right.  Rules with no literal on this
        exact pair go to *both* branches.  In every case the pivot
        literal is stripped from the rule that we propagate, since the
        condition is now encoded by the if/else itself.
        """
        yes: List[Conjunction] = []
        no: List[Conjunction] = []
        for r in rules:
            has_le = any(
                lit.feature_idx == f_idx
                and float(lit.threshold) == threshold
                and lit.op == "<="
                for lit in r.literals
            )
            has_gt = any(
                lit.feature_idx == f_idx
                and float(lit.threshold) == threshold
                and lit.op == ">"
                for lit in r.literals
            )
            stripped = tuple(
                lit
                for lit in r.literals
                if not (
                    lit.feature_idx == f_idx
                    and float(lit.threshold) == threshold
                )
            )
            new_rule = Conjunction(stripped, r.prediction)
            if has_le and not has_gt:
                yes.append(new_rule)
            elif has_gt and not has_le:
                no.append(new_rule)
            else:
                # Either neutral (no literal on this pair) or
                # contradictory (both <= and >; cannot fire — drop).
                if not (has_le and has_gt):
                    yes.append(new_rule)
                    no.append(new_rule)
        return yes, no

    def _tree_shape(self, rules: List[Conjunction]) -> Tuple[int, int]:
        """Count (internal_nodes, leaves) of the reconstructed tree
        without actually generating the C source."""
        if not rules:
            return 0, 1
        for r in rules:
            if not r.literals:
                return 0, 1
        if len({r.prediction for r in rules}) == 1:
            return 0, 1
        pivot = self._pick_pivot(rules)
        if pivot is None:
            return 0, 1
        yes_rules, no_rules = self._partition(rules, pivot[0], pivot[1])
        l_int, l_leaf = self._tree_shape(yes_rules)
        r_int, r_leaf = self._tree_shape(no_rules)
        return 1 + l_int + r_int, l_leaf + r_leaf

    def _format_threshold(self, threshold: float) -> str:
        if self.use_fixed_point:
            return str(int(threshold * (2 ** (self.precision - 1))))
        return f"{float(threshold):.6f}f"

    def _format_return(self, prediction) -> str:
        if self.task_type == "classification":
            return f"return {int(prediction)};"
        return f"return {float(prediction):.6f}f;"
