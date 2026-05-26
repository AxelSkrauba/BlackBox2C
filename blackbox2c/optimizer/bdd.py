"""
Reduced Ordered Binary Decision Diagram (ROBDD) over a RuleSet.

The BDD lives in the same multi-valued universe as :mod:`qm`: each
distinct ``(feature_idx, threshold)`` pair becomes one Boolean variable
``v_k`` with the convention ``v_k = (x[feature_idx] > threshold)``.  We
build a *reduced* and *ordered* BDD on this Boolean alphabet:

* **Ordered**: a fixed total ordering of the variables is followed on
  every root-to-terminal path.  We sort variables by *frequency* across
  the input RuleSet (most-used first), which is a cheap and surprisingly
  effective heuristic for tree-derived rules.
* **Reduced**: the unique-table guarantees that two structurally
  identical sub-DAGs are represented by the same node, and the
  collapse rule eliminates nodes whose two children are equal.

For classification with ``C`` classes we build *one BDD per class*
(treating each class as a Boolean function "this leaf predicts class
c").  We then pick the class whose BDD currently produces ``1``; if
several classes claim the input, the lowest-indexed wins (this can only
happen for an inconsistent / overlapping input RuleSet — tree-derived
RuleSets always partition the space).

The BDD is converted back to a :class:`RuleSet` by enumerating its
``1``-paths.  In v0.2 we do not yet exploit shared subgraphs in the C
output: that optimisation is the natural next step (v0.3) once the
codegen layer can emit functions or labels.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .ir import Conjunction, Literal, RuleSet


# ─────────────────────────────────────────────────────────────────────
# Public configuration
# ─────────────────────────────────────────────────────────────────────

DEFAULT_MAX_LITERALS: int = 24
DEFAULT_MAX_BDD_NODES: int = 200_000


# A BDD variable is identified by its (feature_idx, threshold) pair.
# The semantics are: ``var = True`` iff ``x[feature_idx] > threshold``.
Var = Tuple[int, float]

# Terminal sentinels.
_TERMINAL_FALSE = -1
_TERMINAL_TRUE = -2


@dataclass
class _BDD:
    """Lightweight ROBDD with a unique-table.

    Nodes are stored in ``self.nodes``: each entry is a tuple
    ``(var_index, low_id, high_id)`` where ``var_index`` is the index
    into ``self.var_order`` and ``low_id`` / ``high_id`` are either
    other entries in ``self.nodes`` (non-negative ids) or one of the
    terminal sentinels ``_TERMINAL_FALSE`` / ``_TERMINAL_TRUE``.

    The convention is: traversing ``low`` corresponds to ``var = False``
    (i.e. ``x[f] <= θ``); traversing ``high`` corresponds to ``var =
    True`` (``x[f] > θ``).
    """

    var_order: Tuple[Var, ...]
    nodes: List[Tuple[int, int, int]] = field(default_factory=list)
    unique: Dict[Tuple[int, int, int], int] = field(default_factory=dict)

    # ── Internal helpers ────────────────────────────────────────────
    def _make(self, var_index: int, low: int, high: int) -> int:
        # Reduction rule 1: collapse when both children are equal.
        if low == high:
            return low
        key = (var_index, low, high)
        # Reduction rule 2: unique-table.
        cached = self.unique.get(key)
        if cached is not None:
            return cached
        node_id = len(self.nodes)
        self.nodes.append(key)
        self.unique[key] = node_id
        return node_id

    # ── Construction from a Boolean function ───────────────────────
    def build(self, evaluate) -> int:
        """Build the BDD recursively in the order ``self.var_order``.

        ``evaluate(assignment)`` is a callable that takes a dict
        ``{Var: bool}`` covering every variable in ``var_order`` and
        returns ``True`` / ``False``.

        For a small number of variables (the only regime in which we
        run, see ``DEFAULT_MAX_LITERALS``) this top-down construction
        with memoisation is more than fast enough; the unique-table
        keeps the resulting BDD reduced.
        """
        memo: Dict[FrozenSet[Tuple[int, bool]], int] = {}

        def rec(level: int, assignment: Dict[int, bool]) -> int:
            key = frozenset(assignment.items())
            if key in memo:
                return memo[key]
            if level == len(self.var_order):
                value = evaluate(
                    {self.var_order[i]: v for i, v in assignment.items()}
                )
                node = _TERMINAL_TRUE if value else _TERMINAL_FALSE
                memo[key] = node
                return node
            low = rec(level + 1, {**assignment, level: False})
            high = rec(level + 1, {**assignment, level: True})
            node = self._make(level, low, high)
            memo[key] = node
            return node

        return rec(0, {})

    # ── Path enumeration ───────────────────────────────────────────
    def true_paths(self, root: int) -> List[Dict[int, bool]]:
        """Return every assignment that leads from ``root`` to True.

        Each path is a dict mapping ``var_index → bool``.  Variables
        that do not appear on the path are *don't-cares*.
        """
        if root == _TERMINAL_FALSE:
            return []
        if root == _TERMINAL_TRUE:
            return [{}]

        paths: List[Dict[int, bool]] = []

        def walk(node: int, assignment: Dict[int, bool]) -> None:
            if node == _TERMINAL_FALSE:
                return
            if node == _TERMINAL_TRUE:
                paths.append(dict(assignment))
                return
            var_index, low, high = self.nodes[node]
            walk(low, {**assignment, var_index: False})
            walk(high, {**assignment, var_index: True})

        walk(root, {})
        return paths

    @property
    def n_internal_nodes(self) -> int:
        return len(self.nodes)


# ─────────────────────────────────────────────────────────────────────
# Public optimizer
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BDDOptimizer:
    """ROBDD-based minimizer over a :class:`RuleSet`.

    Parameters
    ----------
    max_literals : int, default 24
        Maximum number of unique ``(feature, threshold)`` pairs.
    max_bdd_nodes : int, default 200_000
        Soft ceiling on the size of the constructed BDD; if exceeded
        during the build we abort and return the input RuleSet.
    """

    max_literals: int = DEFAULT_MAX_LITERALS
    max_bdd_nodes: int = DEFAULT_MAX_BDD_NODES
    last_diagnostics_: Dict[str, object] = field(default_factory=dict)

    def minimize(self, rs: RuleSet) -> RuleSet:
        if rs.n_classes is None:
            warnings.warn(
                "BDDOptimizer is only defined for classification; "
                "returning the input RuleSet unchanged.",
                UserWarning,
                stacklevel=2,
            )
            return rs

        var_order = _build_var_order(rs)
        n_lits = len(var_order)

        if n_lits == 0:
            self.last_diagnostics_ = {
                "n_literals": 0, "applied": False, "reason": "single-leaf",
            }
            return rs

        if n_lits > self.max_literals:
            warnings.warn(
                f"BDDOptimizer: {n_lits} literals exceed "
                f"max_literals={self.max_literals}; returning input "
                "unchanged.",
                UserWarning,
                stacklevel=2,
            )
            self.last_diagnostics_ = {
                "n_literals": n_lits, "applied": False,
                "reason": "max_literals_exceeded",
            }
            return rs

        # Build one BDD per class.
        bdds: List[Tuple[_BDD, int]] = []
        for c in range(rs.n_classes):
            bdd = _BDD(var_order=var_order)
            root = bdd.build(_class_indicator(rs, c, var_order))
            if bdd.n_internal_nodes > self.max_bdd_nodes:
                warnings.warn(
                    f"BDDOptimizer: BDD for class {c} reached "
                    f"{bdd.n_internal_nodes} nodes > "
                    f"max_bdd_nodes={self.max_bdd_nodes}; aborting.",
                    UserWarning,
                    stacklevel=2,
                )
                self.last_diagnostics_ = {
                    "n_literals": n_lits, "applied": False,
                    "reason": "max_bdd_nodes_exceeded",
                }
                return rs
            bdds.append((bdd, root))

        # Re-emit: enumerate true-paths of every per-class BDD.
        new_rules: List[Conjunction] = []
        for c, (bdd, root) in enumerate(bdds):
            for path in bdd.true_paths(root):
                lits = _path_to_literals(path, var_order)
                new_rules.append(Conjunction(lits, c))

        if not new_rules:
            return rs

        new_rs = RuleSet(
            rules=tuple(new_rules),
            n_features=rs.n_features,
            n_classes=rs.n_classes,
            feature_names=rs.feature_names,
        ).simplify()

        total_nodes = sum(bdd.n_internal_nodes for bdd, _ in bdds)
        self.last_diagnostics_ = {
            "n_literals": n_lits,
            "applied": True,
            "reason": "ok",
            "n_rules_in": len(rs.rules),
            "n_rules_out": len(new_rules),
            "bdd_total_internal_nodes": total_nodes,
        }
        return new_rs


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _build_var_order(rs: RuleSet) -> Tuple[Var, ...]:
    """Order variables by descending frequency in the RuleSet.

    Ties are broken by ``feature_idx`` then ``threshold`` so the order
    is deterministic across runs.
    """
    counts: Counter = Counter()
    for rule in rs.rules:
        for lit in rule.literals:
            canonical = lit if lit.op == "<=" else lit.negate()
            counts[(canonical.feature_idx, float(canonical.threshold))] += 1
    return tuple(
        sorted(counts.keys(), key=lambda v: (-counts[v], v[0], v[1]))
    )


def _class_indicator(rs: RuleSet, cls: int, var_order: Tuple[Var, ...]):
    """Return a callable ``assignment -> bool`` that is True iff the
    RuleSet predicts ``cls`` for any input consistent with the given
    variable assignment.

    The callable is closed over a constant point per assignment (a
    representative sample chosen near each threshold) and uses the
    RuleSet's ``predict`` to label that point.
    """
    # Pre-compute a constant offset for each variable: when the variable
    # is True we put the input slightly above the threshold; when False,
    # slightly below or equal.  We pad by 1.0 to the side of the most
    # extreme threshold for the same feature, so the chosen point is
    # consistent with *every* literal on that feature simultaneously.
    by_feature: Dict[int, List[float]] = {}
    for f, t in var_order:
        by_feature.setdefault(f, []).append(t)
    for f in by_feature:
        by_feature[f].sort()

    def make_point(assignment: Dict[Var, bool]) -> np.ndarray:
        x = np.zeros(rs.n_features)
        for f, ts in by_feature.items():
            # Find the tightest interval consistent with every literal
            # on this feature.
            lo = -np.inf
            hi = np.inf
            for t in ts:
                truth = assignment[(f, t)]
                if truth:                      # x > t  →  lo := max(lo, t)
                    lo = max(lo, t)
                else:                           # x <= t →  hi := min(hi, t)
                    hi = min(hi, t)
            if lo == -np.inf and hi == np.inf:
                x[f] = 0.0
            elif lo == -np.inf:
                x[f] = hi  # left-closed at threshold
            elif hi == np.inf:
                x[f] = lo + 1.0
            else:
                if lo >= hi:
                    # Inconsistent assignment (cannot satisfy both
                    # constraints).  Pick *any* point and let the
                    # caller-side don't-care collapse handle it: such
                    # paths cannot lead to a 1 in the BDD anyway because
                    # the underlying RuleSet would also fail to match.
                    x[f] = lo + 1e-9
                else:
                    # Pick a point strictly inside (lo, hi].
                    x[f] = 0.5 * (lo + hi)
        return x

    def evaluate(assignment: Dict[Var, bool]) -> bool:
        x = make_point(assignment)
        try:
            pred = rs.predict(x[None, :])[0]
        except ValueError:
            return False
        return int(pred) == cls

    return evaluate


def _path_to_literals(
    path: Dict[int, bool], var_order: Tuple[Var, ...]
) -> Tuple[Literal, ...]:
    """Convert a BDD assignment to a tuple of canonical Literals.

    ``path`` maps ``var_index → bool``.  We translate ``True`` to
    ``x[f] > θ`` (canonical ``>``) and ``False`` to ``x[f] <= θ``.
    """
    lits: List[Literal] = []
    for var_index, truth in path.items():
        f, t = var_order[var_index]
        op = ">" if truth else "<="
        lits.append(Literal(f, t, op))
    return tuple(lits)
