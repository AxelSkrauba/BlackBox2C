"""
Immutable intermediate representation for surrogate decision rules.

The IR decouples optimization algorithms (Quine-McCluskey, BDD,
legacy pruning) from the concrete sklearn tree representation.

Key types
---------
- :class:`Literal` — a single ``feature[i] <op> threshold`` test.
- :class:`Conjunction` — an AND of literals together with the prediction
  emitted when all literals hold.
- :class:`RuleSet` — a DNF of conjunctions covering the input space; the
  canonical container produced by :mod:`blackbox2c.optimizer.extraction`
  and consumed by every advanced optimizer.

All three classes are :func:`dataclasses.dataclass`-frozen, so optimizer
passes are forced to return new instances rather than mutating shared
state.  This is essential for the equivalence-checking gate that runs
between every optimization step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

# Canonical operators supported by the IR.  Decision-tree splits are
# encoded as ``<=`` (left child of a sklearn split) and ``>`` (right
# child).  Keeping the alphabet minimal simplifies the QM / BDD layers.
_OPS = ("<=", ">")


@dataclass(frozen=True)
class Literal:
    """A single threshold test ``x[feature_idx] <op> threshold``.

    Parameters
    ----------
    feature_idx : int
        Index into the feature vector.
    threshold : float
        Split threshold.
    op : {'<=', '>'}
        Comparison operator.  Only these two are valid; equality and
        strict-less are intentionally excluded to match scikit-learn's
        CART semantics.
    """

    feature_idx: int
    threshold: float
    op: str

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"Literal.op must be one of {_OPS}, got {self.op!r}")
        if self.feature_idx < 0:
            raise ValueError("feature_idx must be non-negative")

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Return a boolean mask over rows of ``X``."""
        col = np.asarray(X)[..., self.feature_idx]
        if self.op == "<=":
            return col <= self.threshold
        return col > self.threshold

    def negate(self) -> "Literal":
        """Return the logical negation of this literal."""
        flipped = ">" if self.op == "<=" else "<="
        return Literal(self.feature_idx, self.threshold, flipped)


@dataclass(frozen=True)
class Conjunction:
    """An AND of literals together with the prediction it emits.

    An empty literal tuple represents the trivially-true conjunction
    (i.e., the rule fires for every input); this is used to encode a
    single-leaf tree.
    """

    literals: Tuple[Literal, ...]
    prediction: Union[int, float]

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Boolean mask: which rows satisfy *all* literals."""
        X = np.asarray(X)
        n = X.shape[0] if X.ndim > 1 else 1
        if not self.literals:
            return np.ones(n, dtype=bool)
        mask = np.ones(n, dtype=bool)
        for lit in self.literals:
            mask &= lit.evaluate(X)
        return mask


@dataclass(frozen=True)
class RuleSet:
    """A DNF (OR of :class:`Conjunction`) representing a decision function.

    The order of ``rules`` matters: :meth:`predict` returns the
    prediction of the *first* matching conjunction, mirroring the
    top-down semantics of an if-else cascade.  For a RuleSet derived
    from a decision tree, conjunctions are mutually exclusive and
    cover the input space, so the order is irrelevant in practice;
    optimization passes (QM, BDD) may produce overlapping rules where
    order *is* significant.

    Parameters
    ----------
    rules : tuple of Conjunction
        Ordered conjunctions.
    n_features : int
        Cardinality of the feature space.
    n_classes : int, optional
        Number of classes for classification.  ``None`` for regression.
    feature_names : tuple of str, optional
        Optional names for downstream code generation.
    """

    rules: Tuple[Conjunction, ...]
    n_features: int
    n_classes: Optional[int] = None
    feature_names: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.n_features < 1:
            raise ValueError("n_features must be >= 1")
        if self.feature_names is not None and len(self.feature_names) != self.n_features:
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) "
                f"does not match n_features ({self.n_features})"
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by returning the first matching conjunction.

        Raises
        ------
        ValueError
            If some input row is matched by no rule (incomplete
            RuleSet).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X.shape[1]}"
            )

        out = np.empty(X.shape[0], dtype=float)
        unmatched = np.ones(X.shape[0], dtype=bool)
        for rule in self.rules:
            if not unmatched.any():
                break
            fires = rule.evaluate(X) & unmatched
            out[fires] = rule.prediction
            unmatched &= ~fires

        if unmatched.any():
            raise ValueError(
                "RuleSet is incomplete: "
                f"{int(unmatched.sum())}/{len(X)} samples matched no rule"
            )

        if self.n_classes is not None:
            return out.astype(np.int64)
        return out

    def complexity(self) -> dict:
        """Summary statistics for benchmarks and tests."""
        total = sum(len(r.literals) for r in self.rules)
        max_depth = max((len(r.literals) for r in self.rules), default=0)
        return {
            "n_rules": len(self.rules),
            "total_literals": total,
            "max_depth": max_depth,
        }

    def unique_literals(self) -> Tuple[Literal, ...]:
        """All distinct literals appearing in the RuleSet (canonical form).

        For boolean-minimization passes we collapse ``feature[i] > θ`` and
        ``feature[i] <= θ`` to the same literal (``<=`` form) since one is
        the negation of the other.
        """
        seen: dict = {}
        for r in self.rules:
            for lit in r.literals:
                canon = lit if lit.op == "<=" else lit.negate()
                key = (canon.feature_idx, canon.threshold)
                seen.setdefault(key, canon)
        return tuple(seen.values())
