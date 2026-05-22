"""
Conversion between scikit-learn decision trees and the BlackBox2C IR.

For now this module exposes only the forward direction
(:func:`from_sklearn_tree`).  The reverse direction is implemented in
:mod:`blackbox2c.optimizer.codegen_bridge` (added in a later phase) so
that downstream code-generation can consume optimized RuleSets directly
without round-tripping through the fragile ``sklearn.tree._tree.Tree``
internal representation.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .ir import Conjunction, Literal, RuleSet


# Sentinel used by sklearn to mark leaf nodes inside ``Tree.feature``.
_TREE_LEAF = -2


def from_sklearn_tree(
    estimator,
    feature_names: Optional[Iterable[str]] = None,
) -> RuleSet:
    """Extract a DNF :class:`RuleSet` from a fitted sklearn decision tree.

    Parameters
    ----------
    estimator : DecisionTreeClassifier or DecisionTreeRegressor
        Any fitted sklearn tree-shaped estimator exposing ``tree_``,
        ``n_features_in_`` and (for classifiers) ``n_classes_``.
    feature_names : iterable of str, optional
        Optional names attached to the resulting RuleSet.  Length must
        equal ``estimator.n_features_in_``.

    Returns
    -------
    RuleSet
        A RuleSet whose ``predict`` is functionally identical to
        ``estimator.predict`` for any input within the tree's domain.

    Notes
    -----
    The traversal is iterative (explicit stack) rather than recursive,
    so trees of arbitrary depth are supported without hitting Python's
    recursion limit.
    """
    if not hasattr(estimator, "tree_"):
        raise TypeError(
            "from_sklearn_tree expects a fitted sklearn tree estimator "
            "(got an object without 'tree_' attribute)"
        )

    tree = estimator.tree_
    is_classifier = hasattr(estimator, "classes_")
    n_classes: Optional[int]
    if is_classifier:
        # ``n_classes_`` may be a numpy scalar or a 1-element array for
        # multi-output trees; we only support single-output here.
        nc = estimator.n_classes_
        n_classes = int(nc if np.ndim(nc) == 0 else nc[0])
    else:
        n_classes = None

    n_features = int(estimator.n_features_in_)
    if feature_names is not None:
        names = tuple(str(name) for name in feature_names)
        if len(names) != n_features:
            raise ValueError(
                f"feature_names has {len(names)} entries but the "
                f"estimator was fitted on {n_features} features"
            )
    else:
        names = None

    feature = tree.feature
    threshold = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    value = tree.value

    rules: list[Conjunction] = []

    # Iterative DFS, left first so rules are emitted in the same order
    # as a top-down if/else cascade would visit them.
    # Stack entry: (node_id, accumulated literals as a tuple).
    stack: list[tuple[int, tuple[Literal, ...]]] = [(0, ())]
    while stack:
        node, lits = stack.pop()
        feat = int(feature[node])

        if feat == _TREE_LEAF:
            if is_classifier:
                pred: int | float = int(np.argmax(value[node, 0]))
            else:
                pred = float(value[node, 0, 0])
            rules.append(Conjunction(lits, pred))
            continue

        thr = float(threshold[node])
        right = int(children_right[node])
        left = int(children_left[node])
        # Push right first so left is processed first (LIFO stack).
        stack.append((right, lits + (Literal(feat, thr, ">"),)))
        stack.append((left, lits + (Literal(feat, thr, "<="),)))

    return RuleSet(
        rules=tuple(rules),
        n_features=n_features,
        n_classes=n_classes,
        feature_names=names,
    )
