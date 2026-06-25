"""
Shared scikit-learn tree sentinels and leaf-detection helpers.

scikit-learn's ``sklearn.tree._tree.Tree`` marks leaf nodes with two
independent sentinels that are *both* expected to hold for a genuine
leaf:

* ``Tree.feature[node] == -2``  (``TREE_LEAF``)
* ``Tree.children_left[node] == Tree.children_right[node] == -1``
  (``TREE_UNDEFINED``)

Historically, different modules in BlackBox2C checked only one of the
two signals, which silently broke when an optimizer mutated one
attribute without updating the other (see v0.2.0 ``features[-2]``
bug).  Centralising the sentinels and providing a single
:func:`is_leaf` helper that accepts *either* signal makes the
codebase robust against future inconsistencies.
"""

from __future__ import annotations

#: Sentinel stored in ``Tree.feature`` for leaf nodes.
TREE_LEAF: int = -2

#: Sentinel stored in ``Tree.children_left`` / ``Tree.children_right``
#: for leaf nodes (sklearn's ``TREE_UNDEFINED``).
TREE_UNDEFINED: int = -1


def is_leaf(feature, children_left, children_right, node_id: int) -> bool:
    """Return ``True`` if ``node_id`` is a leaf.

    A node is considered a leaf when **either** of the two sklearn
    conventions holds: ``feature[node_id] == TREE_LEAF`` *or*
    ``children_left[node_id] == children_right[node_id]``.  For a
    well-formed tree both are true simultaneously; accepting either
    makes detection robust against partially-mutated nodes.

    Parameters
    ----------
    feature : array-like
        ``Tree.feature`` array (or the equivalent dict entry).
    children_left, children_right : array-like
        ``Tree.children_left`` / ``Tree.children_right`` arrays.
    node_id : int
        Index of the node to test.
    """
    return (
        feature[node_id] == TREE_LEAF
        or children_left[node_id] == children_right[node_id]
    )
