"""
Rule optimization module.

Implements algorithms to simplify and optimize decision rules
extracted from models.
"""

import numpy as np
from sklearn.base import is_regressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from typing import List, Tuple, Set, Union

from ..tree_constants import TREE_LEAF, TREE_UNDEFINED

#: Relative tolerance used by the ``'high'`` level to merge sibling
#: regression leaves. A value of ``0.05`` mirrors the ``0.95`` cosine
#: similarity threshold used for classification: two leaves whose
#: predictions differ by up to 5 % are considered interchangeable.
REGRESSION_MERGE_RTOL = 0.05


def _is_regression_tree(tree) -> bool:
    """
    Return ``True`` when ``tree`` predicts a continuous target.

    ``sklearn.base.is_regressor`` is authoritative because it reads the
    estimator's ``_estimator_type``.  Non-conformant estimators (mocks and
    wrappers without that attribute) fall back to the node payload layout:
    regressors store a single scalar mean per node
    (``value.shape == (n_nodes, 1, 1)``), while classifiers store one entry
    per class.
    """
    try:
        return bool(is_regressor(tree))
    except AttributeError:
        return tree.tree_.value.shape[2] == 1


def _regression_value(tree_struct, node_id: int) -> float:
    """Return the scalar prediction stored at ``node_id``."""
    return float(tree_struct.value[node_id][0][0])


def _collapse_to_leaf(
    tree_struct, node_id: int, left_child: int, right_child: int, value
) -> None:
    """
    Turn ``node_id`` into a leaf that predicts ``value``.

    All four structural attributes must be updated consistently: setting only
    ``feature`` to the leaf sentinel leaves ``children_left``/``children_right``
    pointing at the original (distinct) children, which the platform exporters
    detect as an internal node and emit as ``features[-2]`` — an invalid
    negative array index in C (v0.2.0 bug).

    Parameters
    ----------
    tree_struct : sklearn.tree._tree.Tree
        The tree being mutated.
    node_id : int
        Node to convert into a leaf.
    left_child, right_child : int
        The node's children, used to accumulate the sample count.
    value : float or np.ndarray
        Prediction payload for the new leaf: a scalar mean for regressors or
        the summed class counts for classifiers.
    """
    tree_struct.value[node_id] = value
    tree_struct.n_node_samples[node_id] = (
        tree_struct.n_node_samples[left_child]
        + tree_struct.n_node_samples[right_child]
    )
    tree_struct.feature[node_id] = TREE_LEAF
    tree_struct.threshold[node_id] = TREE_LEAF
    tree_struct.children_left[node_id] = TREE_UNDEFINED
    tree_struct.children_right[node_id] = TREE_UNDEFINED


class RuleOptimizer:
    """
    Optimize decision tree rules to reduce redundancy and complexity.
    """
    
    def __init__(self, optimization_level: str = 'medium'):
        """
        Initialize the rule optimizer.
        
        Parameters
        ----------
        optimization_level : str
            One of 'low', 'medium', or 'high'.
        """
        self.optimization_level = optimization_level
    
    def optimize(
        self,
        tree: Union[DecisionTreeClassifier, DecisionTreeRegressor],
    ) -> Union[DecisionTreeClassifier, DecisionTreeRegressor]:
        """
        Optimize the decision tree by pruning and simplifying rules.
        
        Parameters
        ----------
        tree : DecisionTreeClassifier
            The tree to optimize.
        
        Returns
        -------
        DecisionTreeClassifier
            Optimized tree (may be the same object, modified in place).
        """
        if self.optimization_level == 'low':
            return tree

        # Advanced levels (qm/bdd/auto) introduced in v0.2 still benefit
        # from the medium-level pruning before they take over at the IR
        # layer; route them here to the same 'medium' code path.
        effective = self.optimization_level
        if effective in ('qm', 'bdd', 'auto'):
            effective = 'medium'

        # Apply optimizations based on level
        if effective in ('medium', 'high'):
            self._prune_redundant_branches(tree)

        if effective == 'high':
            self._merge_similar_leaves(tree)

        return tree
    
    def _prune_redundant_branches(
        self, tree: Union[DecisionTreeClassifier, DecisionTreeRegressor]
    ):
        """
        Prune branches that lead to the same outcome.

        A branch is redundant when both children are leaves that predict the
        same outcome. For classifiers the outcome is the argmax class; for
        regressors it is the predicted value, which must match exactly so the
        pruning leaves the model function bit-identical.

        This is a simplified version - in production, you'd want
        more sophisticated pruning based on cost-complexity.
        """
        tree_struct = tree.tree_
        is_regression = _is_regression_tree(tree)

        # Identify nodes whose two children are leaves with the same outcome
        for node_id in range(tree_struct.node_count):
            if tree_struct.feature[node_id] == TREE_LEAF:  # Skip leaves
                continue

            left_child = tree_struct.children_left[node_id]
            right_child = tree_struct.children_right[node_id]

            # Check if both children are leaves with same prediction
            if (tree_struct.feature[left_child] == TREE_LEAF and
                tree_struct.feature[right_child] == TREE_LEAF):

                if is_regression:
                    # A regressor stores one scalar mean per node, so
                    # ``argmax`` would always be 0 and every pair would look
                    # identical (v0.2.0 - v0.2.3 regression corruption).
                    # Two regression leaves are redundant only when they
                    # predict exactly the same value.
                    left_value = _regression_value(tree_struct, left_child)
                    right_value = _regression_value(tree_struct, right_child)
                    if left_value != right_value:
                        continue
                    merged_value = left_value
                else:
                    left_class = np.argmax(tree_struct.value[left_child][0])
                    right_class = np.argmax(tree_struct.value[right_child][0])
                    if left_class != right_class:
                        continue
                    # Classifiers store class counts, so summing the
                    # children's payloads reproduces the parent counts.
                    merged_value = (
                        tree_struct.value[left_child]
                        + tree_struct.value[right_child]
                    )

                _collapse_to_leaf(
                    tree_struct, node_id, left_child, right_child, merged_value
                )
    
    def _merge_similar_leaves(
        self, tree: Union[DecisionTreeClassifier, DecisionTreeRegressor]
    ):
        """
        Merge leaves with very similar predictions.
        
        This is useful when the surrogate tree has created
        unnecessary distinctions. Uses iterative approach to merge
        sibling leaves with similar class distributions (classifiers) or
        similar predicted values (:data:`REGRESSION_MERGE_RTOL`).
        """
        tree_struct = tree.tree_
        is_regression = _is_regression_tree(tree)
        changed = True
        merge_count = 0
        
        # Iterate until no more merges possible
        while changed:
            changed = False
            
            for node_id in range(tree_struct.node_count):
                # Skip leaves
                if tree_struct.feature[node_id] == TREE_LEAF:
                    continue

                left_child = tree_struct.children_left[node_id]
                right_child = tree_struct.children_right[node_id]

                # Both children must be leaves
                if (tree_struct.feature[left_child] == TREE_LEAF and
                    tree_struct.feature[right_child] == TREE_LEAF):

                    if is_regression:
                        # Merge sibling leaves whose means are close; the
                        # merged leaf must carry the sample-weighted mean,
                        # because a regressor stores the mean rather than a
                        # count (summing the two means would be wrong).
                        left_value = _regression_value(tree_struct, left_child)
                        right_value = _regression_value(tree_struct, right_child)
                        if not np.isclose(
                            left_value, right_value, rtol=REGRESSION_MERGE_RTOL, atol=0.0
                        ):
                            continue
                        left_count = tree_struct.n_node_samples[left_child]
                        right_count = tree_struct.n_node_samples[right_child]
                        merged_value = (
                            left_value * left_count + right_value * right_count
                        ) / (left_count + right_count)
                    else:
                        # Get class distributions
                        left_dist = tree_struct.value[left_child][0]
                        right_dist = tree_struct.value[right_child][0]

                        # Get predicted classes
                        left_class = np.argmax(left_dist)
                        right_class = np.argmax(right_dist)

                        # Only same-class siblings can be merged
                        if left_class != right_class:
                            continue

                        # Normalize to probabilities
                        left_probs = left_dist / (left_dist.sum() + 1e-10)
                        right_probs = right_dist / (right_dist.sum() + 1e-10)

                        # Calculate cosine similarity
                        similarity = np.dot(left_probs, right_probs) / (
                            np.linalg.norm(left_probs) * np.linalg.norm(right_probs) + 1e-10
                        )

                        # If very similar (>0.95), merge by converting parent to leaf
                        if similarity <= 0.95:
                            continue
                        merged_value = left_dist + right_dist

                    _collapse_to_leaf(
                        tree_struct, node_id, left_child, right_child, merged_value
                    )
                    changed = True
                    merge_count += 1
        
        return merge_count
    
    def analyze_complexity(
        self, tree: Union[DecisionTreeClassifier, DecisionTreeRegressor]
    ) -> dict:
        """
        Analyze the complexity of the decision tree.
        
        Returns
        -------
        dict
            Metrics about tree complexity.
        """
        tree_struct = tree.tree_
        
        n_nodes = tree_struct.node_count
        n_leaves = np.sum(tree_struct.feature == TREE_LEAF)
        max_depth = tree.get_depth()

        # Calculate average path length
        def get_path_lengths(node_id=0, depth=0):
            if tree_struct.feature[node_id] == TREE_LEAF:
                return [depth]
            
            left_paths = get_path_lengths(tree_struct.children_left[node_id], depth + 1)
            right_paths = get_path_lengths(tree_struct.children_right[node_id], depth + 1)
            
            return left_paths + right_paths
        
        path_lengths = get_path_lengths()
        avg_path_length = np.mean(path_lengths)
        
        return {
            'n_nodes': n_nodes,
            'n_leaves': n_leaves,
            'n_internal_nodes': n_nodes - n_leaves,
            'max_depth': max_depth,
            'avg_path_length': avg_path_length,
            'min_path_length': min(path_lengths),
            'max_path_length': max(path_lengths)
        }
    
    def get_feature_importance(
        self, tree: Union[DecisionTreeClassifier, DecisionTreeRegressor]
    ) -> np.ndarray:
        """
        Get feature importance from the tree.
        
        Returns
        -------
        np.ndarray
            Feature importance scores.
        """
        return tree.feature_importances_
    
    def suggest_simplifications(
        self, tree: Union[DecisionTreeClassifier, DecisionTreeRegressor]
    ) -> List[str]:
        """
        Suggest ways to simplify the tree further.
        
        Returns
        -------
        list
            List of suggestions as strings.
        """
        suggestions = []
        complexity = self.analyze_complexity(tree)
        
        if complexity['max_depth'] > 7:
            suggestions.append(
                f"Consider reducing max_depth (current: {complexity['max_depth']})"
            )
        
        if complexity['n_leaves'] > 50:
            suggestions.append(
                f"Tree has many leaves ({complexity['n_leaves']}). "
                "Consider reducing max_depth or increasing min_samples_leaf."
            )
        
        # Check for unbalanced tree
        if complexity['max_path_length'] > 2 * complexity['avg_path_length']:
            suggestions.append(
                "Tree is unbalanced. Consider using balanced tree construction."
            )
        
        # Check feature usage
        feature_importance = self.get_feature_importance(tree)
        n_unused = np.sum(feature_importance == 0)
        if n_unused > 0:
            suggestions.append(
                f"{n_unused} features are unused and could be removed."
            )
        
        if not suggestions:
            suggestions.append("Tree is well-optimized!")
        
        return suggestions
