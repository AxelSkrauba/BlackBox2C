"""
Rule optimization module.

Implements algorithms to simplify and optimize decision rules
extracted from models.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from typing import List, Tuple, Set, Union


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
        
        # Apply optimizations based on level
        if self.optimization_level in ['medium', 'high']:
            self._prune_redundant_branches(tree)
        
        if self.optimization_level == 'high':
            self._merge_similar_leaves(tree)
        
        return tree
    
    def _prune_redundant_branches(
        self, tree: Union[DecisionTreeClassifier, DecisionTreeRegressor]
    ):
        """
        Prune branches that lead to the same outcome.
        
        This is a simplified version - in production, you'd want
        more sophisticated pruning based on cost-complexity.
        """
        tree_struct = tree.tree_
        
        # Identify nodes where both children lead to same class
        for node_id in range(tree_struct.node_count):
            if tree_struct.feature[node_id] == -2:  # Skip leaves
                continue
            
            left_child = tree_struct.children_left[node_id]
            right_child = tree_struct.children_right[node_id]
            
            # Check if both children are leaves with same prediction
            if (tree_struct.feature[left_child] == -2 and
                tree_struct.feature[right_child] == -2):
                
                left_class = np.argmax(tree_struct.value[left_child][0])
                right_class = np.argmax(tree_struct.value[right_child][0])
                
                if left_class == right_class:
                    # Convert this node to a leaf
                    # (This is a simplified approach)
                    tree_struct.feature[node_id] = -2
    
    def _merge_similar_leaves(
        self, tree: Union[DecisionTreeClassifier, DecisionTreeRegressor]
    ):
        """
        Merge leaves with very similar predictions.
        
        This is useful when the surrogate tree has created
        unnecessary distinctions. Uses iterative approach to merge
        sibling leaves with similar class distributions.
        """
        tree_struct = tree.tree_
        changed = True
        merge_count = 0
        
        # Iterate until no more merges possible
        while changed:
            changed = False
            
            for node_id in range(tree_struct.node_count):
                # Skip leaves
                if tree_struct.feature[node_id] == -2:
                    continue
                
                left_child = tree_struct.children_left[node_id]
                right_child = tree_struct.children_right[node_id]
                
                # Both children must be leaves
                if (tree_struct.feature[left_child] == -2 and 
                    tree_struct.feature[right_child] == -2):
                    
                    # Get class distributions
                    left_dist = tree_struct.value[left_child][0]
                    right_dist = tree_struct.value[right_child][0]
                    
                    # Get predicted classes
                    left_class = np.argmax(left_dist)
                    right_class = np.argmax(right_dist)
                    
                    # If same class, check similarity
                    if left_class == right_class:
                        # Normalize to probabilities
                        left_probs = left_dist / (left_dist.sum() + 1e-10)
                        right_probs = right_dist / (right_dist.sum() + 1e-10)
                        
                        # Calculate cosine similarity
                        similarity = np.dot(left_probs, right_probs) / (
                            np.linalg.norm(left_probs) * np.linalg.norm(right_probs) + 1e-10
                        )
                        
                        # If very similar (>0.95), merge by converting parent to leaf
                        if similarity > 0.95:
                            # Combine distributions
                            tree_struct.value[node_id] = np.array([[left_dist + right_dist]])
                            # Mark as leaf
                            tree_struct.feature[node_id] = -2
                            tree_struct.threshold[node_id] = -2
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
        n_leaves = np.sum(tree_struct.feature == -2)
        max_depth = tree.get_depth()
        
        # Calculate average path length
        def get_path_lengths(node_id=0, depth=0):
            if tree_struct.feature[node_id] == -2:
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
