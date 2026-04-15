"""
Tests for rule optimization module.
"""

import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

from blackbox2c.optimizer import RuleOptimizer


class TestRuleOptimizer:
    """Test RuleOptimizer class."""
    
    @pytest.fixture
    def sample_tree(self):
        """Create a sample decision tree."""
        X, y = make_classification(
            n_samples=200,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            n_repeated=0,
            n_classes=3,
            random_state=42
        )
        
        tree = DecisionTreeClassifier(max_depth=6, random_state=42)
        tree.fit(X, y)
        return tree
    
    def test_initialization(self):
        """Test RuleOptimizer initialization."""
        optimizer = RuleOptimizer(optimization_level='high')
        assert optimizer.optimization_level == 'high'
    
    def test_low_optimization(self, sample_tree):
        """Test that low optimization doesn't modify tree."""
        optimizer = RuleOptimizer(optimization_level='low')
        
        original_nodes = sample_tree.tree_.node_count
        optimized_tree = optimizer.optimize(sample_tree)
        
        # Low optimization should return tree as-is
        assert optimized_tree.tree_.node_count == original_nodes
    
    def test_medium_optimization(self, sample_tree):
        """Test medium optimization level."""
        optimizer = RuleOptimizer(optimization_level='medium')
        
        original_nodes = sample_tree.tree_.node_count
        optimized_tree = optimizer.optimize(sample_tree)
        
        # Medium should apply pruning
        assert optimized_tree is not None
        # Node count may be same or less (depends on tree structure)
        assert optimized_tree.tree_.node_count <= original_nodes
    
    def test_high_optimization(self, sample_tree):
        """Test high optimization level with leaf merging."""
        optimizer = RuleOptimizer(optimization_level='high')
        
        original_nodes = sample_tree.tree_.node_count
        optimized_tree = optimizer.optimize(sample_tree)
        
        # High should apply both pruning and merging
        assert optimized_tree is not None
        assert optimized_tree.tree_.node_count <= original_nodes
    
    def test_complexity_analysis(self, sample_tree):
        """Test complexity analysis."""
        optimizer = RuleOptimizer()
        complexity = optimizer.analyze_complexity(sample_tree)
        
        assert 'n_nodes' in complexity
        assert 'n_leaves' in complexity
        assert 'n_internal_nodes' in complexity
        assert 'max_depth' in complexity
        assert 'avg_path_length' in complexity
        assert 'min_path_length' in complexity
        assert 'max_path_length' in complexity
        
        # Sanity checks
        assert complexity['n_nodes'] > 0
        assert complexity['n_leaves'] > 0
        assert complexity['n_internal_nodes'] >= 0
        assert complexity['n_nodes'] == complexity['n_leaves'] + complexity['n_internal_nodes']
        assert complexity['max_depth'] >= 0
        assert complexity['avg_path_length'] > 0
    
    def test_feature_importance(self, sample_tree):
        """Test feature importance extraction."""
        optimizer = RuleOptimizer()
        importance = optimizer.get_feature_importance(sample_tree)
        
        assert isinstance(importance, np.ndarray)
        assert len(importance) == 4  # 4 features
        assert np.all(importance >= 0)
        assert np.isclose(importance.sum(), 1.0)  # Should sum to 1
    
    def test_simplification_suggestions(self, sample_tree):
        """Test simplification suggestions."""
        optimizer = RuleOptimizer()
        suggestions = optimizer.suggest_simplifications(sample_tree)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(s, str) for s in suggestions)
    
    def test_prune_redundant_branches(self, sample_tree):
        """Test redundant branch pruning."""
        optimizer = RuleOptimizer(optimization_level='medium')
        
        # This is hard to test directly, but we can check it doesn't crash
        optimizer._prune_redundant_branches(sample_tree)
        
        # Tree should still be valid
        assert sample_tree.tree_.node_count > 0
    
    def test_merge_similar_leaves(self, sample_tree):
        """Test similar leaf merging."""
        optimizer = RuleOptimizer(optimization_level='high')
        
        original_leaves = np.sum(sample_tree.tree_.feature == -2)
        merge_count = optimizer._merge_similar_leaves(sample_tree)
        
        # Should return number of merges
        assert isinstance(merge_count, (int, np.integer))
        assert merge_count >= 0
        
        # Note: Leaf count may not decrease if no similar leaves found
        # This is expected behavior
        new_leaves = np.sum(sample_tree.tree_.feature == -2)
        assert new_leaves <= original_leaves + merge_count  # Can increase due to parent becoming leaf
