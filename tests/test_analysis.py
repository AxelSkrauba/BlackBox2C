"""
Tests for feature analysis module.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from blackbox2c.analysis import FeatureSensitivityAnalyzer, SensitivityResults


class TestFeatureSensitivityAnalyzer:
    """Test FeatureSensitivityAnalyzer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data with varying feature importance."""
        # Create dataset where some features are more important
        X, y = make_classification(
            n_samples=200,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_repeated=0,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Train a model on sample data."""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X, y)
        return model
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = FeatureSensitivityAnalyzer(n_repeats=5, random_state=123)
        
        assert analyzer.n_repeats == 5
        assert analyzer.random_state == 123
        assert analyzer.results_ is None
    
    def test_analyze_basic(self, trained_model, sample_data):
        """Test basic analysis functionality."""
        X, y = sample_data
        
        analyzer = FeatureSensitivityAnalyzer(n_repeats=5)
        results = analyzer.analyze(trained_model, X, y)
        
        assert results is not None
        assert isinstance(results, SensitivityResults)
        assert len(results.importances) == 5  # 5 features
        assert len(results.importances_std) == 5
        assert results.baseline_score > 0
    
    def test_analyze_with_feature_names(self, trained_model, sample_data):
        """Test analysis with custom feature names."""
        X, y = sample_data
        feature_names = ['feat_a', 'feat_b', 'feat_c', 'feat_d', 'feat_e']
        
        analyzer = FeatureSensitivityAnalyzer()
        results = analyzer.analyze(trained_model, X, y, feature_names=feature_names)
        
        assert results.feature_names == feature_names
    
    def test_analyze_without_feature_names(self, trained_model, sample_data):
        """Test analysis generates default feature names."""
        X, y = sample_data
        
        analyzer = FeatureSensitivityAnalyzer()
        results = analyzer.analyze(trained_model, X, y)
        
        assert results.feature_names[0] == "feature_0"
        assert results.feature_names[4] == "feature_4"
    
    def test_importance_values_reasonable(self, trained_model, sample_data):
        """Test that importance values are in reasonable range."""
        X, y = sample_data
        
        analyzer = FeatureSensitivityAnalyzer(n_repeats=10)
        results = analyzer.analyze(trained_model, X, y)
        
        # Importances should be between -1 and 1 typically
        for imp in results.importances.values():
            assert -1.0 <= imp <= 1.0
        
        # Standard deviations should be non-negative
        for std in results.importances_std.values():
            assert std >= 0
    
    def test_reproducibility(self, trained_model, sample_data):
        """Test that analysis is reproducible with same random_state."""
        X, y = sample_data
        
        analyzer1 = FeatureSensitivityAnalyzer(random_state=42)
        results1 = analyzer1.analyze(trained_model, X, y)
        
        analyzer2 = FeatureSensitivityAnalyzer(random_state=42)
        results2 = analyzer2.analyze(trained_model, X, y)
        
        # Should produce identical results
        for idx in range(5):
            assert results1.importances[idx] == results2.importances[idx]
            assert results1.importances_std[idx] == results2.importances_std[idx]
    
    def test_invalid_model(self, sample_data):
        """Test that invalid model raises error."""
        X, y = sample_data
        
        analyzer = FeatureSensitivityAnalyzer()
        
        with pytest.raises(TypeError, match="must have a score"):
            analyzer.analyze("not a model", X, y)
    
    def test_invalid_X_shape(self, trained_model):
        """Test that invalid X shape raises error."""
        analyzer = FeatureSensitivityAnalyzer()
        
        # 1D array
        with pytest.raises(ValueError, match="must be 2D"):
            analyzer.analyze(trained_model, np.array([1, 2, 3]), np.array([0, 1, 0]))
    
    def test_mismatched_lengths(self, trained_model, sample_data):
        """Test that mismatched X and y lengths raise error."""
        X, y = sample_data
        
        analyzer = FeatureSensitivityAnalyzer()
        
        with pytest.raises(ValueError, match="same length"):
            analyzer.analyze(trained_model, X, y[:10])
    
    def test_wrong_feature_names_length(self, trained_model, sample_data):
        """Test that wrong feature_names length raises error."""
        X, y = sample_data
        
        analyzer = FeatureSensitivityAnalyzer()
        
        with pytest.raises(ValueError, match="feature_names length"):
            analyzer.analyze(trained_model, X, y, feature_names=['a', 'b'])


class TestSensitivityResults:
    """Test SensitivityResults class."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        feature_names = ['temp', 'humidity', 'pressure', 'wind', 'light']
        importances = {
            0: 0.85,  # Critical
            1: 0.15,  # Medium
            2: 0.005, # Very low
            3: 0.30,  # High
            4: 0.02   # Low
        }
        importances_std = {i: 0.01 for i in range(5)}
        baseline_score = 0.95
        
        return SensitivityResults(
            feature_names=feature_names,
            importances=importances,
            importances_std=importances_std,
            baseline_score=baseline_score
        )
    
    def test_summary_generation(self, sample_results):
        """Test summary generation."""
        summary = sample_results.summary()
        
        assert isinstance(summary, str)
        assert "Feature Sensitivity Analysis" in summary
        assert "temp" in summary
        assert "Critical" in summary
        assert "Recommendations" in summary
    
    def test_get_optimal_subset_default(self, sample_results):
        """Test optimal subset with default threshold."""
        selected = sample_results.get_optimal_subset(threshold=0.01)
        
        # Should exclude feature 2 (importance = 0.005)
        assert 2 not in selected
        
        # Should include features 0, 1, 3, 4
        assert 0 in selected
        assert 1 in selected
        assert 3 in selected
        assert 4 in selected
    
    def test_get_optimal_subset_high_threshold(self, sample_results):
        """Test optimal subset with high threshold."""
        selected = sample_results.get_optimal_subset(threshold=0.2)
        
        # Should only include features 0 and 3
        assert len(selected) == 2
        assert 0 in selected
        assert 3 in selected
    
    def test_get_optimal_subset_min_features(self, sample_results):
        """Test optimal subset respects min_features."""
        # Very high threshold would normally select nothing
        selected = sample_results.get_optimal_subset(threshold=0.9, min_features=2)
        
        # Should still return 2 features (top 2)
        assert len(selected) == 2
        assert 0 in selected  # temp (0.85)
        assert 3 in selected  # wind (0.30)
    
    def test_get_top_features(self, sample_results):
        """Test getting top N features."""
        top_3 = sample_results.get_top_features(n=3)
        
        assert len(top_3) == 3
        
        # Should be sorted by importance (descending)
        assert top_3[0][0] == 0  # temp (0.85)
        assert top_3[1][0] == 3  # wind (0.30)
        assert top_3[2][0] == 1  # humidity (0.15)
        
        # Check structure
        idx, name, importance = top_3[0]
        assert idx == 0
        assert name == 'temp'
        assert importance == 0.85
    
    def test_get_redundant_features(self, sample_results):
        """Test getting redundant features."""
        redundant = sample_results.get_redundant_features(threshold=0.01)
        
        # Only feature 2 (pressure, 0.005) should be redundant
        assert len(redundant) == 1
        assert 2 in redundant
    
    def test_get_redundant_features_high_threshold(self, sample_results):
        """Test redundant features with high threshold."""
        redundant = sample_results.get_redundant_features(threshold=0.1)
        
        # Features 2 (0.005), 4 (0.02) should be redundant
        assert len(redundant) == 2
        assert 2 in redundant
        assert 4 in redundant


class TestIntegrationWithIris:
    """Integration tests with real Iris dataset."""
    
    def test_iris_analysis(self):
        """Test analysis on Iris dataset."""
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Train model
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X, y)
        
        # Analyze
        analyzer = FeatureSensitivityAnalyzer(n_repeats=5)
        results = analyzer.analyze(model, X, y, feature_names=iris.feature_names)
        
        # Check results
        assert len(results.importances) == 4
        assert results.baseline_score > 0.9  # Iris is easy to classify
        
        # Petal length and width should be most important for Iris
        top_features = results.get_top_features(2)
        top_names = [name for _, name, _ in top_features]
        
        # At least one petal feature should be in top 2
        assert any('petal' in name for name in top_names)
    
    def test_iris_feature_selection(self):
        """Test feature selection reduces features while maintaining accuracy."""
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Train full model
        model_full = RandomForestClassifier(n_estimators=10, random_state=42)
        model_full.fit(X, y)
        score_full = model_full.score(X, y)
        
        # Analyze and select features
        analyzer = FeatureSensitivityAnalyzer()
        results = analyzer.analyze(model_full, X, y)
        selected = results.get_optimal_subset(threshold=0.05)
        
        # Should reduce features
        assert len(selected) < 4
        
        # Train reduced model
        X_reduced = X[:, selected]
        model_reduced = RandomForestClassifier(n_estimators=10, random_state=42)
        model_reduced.fit(X_reduced, y)
        score_reduced = model_reduced.score(X_reduced, y)
        
        # Should maintain reasonable accuracy (within 10%)
        assert score_reduced >= score_full - 0.1
