"""
Tests for surrogate extraction module.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

from blackbox2c.surrogate import SurrogateExtractor


class TestSurrogateExtractor:
    """Test SurrogateExtractor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            n_repeated=0,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42
        )
        return X, y
    
    def test_initialization(self):
        """Test SurrogateExtractor initialization."""
        extractor = SurrogateExtractor(max_depth=5, n_samples=1000)
        
        assert extractor.max_depth == 5
        assert extractor.n_samples == 1000
        assert extractor.random_state == 42
        assert extractor.surrogate_tree_ is None
    
    def test_extract_from_random_forest(self, sample_data):
        """Test extraction from Random Forest model."""
        X, y = sample_data
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        rf.fit(X, y)
        
        # Extract surrogate
        extractor = SurrogateExtractor(max_depth=5, n_samples=1000)
        surrogate = extractor.extract(rf, X)
        
        assert surrogate is not None
        assert isinstance(surrogate, DecisionTreeClassifier)
        assert surrogate.get_depth() <= 5
        assert extractor.n_features_ == 4
        assert extractor.n_classes_ == 3
    
    def test_extract_from_svm(self, sample_data):
        """Test extraction from SVM model."""
        X, y = sample_data
        
        # Train SVM
        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(X, y)
        
        # Extract surrogate
        extractor = SurrogateExtractor(max_depth=5, n_samples=1000)
        surrogate = extractor.extract(svm, X)
        
        assert surrogate is not None
        assert isinstance(surrogate, DecisionTreeClassifier)
    
    def test_fidelity_calculation(self, sample_data):
        """Test fidelity calculation between original and surrogate."""
        X, y = sample_data
        X_train, X_test = X[:150], X[150:]
        y_train = y[:150]
        
        # Train model
        rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        
        # Extract surrogate
        extractor = SurrogateExtractor(max_depth=5, n_samples=1000)
        extractor.extract(rf, X_train)
        
        # Calculate fidelity
        fidelity = extractor.get_fidelity(rf, X_test)
        
        assert 0.0 <= fidelity <= 1.0
        assert fidelity > 0.8  # Should have reasonable agreement
    
    def test_fidelity_without_extraction(self):
        """Test that fidelity raises error if extract not called."""
        extractor = SurrogateExtractor()
        
        with pytest.raises(ValueError, match="Must call extract"):
            extractor.get_fidelity(None, np.array([[1, 2, 3]]))
    
    def test_boundary_sample_generation(self, sample_data):
        """Test boundary sample generation."""
        X, y = sample_data
        
        extractor = SurrogateExtractor(n_samples=1000)
        extractor.n_features_ = X.shape[1]  # Initialize before calling private method
        X_synthetic = extractor._generate_boundary_samples(X)
        
        assert X_synthetic.shape[0] == 1000
        assert X_synthetic.shape[1] == X.shape[1]
        
        # Check that samples are within reasonable range
        feature_range = X.max() - X.min()
        assert X_synthetic.min() >= X.min() - 0.2 * feature_range
        assert X_synthetic.max() <= X.max() + 0.2 * feature_range
    
    def test_feature_names(self, sample_data):
        """Test feature name handling."""
        X, y = sample_data
        
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(X, y)
        
        # Without feature names
        extractor = SurrogateExtractor()
        extractor.extract(rf, X)
        assert len(extractor.feature_names_) == 4
        assert extractor.feature_names_[0] == "feature_0"
        
        # With feature names
        feature_names = ['feat_a', 'feat_b', 'feat_c', 'feat_d']
        extractor = SurrogateExtractor()
        extractor.extract(rf, X, feature_names=feature_names)
        assert extractor.feature_names_ == feature_names
