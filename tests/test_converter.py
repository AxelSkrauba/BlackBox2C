"""
Tests for main converter module.
"""

import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification

from blackbox2c import Converter, ConversionConfig, convert


class TestConverter:
    """Test Converter class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        X, y = make_classification(
            n_samples=150,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            n_repeated=0,
            n_classes=3,
            random_state=42
        )
        X_train, X_test = X[:100], X[100:]
        y_train, y_test = y[:100], y[100:]
        return X_train, X_test, y_train, y_test
    
    def test_converter_initialization(self):
        """Test Converter initialization."""
        config = ConversionConfig(max_depth=5)
        converter = Converter(config)
        
        assert converter.config.max_depth == 5
        assert converter.surrogate_tree_ is None
        assert converter.metrics_ == {}
    
    def test_convert_decision_tree(self, sample_data):
        """Test conversion of decision tree."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        converter = Converter()
        c_code = converter.convert(model, X_train, X_test=X_test)
        
        assert isinstance(c_code, str)
        assert len(c_code) > 0
        assert 'uint8_t predict(' in c_code
        
        # Check metrics
        metrics = converter.get_metrics()
        assert 'fidelity' in metrics
        assert metrics['fidelity'] == 1.0  # Perfect for decision tree
    
    def test_convert_random_forest(self, sample_data):
        """Test conversion of random forest."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        converter = Converter()
        c_code = converter.convert(model, X_train, X_test=X_test)
        
        assert isinstance(c_code, str)
        assert len(c_code) > 0
        
        # Check metrics
        metrics = converter.get_metrics()
        assert 'fidelity' in metrics
        assert metrics['fidelity'] > 0.8  # Should have good fidelity
    
    def test_convert_svm(self, sample_data):
        """Test conversion of SVM."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = SVC(kernel='rbf', random_state=42)
        model.fit(X_train, y_train)
        
        converter = Converter(ConversionConfig(max_depth=5, n_samples=1000))
        c_code = converter.convert(model, X_train, X_test=X_test)
        
        assert isinstance(c_code, str)
        assert len(c_code) > 0
        
        metrics = converter.get_metrics()
        assert 'fidelity' in metrics
        assert metrics['fidelity'] > 0.7  # Reasonable fidelity
    
    def test_convert_with_feature_names(self, sample_data):
        """Test conversion with custom feature names."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        feature_names = ['feat_a', 'feat_b', 'feat_c', 'feat_d']
        converter = Converter()
        c_code = converter.convert(model, X_train, feature_names=feature_names)
        
        # Feature names should appear in code
        assert 'feat_a' in c_code or 'features[0]' in c_code
    
    def test_convert_with_class_names(self, sample_data):
        """Test conversion with custom class names."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        class_names = ['LOW', 'MEDIUM', 'HIGH']
        converter = Converter()
        c_code = converter.convert(model, X_train, class_names=class_names)
        
        # Class names should appear in defines
        assert '#define LOW 0' in c_code
        assert '#define MEDIUM 1' in c_code
        assert '#define HIGH 2' in c_code
    
    def test_metrics_collection(self, sample_data):
        """Test that metrics are properly collected."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        converter = Converter()
        converter.convert(model, X_train, X_test=X_test)
        
        metrics = converter.get_metrics()
        
        assert 'fidelity' in metrics
        assert 'complexity' in metrics
        assert 'size_estimate' in metrics
        
        assert 'n_nodes' in metrics['complexity']
        assert 'flash_bytes' in metrics['size_estimate']
    
    def test_convenience_function(self, sample_data):
        """Test convenience convert() function."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        c_code = convert(
            model,
            X_train,
            X_test=X_test,
            max_depth=4,
            optimize_rules='medium'
        )
        
        assert isinstance(c_code, str)
        assert len(c_code) > 0
    
    # Validation tests
    def test_invalid_model(self, sample_data):
        """Test that invalid model raises error."""
        X_train, _, _, _ = sample_data
        
        converter = Converter()
        
        with pytest.raises(TypeError, match="Model must have a predict"):
            converter.convert("not a model", X_train)
    
    def test_invalid_X_train_shape(self, sample_data):
        """Test that invalid X_train shape raises error."""
        X_train, _, y_train, _ = sample_data
        
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        converter = Converter()
        
        # 1D array
        with pytest.raises(ValueError, match="must be 2-dimensional"):
            converter.convert(model, X_train[0])
    
    def test_too_few_samples(self, sample_data):
        """Test that too few samples raises error."""
        X_train, _, y_train, _ = sample_data
        
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        converter = Converter()
        
        with pytest.raises(ValueError, match="at least 5 samples"):
            converter.convert(model, X_train[:3])
    
    def test_nan_in_X_train(self, sample_data):
        """Test that NaN in X_train raises error."""
        X_train, _, y_train, _ = sample_data
        
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        X_train_nan = X_train.copy()
        X_train_nan[0, 0] = np.nan
        
        converter = Converter()
        
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            converter.convert(model, X_train_nan)
    
    def test_mismatched_X_test_features(self, sample_data):
        """Test that mismatched X_test features raises error."""
        X_train, X_test, y_train, _ = sample_data
        
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        X_test_wrong = X_test[:, :2]  # Only 2 features instead of 4
        
        converter = Converter()
        
        with pytest.raises(ValueError, match="same number of features"):
            converter.convert(model, X_train, X_test=X_test_wrong)
    
    def test_invalid_feature_names_length(self, sample_data):
        """Test that wrong number of feature names raises error."""
        X_train, _, y_train, _ = sample_data
        
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        converter = Converter()
        
        with pytest.raises(ValueError, match="Length of feature_names"):
            converter.convert(model, X_train, feature_names=['a', 'b'])  # Only 2 instead of 4
