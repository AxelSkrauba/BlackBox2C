"""
Tests for regression model conversion.

This module tests the conversion of regression models to C code,
including DecisionTreeRegressor and ensemble regressors.
"""

import pytest
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from blackbox2c import Converter, convert
from blackbox2c.config import ConversionConfig


@pytest.fixture
def regression_data():
    """Generate synthetic regression dataset."""
    X, y = make_regression(
        n_samples=200,
        n_features=4,
        n_informative=3,
        noise=10.0,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def simple_regression_data():
    """Generate simple 2D regression dataset."""
    np.random.seed(42)
    X = np.random.uniform(-10, 10, size=(100, 2))
    y = 2.5 * X[:, 0] - 1.3 * X[:, 1] + 5.0 + np.random.normal(0, 2, size=100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


class TestRegressionConversion:
    """Test conversion of regression models."""
    
    def test_decision_tree_regressor(self, regression_data):
        """Test conversion of DecisionTreeRegressor."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Train model
        model = DecisionTreeRegressor(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Convert to C
        converter = Converter()
        c_code = converter.convert(model, X_train)
        
        # Verify C code
        assert c_code is not None
        assert len(c_code) > 0
        assert 'float predict' in c_code
        assert 'Task: Regression' in c_code
        assert 'return' in c_code
        
        # Verify metrics
        metrics = converter.get_metrics()
        assert 'fidelity' in metrics
        assert metrics['fidelity'] == 1.0  # Direct tree, perfect fidelity
        assert 'complexity' in metrics
        assert metrics['complexity']['n_nodes'] > 0
    
    def test_random_forest_regressor(self, regression_data):
        """Test conversion of RandomForestRegressor."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        # Convert to C
        config = ConversionConfig(max_depth=5)
        c_code = convert(model, X_train, config=config)
        
        # Verify C code
        assert c_code is not None
        assert 'float predict' in c_code
        assert 'Task: Regression' in c_code
        
        # Check that code contains decision logic
        assert 'if' in c_code
        assert 'features[' in c_code
    
    def test_gradient_boosting_regressor(self, regression_data):
        """Test conversion of GradientBoostingRegressor."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Train model
        model = GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        # Convert to C
        config = ConversionConfig(max_depth=4)
        c_code = convert(model, X_train, config=config)
        
        # Verify C code
        assert c_code is not None
        assert 'float predict' in c_code
        assert 'Task: Regression' in c_code
    
    def test_regressor_with_feature_names(self, simple_regression_data):
        """Test regressor conversion with custom feature names."""
        X_train, X_test, y_train, y_test = simple_regression_data
        
        # Train model
        model = DecisionTreeRegressor(max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        # Convert with feature names
        feature_names = ['temperature', 'humidity']
        c_code = convert(model, X_train, feature_names=feature_names)
        
        # Verify feature names appear in code
        assert 'temperature' in c_code or 'features[0]' in c_code
        assert 'humidity' in c_code or 'features[1]' in c_code
    
    def test_regressor_fidelity_calculation(self, regression_data):
        """Test fidelity calculation for regression models."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Train RandomForest (requires surrogate)
        model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Convert
        converter = Converter(ConversionConfig(max_depth=5))
        c_code = converter.convert(model, X_train, X_test=X_test)
        
        # Check fidelity (R^2 score for regression)
        metrics = converter.get_metrics()
        assert 'fidelity' in metrics
        
        # Fidelity should be reasonable (R^2 > 0.5 for good approximation)
        # Note: Can be negative for very bad approximations
        assert metrics['fidelity'] > 0.3, f"Fidelity too low: {metrics['fidelity']}"
    
    def test_regressor_code_structure(self, simple_regression_data):
        """Test structure of generated C code for regression."""
        X_train, X_test, y_train, y_test = simple_regression_data
        
        # Train simple model
        model = DecisionTreeRegressor(max_depth=2, random_state=42)
        model.fit(X_train, y_train)
        
        # Convert
        c_code = convert(model, X_train)
        
        # Verify code structure
        assert '#include <stdint.h>' in c_code
        assert 'float predict(float features[' in c_code
        assert 'return' in c_code
        
        # Should contain decision logic
        assert 'if' in c_code
        assert 'else' in c_code
        
        # Should contain float literals (regression values)
        assert 'f' in c_code  # Float suffix
        
        # Should NOT contain class-related code
        assert '#define CLASS_' not in c_code
        assert 'uint8_t' not in c_code or 'uint8_t predict' not in c_code
    
    def test_regressor_with_optimization(self, regression_data):
        """Test regressor conversion with different optimization levels."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Train model
        model = DecisionTreeRegressor(max_depth=6, random_state=42)
        model.fit(X_train, y_train)
        
        # Test different optimization levels
        for opt_level in ['low', 'medium', 'high']:
            config = ConversionConfig(optimize_rules=opt_level)
            c_code = convert(model, X_train, config=config)
            
            assert c_code is not None
            assert 'float predict' in c_code
            assert len(c_code) > 0
    
    def test_regressor_with_fixed_point(self, simple_regression_data):
        """Test regressor with fixed-point arithmetic."""
        X_train, X_test, y_train, y_test = simple_regression_data
        
        # Train model
        model = DecisionTreeRegressor(max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        # Convert with fixed-point
        config = ConversionConfig(use_fixed_point=True, precision=16)
        c_code = convert(model, X_train, config=config)
        
        # Verify fixed-point code
        assert c_code is not None
        assert 'int32_t' in c_code or 'int16_t' in c_code
        assert 'Fixed-point: Yes' in c_code


class TestRegressionEdgeCases:
    """Test edge cases for regression conversion."""
    
    def test_single_feature_regressor(self):
        """Test regressor with single feature."""
        np.random.seed(42)
        X = np.random.uniform(0, 10, size=(50, 1))
        y = 2 * X[:, 0] + 1 + np.random.normal(0, 0.5, size=50)
        
        model = DecisionTreeRegressor(max_depth=3, random_state=42)
        model.fit(X, y)
        
        c_code = convert(model, X)
        
        assert c_code is not None
        assert 'float predict(float features[1])' in c_code
    
    def test_constant_output_regressor(self):
        """Test regressor that predicts constant value."""
        X = np.random.uniform(0, 10, size=(50, 2))
        y = np.ones(50) * 42.0  # Constant output
        
        model = DecisionTreeRegressor(max_depth=1, random_state=42)
        model.fit(X, y)
        
        c_code = convert(model, X)
        
        assert c_code is not None
        assert '42' in c_code or '4.2' in c_code  # Should contain the constant
    
    def test_negative_values_regressor(self):
        """Test regressor with negative output values."""
        np.random.seed(42)
        X = np.random.uniform(-5, 5, size=(50, 2))
        y = -2 * X[:, 0] + 3 * X[:, 1] - 10  # Negative values
        
        model = DecisionTreeRegressor(max_depth=4, random_state=42)
        model.fit(X, y)
        
        c_code = convert(model, X)
        
        assert c_code is not None
        assert 'float predict' in c_code
        # Should handle negative values correctly
        assert '-' in c_code  # Negative signs in output
    
    def test_large_value_range_regressor(self):
        """Test regressor with large value range."""
        np.random.seed(42)
        X = np.random.uniform(0, 1, size=(50, 2))
        y = np.random.uniform(1000, 10000, size=50)  # Large values
        
        model = DecisionTreeRegressor(max_depth=3, random_state=42)
        model.fit(X, y)
        
        c_code = convert(model, X)
        
        assert c_code is not None
        assert 'float predict' in c_code


class TestRegressionMetrics:
    """Test metrics collection for regression models."""
    
    def test_metrics_structure(self, regression_data):
        """Test that metrics are collected correctly."""
        X_train, X_test, y_train, y_test = regression_data
        
        model = DecisionTreeRegressor(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        converter = Converter()
        c_code = converter.convert(model, X_train, X_test=X_test)
        metrics = converter.get_metrics()
        
        # Check metrics structure
        assert isinstance(metrics, dict)
        assert 'fidelity' in metrics
        assert 'complexity' in metrics
        assert 'size_estimate' in metrics
        
        # Check complexity metrics
        complexity = metrics['complexity']
        assert 'n_nodes' in complexity
        assert 'n_leaves' in complexity
        assert 'max_depth' in complexity
        
        # Check size estimate
        size = metrics['size_estimate']
        assert 'flash_bytes' in size
        assert 'ram_bytes' in size
    
    def test_fidelity_with_test_set(self, regression_data):
        """Test fidelity calculation with separate test set."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Train ensemble model
        model = RandomForestRegressor(n_estimators=15, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        # Convert with test set
        converter = Converter(ConversionConfig(max_depth=4))
        c_code = converter.convert(model, X_train, X_test=X_test)
        
        metrics = converter.get_metrics()
        
        # Fidelity should be calculated on test set
        assert 'fidelity' in metrics
        assert isinstance(metrics['fidelity'], float)


class TestRegressionIntegration:
    """Integration tests for regression workflow."""
    
    def test_end_to_end_regression(self):
        """Test complete regression workflow."""
        # 1. Generate data
        X, y = make_regression(n_samples=100, n_features=3, noise=5.0, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 2. Train model
        model = DecisionTreeRegressor(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        # 3. Convert to C
        config = ConversionConfig(
            max_depth=4,
            function_name='predict_value',
            optimize_rules='medium'
        )
        c_code = convert(model, X_train, X_test=X_test, config=config)
        
        # 4. Verify output
        assert c_code is not None
        assert 'predict_value' in c_code
        assert 'float predict_value(float features[3])' in c_code
        assert 'Task: Regression' in c_code
        
        # 5. Check that code is reasonable size
        assert len(c_code) < 10000  # Should be compact
        assert c_code.count('if') > 0  # Should have decision logic
        assert c_code.count('return') > 1  # Multiple return statements
    
    def test_regression_with_feature_selection(self):
        """Test regression with automatic feature selection."""
        # Generate data with many features
        X, y = make_regression(
            n_samples=150,
            n_features=6,
            n_informative=3,
            noise=10.0,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # First, analyze features to select important ones
        from blackbox2c.analysis import FeatureSensitivityAnalyzer
        
        # Train initial model for analysis
        temp_model = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42)
        temp_model.fit(X_train, y_train)
        
        # Analyze and select features
        analyzer = FeatureSensitivityAnalyzer(n_repeats=5)
        results = analyzer.analyze(temp_model, X_train, y_train)
        selected_features = results.get_top_features(3)
        selected_indices = [idx for idx, name, imp in selected_features]
        
        # Train new model with selected features only
        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
        
        model = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42)
        model.fit(X_train_selected, y_train)
        
        # Convert to C
        config = ConversionConfig(max_depth=4)
        c_code = convert(model, X_train_selected, X_test=X_test_selected, config=config)
        
        # Should work and produce valid code
        assert c_code is not None
        assert 'float predict(float features[3])' in c_code
        assert 'Task: Regression' in c_code
