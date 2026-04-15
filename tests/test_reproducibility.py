"""
Reproducibility tests using Iris dataset to ensure consistency across runs.

These tests verify that the framework produces consistent results with the same
random_state, preventing regressions in the codebase.
"""

import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from blackbox2c import convert, Converter, ConversionConfig


class TestIrisReproducibility:
    """Reproducibility tests with Iris dataset to prevent software regressions."""
    
    @pytest.fixture
    def iris_data(self):
        """Load and split Iris dataset."""
        iris = load_iris()
        X, y = iris.data, iris.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, iris.feature_names, iris.target_names
    
    def test_decision_tree_conversion(self, iris_data):
        """Test decision tree conversion produces valid C code."""
        X_train, X_test, y_train, y_test, feature_names, class_names = iris_data
        
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        c_code = convert(model, X_train, X_test=X_test, max_depth=5)
        
        # Verify code structure
        assert '#include <stdint.h>' in c_code
        assert 'uint8_t predict(' in c_code
        assert 'return' in c_code
        
        # Verify it's not empty
        assert len(c_code) > 100
    
    def test_random_forest_fidelity(self, iris_data):
        """Test that Random Forest conversion maintains high fidelity."""
        X_train, X_test, y_train, y_test, _, _ = iris_data
        
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        converter = Converter(ConversionConfig(max_depth=5, n_samples=5000))
        c_code = converter.convert(model, X_train, X_test=X_test)
        
        metrics = converter.get_metrics()
        
        # Should maintain high fidelity
        assert metrics['fidelity'] >= 0.90
    
    def test_code_size_reasonable(self, iris_data):
        """Test that generated code size is reasonable."""
        X_train, X_test, y_train, y_test, _, _ = iris_data
        
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        converter = Converter(ConversionConfig(max_depth=5))
        c_code = converter.convert(model, X_train, X_test=X_test)
        
        metrics = converter.get_metrics()
        
        # Code should be compact
        assert metrics['size_estimate']['flash_bytes'] < 1000
        assert metrics['size_estimate']['ram_bytes'] < 100
    
    def test_optimization_levels(self, iris_data):
        """Test different optimization levels."""
        X_train, X_test, y_train, y_test, _, _ = iris_data
        
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        results = {}
        
        for opt_level in ['low', 'medium', 'high']:
            config = ConversionConfig(max_depth=5, optimize_rules=opt_level)
            converter = Converter(config)
            c_code = converter.convert(model, X_train, X_test=X_test)
            
            metrics = converter.get_metrics()
            results[opt_level] = metrics
        
        # All should maintain reasonable fidelity
        for opt_level, metrics in results.items():
            assert metrics['fidelity'] >= 0.85, f"{opt_level} optimization has low fidelity"
        
        # High optimization should have fewer or equal nodes
        assert (results['high']['complexity']['n_nodes'] <= 
                results['low']['complexity']['n_nodes'])
    
    def test_reproducibility(self, iris_data):
        """Test that conversion is reproducible with same random_state."""
        X_train, X_test, y_train, y_test, _, _ = iris_data
        
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        config = ConversionConfig(max_depth=5, random_state=42)
        
        # Convert twice
        converter1 = Converter(config)
        c_code1 = converter1.convert(model, X_train, X_test=X_test)
        metrics1 = converter1.get_metrics()
        
        converter2 = Converter(config)
        c_code2 = converter2.convert(model, X_train, X_test=X_test)
        metrics2 = converter2.get_metrics()
        
        # Should produce identical results
        assert c_code1 == c_code2
        assert metrics1['fidelity'] == metrics2['fidelity']
        assert metrics1['complexity']['n_nodes'] == metrics2['complexity']['n_nodes']
    
    def test_minimal_config(self, iris_data):
        """Test minimal configuration for resource-constrained devices."""
        X_train, X_test, y_train, y_test, _, _ = iris_data
        
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        config = ConversionConfig(
            max_depth=3,
            precision=8,
            optimize_rules='high',
            use_fixed_point=True,
            memory_budget_kb=0.5
        )
        
        converter = Converter(config)
        c_code = converter.convert(model, X_train, X_test=X_test)
        
        metrics = converter.get_metrics()
        
        # Should produce very compact code
        assert metrics['size_estimate']['flash_bytes'] < 300
        assert 'int8_t features[' in c_code  # Fixed point
