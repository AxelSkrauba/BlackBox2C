"""
Tests for multi-format exporters.

This module tests the code generation for different target platforms:
- C++ (modern C++11)
- Arduino (optimized for embedded)
- MicroPython (for microcontrollers)
"""

import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, make_regression

from blackbox2c.exporters import (
    BaseExporter,
    CppExporter,
    ArduinoExporter,
    MicroPythonExporter,
    create_exporter
)


class TestCppExporter:
    """Tests for C++ code exporter."""
    
    def test_initialization(self):
        """Test CppExporter initialization."""
        exporter = CppExporter()
        assert exporter.function_name == "predict"
        assert exporter.class_name == "Predictor"
        assert exporter.use_namespace is True
        assert exporter.namespace == "ml"
        assert exporter.use_fixed_point is False
    
    def test_custom_initialization(self):
        """Test CppExporter with custom parameters."""
        exporter = CppExporter(
            function_name="classify",
            class_name="MyClassifier",
            use_namespace=False,
            namespace="custom"
        )
        assert exporter.function_name == "classify"
        assert exporter.class_name == "MyClassifier"
        assert exporter.use_namespace is False
        assert exporter.namespace == "custom"
    
    def test_generate_classification(self):
        """Test C++ code generation for classification."""
        # Train simple model
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(iris.data, iris.target)
        
        # Generate C++ code
        exporter = CppExporter()
        code = exporter.generate(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names.tolist()
        )
        
        # Verify code structure
        assert "namespace ml {" in code
        assert "class Predictor {" in code
        assert "static uint8_t predict" in code
        assert "FEATURE_NAMES" in code
        assert "CLASS_NAMES" in code
        assert "get_class_name" in code
        assert "#include <cstdint>" in code
        assert "#include <array>" in code
    
    def test_generate_regression(self):
        """Test C++ code generation for regression."""
        # Generate regression data
        X, y = make_regression(n_samples=100, n_features=3, random_state=42)
        model = DecisionTreeRegressor(max_depth=3, random_state=42)
        model.fit(X, y)
        
        # Generate C++ code
        exporter = CppExporter()
        code = exporter.generate(
            model,
            feature_names=['f0', 'f1', 'f2']
        )
        
        # Verify code structure
        assert "namespace ml {" in code
        assert "class Predictor {" in code
        assert "static float predict" in code  # float for regression
        assert "FEATURE_NAMES" in code
        assert "CLASS_NAMES" not in code  # No class names for regression
        assert "get_class_name" not in code
    
    def test_without_namespace(self):
        """Test C++ code generation without namespace."""
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(iris.data, iris.target)
        
        exporter = CppExporter(use_namespace=False)
        code = exporter.generate(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names.tolist()
        )
        
        assert "namespace ml {" not in code
        assert "class Predictor {" in code
    
    def test_custom_class_name(self):
        """Test C++ code with custom class name."""
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(iris.data, iris.target)
        
        exporter = CppExporter(class_name="IrisClassifier")
        code = exporter.generate(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names.tolist()
        )
        
        assert "class IrisClassifier {" in code
        assert "class Predictor {" not in code
    
    def test_code_compiles_structure(self):
        """Test that generated C++ code has valid structure."""
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(iris.data, iris.target)
        
        exporter = CppExporter()
        code = exporter.generate(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names.tolist()
        )
        
        # Check for proper C++ syntax
        assert code.count("namespace ml {") == code.count("} // namespace ml")
        assert "std::array<float," in code
        assert "constexpr" in code
        assert "static" in code


class TestArduinoExporter:
    """Tests for Arduino code exporter."""
    
    def test_initialization(self):
        """Test ArduinoExporter initialization."""
        exporter = ArduinoExporter()
        assert exporter.function_name == "predict"
        assert exporter.use_progmem is True
        assert exporter.use_fixed_point is False
    
    def test_custom_initialization(self):
        """Test ArduinoExporter with custom parameters."""
        exporter = ArduinoExporter(
            function_name="classify_sensor",
            use_progmem=False
        )
        assert exporter.function_name == "classify_sensor"
        assert exporter.use_progmem is False
    
    def test_generate_classification(self):
        """Test Arduino code generation for classification."""
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(iris.data, iris.target)
        
        exporter = ArduinoExporter()
        code = exporter.generate(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names.tolist()
        )
        
        # Verify Arduino-specific code
        assert "#include <Arduino.h>" in code
        assert "PROGMEM" in code
        assert "FEATURE_NAMES" in code
        assert "CLASS_NAMES" in code
        assert "uint8_t predict" in code
        assert "get_class_name" in code
    
    def test_generate_regression(self):
        """Test Arduino code generation for regression."""
        X, y = make_regression(n_samples=100, n_features=2, random_state=42)
        model = DecisionTreeRegressor(max_depth=3, random_state=42)
        model.fit(X, y)
        
        exporter = ArduinoExporter()
        code = exporter.generate(
            model,
            feature_names=['sensor1', 'sensor2']
        )
        
        # Verify code structure
        assert "#include <Arduino.h>" in code
        assert "float predict" in code  # float for regression
        assert "FEATURE_NAMES" in code
        assert "CLASS_NAMES" not in code
    
    def test_without_progmem(self):
        """Test Arduino code without PROGMEM."""
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(iris.data, iris.target)
        
        exporter = ArduinoExporter(use_progmem=False)
        code = exporter.generate(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names.tolist()
        )
        
        # Should not have PROGMEM in feature/class names
        assert "const char* const FEATURE_NAMES[]" in code
        assert "PROGMEM FEATURE_NAMES" not in code
    
    def test_custom_function_name(self):
        """Test Arduino code with custom function name."""
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(iris.data, iris.target)
        
        exporter = ArduinoExporter(function_name="predict_iris")
        code = exporter.generate(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names.tolist()
        )
        
        assert "uint8_t predict_iris(" in code
        assert "uint8_t predict(" not in code
    
    def test_usage_example_included(self):
        """Test that Arduino code includes usage example."""
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(iris.data, iris.target)
        
        exporter = ArduinoExporter()
        code = exporter.generate(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names.tolist()
        )
        
        # Check for Arduino sketch example
        assert "void setup()" in code or "Arduino Sketch Example" in code
        assert "void loop()" in code or "loop()" in code


class TestMicroPythonExporter:
    """Tests for MicroPython code exporter."""
    
    def test_initialization(self):
        """Test MicroPythonExporter initialization."""
        exporter = MicroPythonExporter()
        assert exporter.function_name == "predict"
        assert exporter.class_name == "Predictor"
        assert exporter.use_const is True
    
    def test_custom_initialization(self):
        """Test MicroPythonExporter with custom parameters."""
        exporter = MicroPythonExporter(
            function_name="classify",
            class_name="MyPredictor",
            use_const=False
        )
        assert exporter.function_name == "classify"
        assert exporter.class_name == "MyPredictor"
        assert exporter.use_const is False
    
    def test_generate_classification(self):
        """Test MicroPython code generation for classification."""
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(iris.data, iris.target)
        
        exporter = MicroPythonExporter()
        code = exporter.generate(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names.tolist()
        )
        
        # Verify Python code structure
        assert "from micropython import const" in code
        assert "class Predictor:" in code
        assert "def predict(features):" in code
        assert "FEATURE_NAMES" in code
        assert "CLASS_NAMES" in code
        assert "get_class_name" in code
        assert "@staticmethod" in code
    
    def test_generate_regression(self):
        """Test MicroPython code generation for regression."""
        X, y = make_regression(n_samples=100, n_features=3, random_state=42)
        model = DecisionTreeRegressor(max_depth=3, random_state=42)
        model.fit(X, y)
        
        exporter = MicroPythonExporter()
        code = exporter.generate(
            model,
            feature_names=['temp', 'humidity', 'pressure']
        )
        
        # Verify code structure
        assert "class Predictor:" in code
        assert "def predict(features):" in code
        assert "FEATURE_NAMES" in code
        assert "CLASS_NAMES" not in code
        assert "get_class_name" not in code
    
    def test_without_const(self):
        """Test MicroPython code without const optimization."""
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(iris.data, iris.target)
        
        exporter = MicroPythonExporter(use_const=False)
        code = exporter.generate(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names.tolist()
        )
        
        assert "from micropython import const" not in code
        assert "class Predictor:" in code
    
    def test_custom_class_name(self):
        """Test MicroPython code with custom class name."""
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(iris.data, iris.target)
        
        exporter = MicroPythonExporter(class_name="IrisClassifier")
        code = exporter.generate(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names.tolist()
        )
        
        assert "class IrisClassifier:" in code
        assert "class Predictor:" not in code
    
    def test_valid_python_syntax(self):
        """Test that generated code has valid Python syntax."""
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(iris.data, iris.target)
        
        exporter = MicroPythonExporter()
        code = exporter.generate(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names.tolist()
        )
        
        # Try to compile the code (syntax check)
        try:
            compile(code, '<string>', 'exec')
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False
        
        assert syntax_valid, "Generated Python code has syntax errors"
    
    def test_feature_validation(self):
        """Test that generated code includes feature validation."""
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(iris.data, iris.target)
        
        exporter = MicroPythonExporter()
        code = exporter.generate(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names.tolist()
        )
        
        # Check for feature length validation
        assert "len(features)" in code
        assert "ValueError" in code or "raise" in code


class TestFactoryPattern:
    """Tests for the exporter factory pattern."""
    
    def test_create_cpp_exporter(self):
        """Test creating C++ exporter via factory."""
        exporter = create_exporter('cpp')
        assert isinstance(exporter, CppExporter)
    
    def test_create_cpp_exporter_alternate(self):
        """Test creating C++ exporter with alternate name."""
        exporter = create_exporter('c++')
        assert isinstance(exporter, CppExporter)
    
    def test_create_arduino_exporter(self):
        """Test creating Arduino exporter via factory."""
        exporter = create_exporter('arduino')
        assert isinstance(exporter, ArduinoExporter)
    
    def test_create_micropython_exporter(self):
        """Test creating MicroPython exporter via factory."""
        exporter = create_exporter('micropython')
        assert isinstance(exporter, MicroPythonExporter)
    
    def test_create_micropython_exporter_alternate(self):
        """Test creating MicroPython exporter with alternate name."""
        exporter = create_exporter('python')
        assert isinstance(exporter, MicroPythonExporter)
    
    def test_factory_with_kwargs(self):
        """Test factory pattern with custom parameters."""
        exporter = create_exporter('cpp', class_name='CustomClass')
        assert isinstance(exporter, CppExporter)
        assert exporter.class_name == 'CustomClass'
    
    def test_invalid_format(self):
        """Test factory with invalid format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            create_exporter('invalid_format')
    
    def test_case_insensitive(self):
        """Test that factory is case-insensitive."""
        exporter1 = create_exporter('CPP')
        exporter2 = create_exporter('Arduino')
        exporter3 = create_exporter('MicroPython')
        
        assert isinstance(exporter1, CppExporter)
        assert isinstance(exporter2, ArduinoExporter)
        assert isinstance(exporter3, MicroPythonExporter)


class TestExporterIntegration:
    """Integration tests for exporters."""
    
    def test_all_formats_classification(self):
        """Test that all formats work for classification."""
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(iris.data, iris.target)
        
        formats = ['cpp', 'arduino', 'micropython']
        
        for fmt in formats:
            exporter = create_exporter(fmt)
            code = exporter.generate(
                model,
                feature_names=iris.feature_names,
                class_names=iris.target_names.tolist()
            )
            
            assert len(code) > 0, f"Empty code for format: {fmt}"
            assert "predict" in code.lower(), f"No predict function in {fmt}"
    
    def test_all_formats_regression(self):
        """Test that all formats work for regression."""
        X, y = make_regression(n_samples=100, n_features=3, random_state=42)
        model = DecisionTreeRegressor(max_depth=3, random_state=42)
        model.fit(X, y)
        
        formats = ['cpp', 'arduino', 'micropython']
        
        for fmt in formats:
            exporter = create_exporter(fmt)
            code = exporter.generate(
                model,
                feature_names=['f0', 'f1', 'f2']
            )
            
            assert len(code) > 0, f"Empty code for format: {fmt}"
            assert "predict" in code.lower(), f"No predict function in {fmt}"
    
    def test_code_size_comparison(self):
        """Test and compare code sizes across formats."""
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(iris.data, iris.target)
        
        sizes = {}
        for fmt in ['cpp', 'arduino', 'micropython']:
            exporter = create_exporter(fmt)
            code = exporter.generate(
                model,
                feature_names=iris.feature_names,
                class_names=iris.target_names.tolist()
            )
            sizes[fmt] = len(code)
        
        # All formats should generate reasonable code
        for fmt, size in sizes.items():
            assert 500 < size < 10000, f"Unusual code size for {fmt}: {size}"
    
    def test_feature_names_preserved(self):
        """Test that feature names are preserved in all formats."""
        feature_names = ['custom_f1', 'custom_f2', 'custom_f3']
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(X, y)
        
        for fmt in ['cpp', 'arduino', 'micropython']:
            exporter = create_exporter(fmt)
            code = exporter.generate(
                model,
                feature_names=feature_names,
                class_names=['class0', 'class1']
            )
            
            # Check that custom feature names appear in code
            for name in feature_names:
                assert name in code, f"Feature name {name} not in {fmt} code"
    
    def test_deep_tree_handling(self):
        """Test that exporters handle deeper trees correctly."""
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=7, random_state=42)
        model.fit(iris.data, iris.target)
        
        for fmt in ['cpp', 'arduino', 'micropython']:
            exporter = create_exporter(fmt)
            code = exporter.generate(
                model,
                feature_names=iris.feature_names,
                class_names=iris.target_names.tolist()
            )
            
            # Should generate valid code even for deep trees
            assert len(code) > 0
            assert "predict" in code.lower()


class TestExporterEdgeCases:
    """Test edge cases for exporters."""
    
    def test_single_feature(self):
        """Test exporters with single feature."""
        X = np.random.rand(50, 1)
        y = np.random.randint(0, 2, 50)
        
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)
        
        for fmt in ['cpp', 'arduino', 'micropython']:
            exporter = create_exporter(fmt)
            code = exporter.generate(
                model,
                feature_names=['single_feature'],
                class_names=['class0', 'class1']
            )
            
            assert len(code) > 0
            assert 'single_feature' in code
    
    def test_many_features(self):
        """Test exporters with many features."""
        X = np.random.rand(100, 20)
        y = np.random.randint(0, 3, 100)
        
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X, y)
        
        feature_names = [f'feature_{i}' for i in range(20)]
        
        for fmt in ['cpp', 'arduino', 'micropython']:
            exporter = create_exporter(fmt)
            code = exporter.generate(
                model,
                feature_names=feature_names,
                class_names=['c0', 'c1', 'c2']
            )
            
            assert len(code) > 0
    
    def test_special_characters_in_names(self):
        """Test handling of special characters in feature names."""
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(X, y)
        
        # Feature names with spaces and special chars
        feature_names = ['temp (C)', 'humidity %', 'pressure-hPa']
        
        for fmt in ['cpp', 'arduino', 'micropython']:
            exporter = create_exporter(fmt)
            code = exporter.generate(
                model,
                feature_names=feature_names,
                class_names=['low', 'high']
            )
            
            # Should generate code without errors
            assert len(code) > 0
    
    def test_binary_classification(self):
        """Test exporters with binary classification."""
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)
        
        for fmt in ['cpp', 'arduino', 'micropython']:
            exporter = create_exporter(fmt)
            code = exporter.generate(
                model,
                feature_names=['f1', 'f2', 'f3'],
                class_names=['negative', 'positive']
            )
            
            assert 'negative' in code
            assert 'positive' in code
    
    def test_multiclass_classification(self):
        """Test exporters with multiclass classification."""
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 5, 100)
        
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X, y)
        
        class_names = ['class_A', 'class_B', 'class_C', 'class_D', 'class_E']
        
        for fmt in ['cpp', 'arduino', 'micropython']:
            exporter = create_exporter(fmt)
            code = exporter.generate(
                model,
                feature_names=['f1', 'f2', 'f3', 'f4'],
                class_names=class_names
            )
            
            # All class names should appear
            for name in class_names:
                assert name in code


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
