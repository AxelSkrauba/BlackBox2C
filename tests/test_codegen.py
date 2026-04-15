"""
Tests for C code generation module.
"""

import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

from blackbox2c.codegen import CCodeGenerator


class TestCCodeGenerator:
    """Test CCodeGenerator class."""
    
    @pytest.fixture
    def sample_tree(self):
        """Create a sample decision tree."""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            n_repeated=0,
            n_classes=3,
            random_state=42
        )
        
        tree = DecisionTreeClassifier(max_depth=4, random_state=42)
        tree.fit(X, y)
        return tree
    
    def test_initialization(self):
        """Test CCodeGenerator initialization."""
        gen = CCodeGenerator(
            function_name='my_predict',
            use_fixed_point=True,
            precision=16
        )
        
        assert gen.function_name == 'my_predict'
        assert gen.use_fixed_point is True
        assert gen.precision == 16
    
    def test_generate_basic_code(self, sample_tree):
        """Test basic C code generation."""
        gen = CCodeGenerator()
        feature_names = [f"features[{i}]" for i in range(4)]
        class_names = ['CLASS_0', 'CLASS_1', 'CLASS_2']
        
        c_code = gen.generate(sample_tree, feature_names, class_names)
        
        assert isinstance(c_code, str)
        assert len(c_code) > 0
        
        # Check for essential components
        assert '#include <stdint.h>' in c_code
        assert 'uint8_t predict(' in c_code
        assert 'return' in c_code
        assert 'if' in c_code
        
        # Check for class definitions
        assert '#define CLASS_0 0' in c_code
        assert '#define CLASS_1 1' in c_code
        assert '#define CLASS_2 2' in c_code
    
    def test_generate_with_custom_function_name(self, sample_tree):
        """Test code generation with custom function name."""
        gen = CCodeGenerator(function_name='classify_sample')
        feature_names = [f"features[{i}]" for i in range(4)]
        
        c_code = gen.generate(sample_tree, feature_names)
        
        assert 'uint8_t classify_sample(' in c_code
    
    def test_generate_with_fixed_point(self, sample_tree):
        """Test code generation with fixed-point arithmetic."""
        gen = CCodeGenerator(use_fixed_point=True, precision=8)
        feature_names = [f"features[{i}]" for i in range(4)]
        
        c_code = gen.generate(sample_tree, feature_names)
        
        assert 'int8_t features[' in c_code
    
    def test_generate_with_16bit_fixed_point(self, sample_tree):
        """Test code generation with 16-bit fixed-point."""
        gen = CCodeGenerator(use_fixed_point=True, precision=16)
        feature_names = [f"features[{i}]" for i in range(4)]
        
        c_code = gen.generate(sample_tree, feature_names)
        
        assert 'int16_t features[' in c_code
    
    def test_generate_with_float(self, sample_tree):
        """Test code generation with floating-point."""
        gen = CCodeGenerator(use_fixed_point=False)
        feature_names = [f"features[{i}]" for i in range(4)]
        
        c_code = gen.generate(sample_tree, feature_names)
        
        assert 'float features[' in c_code
        assert '.0f' in c_code or 'f' in c_code  # Float literals
    
    def test_code_size_estimation(self, sample_tree):
        """Test code size estimation."""
        gen = CCodeGenerator()
        size_estimate = gen.estimate_code_size(sample_tree)
        
        assert 'flash_bytes' in size_estimate
        assert 'ram_bytes' in size_estimate
        assert 'n_nodes' in size_estimate
        assert 'n_conditions' in size_estimate
        assert 'n_leaves' in size_estimate
        
        # Sanity checks
        assert size_estimate['flash_bytes'] > 0
        assert size_estimate['ram_bytes'] > 0
        assert size_estimate['n_nodes'] > 0
        assert size_estimate['n_leaves'] > 0
        assert size_estimate['n_conditions'] >= 0
    
    def test_header_generation(self, sample_tree):
        """Test header comment generation."""
        # Test classification header
        gen_class = CCodeGenerator(precision=16, task_type='classification')
        header_class = gen_class._generate_header(4, 3)
        
        assert '* Auto-generated C code by BlackBox2C' in header_class
        assert '*   - Input features: 4' in header_class
        assert '*   - Output classes: 3' in header_class
        assert '*   - Precision: 16-bit' in header_class
        assert '#include <stdint.h>' in header_class
        
        # Test regression header
        gen_reg = CCodeGenerator(precision=16, task_type='regression')
        header_reg = gen_reg._generate_header(4, None)
        
        assert '* Auto-generated C code by BlackBox2C' in header_reg
        assert '*   - Input features: 4' in header_reg
        assert '*   - Task: Regression' in header_reg
        assert '*   - Precision: 16-bit' in header_reg
    
    def test_defines_generation(self):
        """Test class label defines generation."""
        gen = CCodeGenerator()
        class_names = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']
        defines = gen._generate_defines(class_names)
        
        assert '#define SETOSA 0' in defines
        assert '#define VERSICOLOR 1' in defines
        assert '#define VIRGINICA 2' in defines
    
    def test_usage_comment_generation(self):
        """Test usage comment generation."""
        gen = CCodeGenerator(function_name='my_predict')
        feature_names = ['feat_a', 'feat_b']
        class_names = ['CLASS_A', 'CLASS_B']
        
        comment = gen._generate_usage_comment(feature_names, class_names)
        
        assert 'Usage Example' in comment
        assert 'my_predict' in comment
        assert 'feat_a' in comment
        assert 'CLASS_A' in comment
