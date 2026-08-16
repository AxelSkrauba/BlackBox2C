"""
Tests for configuration module.
"""

import pytest
from blackbox2c.config import ConversionConfig


class TestConversionConfig:
    """Test ConversionConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ConversionConfig()
        
        assert config.max_depth == 5
        assert config.precision == 8
        assert config.optimize_rules == 'medium'
        assert config.use_fixed_point is False
        assert config.function_name == 'predict'
        assert config.n_samples == 10000
        assert config.random_state == 42
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ConversionConfig(
            max_depth=7,
            precision=16,
            optimize_rules='high',
            use_fixed_point=True
        )
        
        assert config.max_depth == 7
        assert config.precision == 16
        assert config.optimize_rules == 'high'
        assert config.use_fixed_point is True
    
    def test_invalid_max_depth(self):
        """Test validation of max_depth parameter."""
        with pytest.raises(ValueError, match="max_depth must be between 1 and 10"):
            ConversionConfig(max_depth=0)
        
        with pytest.raises(ValueError, match="max_depth must be between 1 and 10"):
            ConversionConfig(max_depth=11)
    
    def test_invalid_precision(self):
        """Test validation of precision parameter."""
        with pytest.raises(ValueError, match="precision must be"):
            ConversionConfig(precision=7)
        
        with pytest.raises(ValueError, match="precision must be"):
            ConversionConfig(precision=64)
    
    def test_invalid_optimize_rules(self):
        """Test validation of optimize_rules parameter."""
        with pytest.raises(ValueError, match="optimize_rules must be"):
            ConversionConfig(optimize_rules='invalid')
    
    def test_memory_budget_adjustment(self):
        """Test automatic parameter adjustment based on memory budget."""
        # Very low budget
        config = ConversionConfig(memory_budget_kb=0.5)
        assert config.max_depth <= 3
        assert config.precision == 8
        assert config.use_fixed_point is True
        
        # Low budget
        config = ConversionConfig(memory_budget_kb=1.5)
        assert config.max_depth <= 4
        assert config.precision == 8
        
        # Medium budget
        config = ConversionConfig(memory_budget_kb=3.0)
        assert config.max_depth <= 6
    
    def test_fidelity_warning_threshold(self):
        """Test fidelity warning threshold configuration."""
        assert ConversionConfig().fidelity_warning_threshold == 0.95
        assert ConversionConfig(fidelity_warning_threshold=None).fidelity_warning_threshold is None
        assert ConversionConfig(fidelity_warning_threshold=-0.5).fidelity_warning_threshold == -0.5

        with pytest.raises(ValueError, match="fidelity_warning_threshold"):
            ConversionConfig(fidelity_warning_threshold=1.1)

    def test_bridge_node_limit(self):
        """Test validation of max_bridge_nodes."""
        assert ConversionConfig().max_bridge_nodes == 4096
        assert ConversionConfig(max_bridge_nodes=12).max_bridge_nodes == 12

        with pytest.raises(ValueError, match="max_bridge_nodes"):
            ConversionConfig(max_bridge_nodes=0)

    def test_invalid_memory_budget(self):
        """Test validation of memory_budget_kb parameter."""
        with pytest.raises(ValueError, match="memory_budget_kb must be positive"):
            ConversionConfig(memory_budget_kb=-1.0)
        
        with pytest.raises(ValueError, match="memory_budget_kb must be positive"):
            ConversionConfig(memory_budget_kb=0.0)
