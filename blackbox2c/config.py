"""
Configuration module for BlackBox2C conversion parameters.
"""

import warnings
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ConversionConfig:
    """
    Configuration for model-to-C conversion.
    
    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the surrogate decision tree.
        Controls code complexity. Range: 1-10.
        
    precision : int, default=8
        Bits of precision for quantization (8, 16, or 32).
        8-bit is optimal for most MCUs; 16-bit balances range and size;
        32-bit matches standard int32_t width. Affects model size and accuracy.
        
    optimize_rules : Literal['low', 'medium', 'high'], default='medium'
        Level of rule optimization to reduce redundancies.
        
    feature_threshold : Optional[int], default=None
        Maximum number of features to use after automatic selection.
        When set, performs feature sensitivity analysis and keeps only
        the most important features up to this limit.
        None means use all features without selection.
        
    memory_budget_kb : Optional[float], default=None
        Target memory budget in KB. Auto-adjusts other parameters.
        
    use_fixed_point : bool, default=False
        Use fixed-point arithmetic instead of floating-point.
        
    function_name : str, default='predict'
        Name of the generated C function.
        
    include_probabilities : bool, default=False
        Generate probability estimates in addition to class prediction.
        When True, generates an additional function that returns class
        probabilities. Increases code size by ~30%.
        Note: Currently only supported for classification tasks.
        .. deprecated:: 0.1.0
            Not yet implemented. Will raise a warning if set to True.
        
    n_samples : int, default=10000
        Number of samples for boundary analysis (for non-tree models).
        
    random_state : Optional[int], default=42
        Random seed for reproducibility.
    """
    
    max_depth: int = 5
    precision: int = 8
    optimize_rules: Literal['low', 'medium', 'high'] = 'medium'
    feature_threshold: Optional[int] = None
    memory_budget_kb: Optional[float] = None
    use_fixed_point: bool = False
    function_name: str = 'predict'
    include_probabilities: bool = False
    n_samples: int = 10000
    random_state: Optional[int] = 42
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 1 <= self.max_depth <= 10:
            raise ValueError("max_depth must be between 1 and 10")
        
        if self.precision not in [8, 16, 32]:
            raise ValueError("precision must be 8, 16, or 32")
        
        if self.optimize_rules not in ['low', 'medium', 'high']:
            raise ValueError("optimize_rules must be 'low', 'medium', or 'high'")
        
        if self.feature_threshold is not None and self.feature_threshold < 1:
            raise ValueError("feature_threshold must be at least 1")
        
        if self.memory_budget_kb is not None and self.memory_budget_kb <= 0:
            raise ValueError("memory_budget_kb must be positive")
        
        if self.include_probabilities:
            warnings.warn(
                "include_probabilities=True is not yet implemented and will be "
                "ignored. This feature is planned for a future release.",
                UserWarning,
                stacklevel=2,
            )
        
        # Auto-adjust parameters based on memory budget
        if self.memory_budget_kb is not None:
            self._adjust_for_memory_budget()
    
    def _adjust_for_memory_budget(self):
        """Automatically adjust parameters based on memory budget."""
        budget = self.memory_budget_kb
        
        if budget < 1.0:
            self.max_depth = min(self.max_depth, 3)
            self.precision = 8
            self.use_fixed_point = True
        elif budget < 2.0:
            self.max_depth = min(self.max_depth, 4)
            self.precision = 8
        elif budget < 4.0:
            self.max_depth = min(self.max_depth, 6)
