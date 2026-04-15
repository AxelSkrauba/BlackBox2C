"""
Main conversion module - orchestrates the entire conversion process.
"""

import numpy as np
from sklearn.base import BaseEstimator, is_regressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from typing import Optional, Union, Dict, Any

from .config import ConversionConfig
from .surrogate import SurrogateExtractor
from .codegen import CCodeGenerator
from .optimizer import RuleOptimizer
from .exporters import create_exporter


class _FeatureSubsetWrapper:
    """
    Wraps a model so it accepts a reduced feature set by expanding back to the
    original feature count (padding unseen features with column-wise means).
    Used internally when feature_threshold slices X_train.
    """

    def __init__(self, model, selected_indices: list, X_train_orig: np.ndarray):
        self._model = model
        self._indices = selected_indices
        self._n_features_orig = X_train_orig.shape[1]
        self._col_means = X_train_orig.mean(axis=0)
        # Mirror the wrapped model's estimator type for attribute inspection
        self._estimator_type = 'regressor' if is_regressor(model) else 'classifier'
        if hasattr(model, 'classes_'):
            self.classes_ = model.classes_

    def predict(self, X_reduced: np.ndarray) -> np.ndarray:
        X_full = np.tile(self._col_means, (len(X_reduced), 1))
        X_full[:, self._indices] = X_reduced
        return self._model.predict(X_full)

    def score(self, X_reduced: np.ndarray, y) -> float:
        X_full = np.tile(self._col_means, (len(X_reduced), 1))
        X_full[:, self._indices] = X_reduced
        return self._model.score(X_full, y)


class Converter:
    """
    Main converter class that orchestrates the model-to-C conversion.
    """
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        """
        Initialize the converter.
        
        Parameters
        ----------
        config : ConversionConfig, optional
            Configuration for the conversion. If None, uses defaults.
        """
        self.config = config or ConversionConfig()
        self.surrogate_tree_ = None
        self.feature_names_ = None
        self.class_names_ = None
        self.metrics_ = {}
    
    def convert(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        feature_names: Optional[list] = None,
        class_names: Optional[list] = None,
        X_test: Optional[np.ndarray] = None,
        target: str = 'c',
    ) -> str:
        """
        Convert a trained model to embedded code.
        
        Parameters
        ----------
        model : BaseEstimator
            The trained scikit-learn model to convert.
        X_train : np.ndarray
            Training data used to understand feature ranges.
        feature_names : list, optional
            Names of input features.
        class_names : list, optional
            Names of output classes.
        X_test : np.ndarray, optional
            Test data for fidelity evaluation.
        target : str, default='c'
            Output format. One of 'c', 'cpp', 'arduino', 'micropython'.
            'c' uses the built-in C code generator; the others use
            the corresponding platform-specific exporter.
        
        Returns
        -------
        str
            Generated code in the requested target format.
        """
        # Validate inputs
        X_train = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
        if X_test is not None:
            X_test = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
        self._validate_inputs(model, X_train, X_test, feature_names, class_names)
        
        # Feature selection if threshold specified
        selected_features = None
        X_train_orig = X_train  # keep original for model labeling
        if self.config.feature_threshold is not None:
            print(f"\n[Feature Selection] Analyzing {X_train.shape[1]} features...")
            
            from .analysis import FeatureSensitivityAnalyzer
            
            # Analyze features using the original (full) X_train
            y_train = model.predict(X_train)
            analyzer = FeatureSensitivityAnalyzer(
                n_repeats=5,
                random_state=self.config.random_state
            )
            temp_feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
            results = analyzer.analyze(model, X_train, y_train, feature_names=temp_feature_names)
            
            # Select optimal subset
            selected_features = results.get_optimal_subset(
                threshold=0.01,
                min_features=min(self.config.feature_threshold, X_train.shape[1])
            )
            
            print(f"  Selected {len(selected_features)}/{X_train.shape[1]} features: {selected_features}")
            removed = [i for i in range(X_train.shape[1]) if i not in selected_features]
            if removed:
                removed_names = [temp_feature_names[i] for i in removed]
                print(f"  Removed features: {removed_names}")
            
            # Reduce datasets for surrogate training only
            # NOTE: X_train_orig is kept so the original model can label synthetic data.
            # The surrogate tree will be trained on the reduced feature set.
            # This requires the surrogate extractor to label using the original model
            # on full X; we achieve this by passing a feature-mapping wrapper.
            X_train = X_train[:, selected_features]
            if X_test is not None:
                X_test = X_test[:, selected_features]
            if feature_names:
                feature_names = [feature_names[i] for i in selected_features]
        
        # Store feature and class names
        n_features = X_train.shape[1]
        self.feature_names_ = feature_names or [f"features[{i}]" for i in range(n_features)]
        
        # Detect task type (classification vs regression)
        # Use sklearn's is_regressor() for proper detection; fall back to
        # _estimator_type attribute inspection for non-conformant mock models.
        try:
            is_regression = is_regressor(model)
        except AttributeError:
            is_regression = getattr(model, '_estimator_type', 'classifier') == 'regressor'
        
        if is_regression:
            # Regression task
            n_classes = None
            self.class_names_ = None
            print(f"\nStarting conversion for model: {type(model).__name__}")
            print(f"  Task: Regression, Features: {n_features}, Max depth: {self.config.max_depth}")
        else:
            # Classification task
            if hasattr(model, 'classes_'):
                n_classes = len(model.classes_)
            else:
                y_sample = model.predict(X_train[:min(100, len(X_train))])
                n_classes = len(np.unique(y_sample))
            
            self.class_names_ = class_names or [f"CLASS_{i}" for i in range(n_classes)]
            print(f"\nStarting conversion for model: {type(model).__name__}")
            print(f"  Task: Classification, Features: {n_features}, Classes: {n_classes}, Max depth: {self.config.max_depth}")
        
        # Extract surrogate tree
        print("\n[1/4] Extracting surrogate decision tree...")
        surrogate_extractor = SurrogateExtractor(
            max_depth=self.config.max_depth,
            n_samples=self.config.n_samples,
            random_state=self.config.random_state
        )
        
        # When feature selection is active, wrap the original model so that the
        # surrogate extractor can label synthetic data (with reduced features)
        # by mapping them back to the full feature space.
        model_for_surrogate = model
        if selected_features is not None and not isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
            model_for_surrogate = _FeatureSubsetWrapper(model, selected_features, X_train_orig)
        
        if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
            print("  Model is already a decision tree, using directly.")
            self.surrogate_tree_ = model
            fidelity = 1.0
        else:
            task_type = 'regression' if is_regression else 'classification'
            self.surrogate_tree_ = surrogate_extractor.extract(
                model_for_surrogate, X_train, self.feature_names_,
                task_type=task_type
            )
            fidelity = surrogate_extractor.get_fidelity(
                model_for_surrogate, X_test if X_test is not None else X_train
            )
            print(f"  Surrogate fidelity: {fidelity:.4f}")
        
        self.metrics_['fidelity'] = fidelity
        
        # Optimize rules
        print("\n[2/4] Optimizing decision rules...")
        optimizer = RuleOptimizer(optimization_level=self.config.optimize_rules)
        self.surrogate_tree_ = optimizer.optimize(self.surrogate_tree_)
        
        complexity = optimizer.analyze_complexity(self.surrogate_tree_)
        print(f"  Nodes: {complexity['n_nodes']}, Leaves: {complexity['n_leaves']}, Depth: {complexity['max_depth']}")
        self.metrics_['complexity'] = complexity
        
        # Generate C code
        print("\n[3/4] Generating C code...")
        code_generator = CCodeGenerator(
            function_name=self.config.function_name,
            use_fixed_point=self.config.use_fixed_point,
            precision=self.config.precision,
            optimize_rules=self.config.optimize_rules
        )
        
        c_code = code_generator.generate(self.surrogate_tree_, self.feature_names_, self.class_names_)
        
        # Estimate code size
        print("\n[4/4] Estimating code size...")
        size_estimate = code_generator.estimate_code_size(self.surrogate_tree_)
        print(f"  Estimated FLASH: {size_estimate['flash_bytes']} bytes, RAM: {size_estimate['ram_bytes']} bytes")
        self.metrics_['size_estimate'] = size_estimate
        
        print("\n[OK] Conversion complete!")
        
        target_lower = target.lower()
        if target_lower == 'c':
            return c_code
        
        # Use platform-specific exporter for non-C targets
        print(f"\n[Export] Generating {target} code...")
        # Build only the kwargs each exporter accepts
        exporter_kwargs = {"function_name": self.config.function_name}
        if target_lower in ('cpp', 'c++'):
            exporter_kwargs["use_fixed_point"] = self.config.use_fixed_point
            exporter_kwargs["precision"] = self.config.precision
        elif target_lower == 'arduino':
            exporter_kwargs["use_fixed_point"] = self.config.use_fixed_point
            exporter_kwargs["precision"] = self.config.precision
        # MicroPython exporter does not accept use_fixed_point/precision
        exporter = create_exporter(target_lower, **exporter_kwargs)
        return exporter.generate(
            self.surrogate_tree_,
            feature_names=self.feature_names_,
            class_names=self.class_names_,
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get conversion metrics."""
        return self.metrics_
    
    def _validate_inputs(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        X_test: Optional[np.ndarray],
        feature_names: Optional[list],
        class_names: Optional[list]
    ):
        """
        Validate all inputs before conversion.
        
        Raises
        ------
        ValueError
            If any input is invalid.
        TypeError
            If types are incorrect.
        """
        # Validate model
        if not hasattr(model, 'predict'):
            raise TypeError(
                f"Model must have a predict() method. Got type: {type(model).__name__}"
            )
        
        # X_train already converted to np.ndarray before this call
        if not isinstance(X_train, np.ndarray):
            raise TypeError("X_train must be a numpy array")
        
        if X_train.ndim != 2:
            raise ValueError(
                f"X_train must be 2-dimensional (samples, features). "
                f"Got shape: {X_train.shape}"
            )
        
        if X_train.shape[0] < 5:
            raise ValueError(
                f"X_train must have at least 5 samples for meaningful conversion. "
                f"Got: {X_train.shape[0]} samples"
            )
        
        if X_train.shape[1] < 1:
            raise ValueError("X_train must have at least 1 feature")
        
        if X_train.shape[1] > 100:
            raise ValueError(
                f"Too many features ({X_train.shape[1]}). "
                f"Consider feature selection or set feature_threshold"
            )
        
        # Check for NaN or Inf
        if not np.isfinite(X_train).all():
            raise ValueError("X_train contains NaN or Inf values")
        
        # Validate X_test if provided
        if X_test is not None:
            if not isinstance(X_test, np.ndarray):
                raise TypeError("X_test must be a numpy array")
            
            if X_test.ndim != 2:
                raise ValueError(
                    f"X_test must be 2-dimensional. Got shape: {X_test.shape}"
                )
            
            if X_test.shape[1] != X_train.shape[1]:
                raise ValueError(
                    f"X_test must have same number of features as X_train. "
                    f"X_train: {X_train.shape[1]}, X_test: {X_test.shape[1]}"
                )
            
            if not np.isfinite(X_test).all():
                raise ValueError("X_test contains NaN or Inf values")
        
        # Validate feature_names if provided
        if feature_names is not None:
            if not isinstance(feature_names, (list, tuple)):
                raise TypeError("feature_names must be a list or tuple")
            
            if len(feature_names) != X_train.shape[1]:
                raise ValueError(
                    f"Length of feature_names ({len(feature_names)}) must match "
                    f"number of features in X_train ({X_train.shape[1]})"
                )
        
        # Validate class_names if provided
        if class_names is not None:
            if not isinstance(class_names, (list, tuple)):
                raise TypeError("class_names must be a list or tuple")
            
            if len(class_names) < 2:
                raise ValueError("class_names must have at least 2 classes")
