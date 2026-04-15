"""
Surrogate model extraction for black-box models.

This module implements algorithms to extract decision rules from
arbitrary ML models by analyzing their decision boundaries.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import BaseEstimator, is_regressor
from typing import Optional, Tuple, Union


class SurrogateExtractor:
    """
    Extract a surrogate decision tree from any black-box model.
    
    The surrogate tree approximates the original model's decision
    boundaries (classification) or function (regression) using a 
    simpler, interpretable structure.
    
    Supports both classification and regression tasks.
    """
    
    def __init__(
        self,
        max_depth: int = 5,
        n_samples: int = 10000,
        random_state: Optional[int] = 42
    ):
        """
        Initialize the surrogate extractor.
        
        Parameters
        ----------
        max_depth : int
            Maximum depth of the surrogate tree.
        n_samples : int
            Number of samples to generate for boundary analysis.
        random_state : int or None
            Random seed for reproducibility.
        """
        self.max_depth = max_depth
        self.n_samples = n_samples
        self.random_state = random_state
        self.surrogate_tree_ = None
        self.feature_names_ = None
        self.n_features_ = None
        self.n_classes_ = None
    
    def extract(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        feature_names: Optional[list] = None,
        task_type: Optional[str] = None
    ) -> Union[DecisionTreeClassifier, DecisionTreeRegressor]:
        """
        Extract a surrogate decision tree from the black-box model.
        
        Automatically detects if the task is classification or regression
        based on the model type.
        
        Parameters
        ----------
        model : BaseEstimator
            The trained model to approximate.
        X_train : np.ndarray
            Training data used to understand feature ranges.
        feature_names : list, optional
            Names of features for better code generation.
        task_type : str, optional
            Override task detection: 'regression' or 'classification'.
            If None, auto-detected from model.
        
        Returns
        -------
        Union[DecisionTreeClassifier, DecisionTreeRegressor]
            The fitted surrogate tree (type depends on task).
        """
        self.n_features_ = X_train.shape[1]
        self.feature_names_ = feature_names or [f"feature_{i}" for i in range(self.n_features_)]
        
        # Detect task type (classification vs regression)
        if task_type is not None:
            is_regression = task_type == 'regression'
        else:
            try:
                is_regression = is_regressor(model)
            except AttributeError:
                is_regression = getattr(model, '_estimator_type', 'classifier') == 'regressor'
        
        # Generate synthetic samples around decision boundaries
        X_synthetic = self._generate_boundary_samples(X_train)
        
        # Get predictions from the original model
        y_synthetic = model.predict(X_synthetic)
        
        # Store number of classes (only for classification)
        if not is_regression:
            self.n_classes_ = len(np.unique(y_synthetic))
        else:
            self.n_classes_ = None
        
        # Train surrogate decision tree (classification or regression)
        if is_regression:
            self.surrogate_tree_ = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state,
                min_samples_split=max(2, self.n_samples // 1000),
                min_samples_leaf=max(1, self.n_samples // 2000)
            )
        else:
            self.surrogate_tree_ = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state,
                min_samples_split=max(2, self.n_samples // 1000),
                min_samples_leaf=max(1, self.n_samples // 2000)
            )
        
        self.surrogate_tree_.fit(X_synthetic, y_synthetic)
        
        return self.surrogate_tree_
    
    def _generate_boundary_samples(self, X_train: np.ndarray) -> np.ndarray:
        """
        Generate samples focused on decision boundaries.
        
        Uses a combination of:
        1. Random sampling within feature ranges
        2. Perturbations around training samples
        3. Grid sampling in critical regions
        """
        rng = np.random.RandomState(self.random_state)
        
        # Calculate feature ranges
        feature_min = X_train.min(axis=0)
        feature_max = X_train.max(axis=0)
        feature_range = feature_max - feature_min
        
        # Handle case where range is zero (constant feature)
        feature_range = np.where(feature_range == 0, 1.0, feature_range)
        
        # Expand ranges slightly to cover edge cases
        feature_min -= 0.1 * feature_range
        feature_max += 0.1 * feature_range
        
        samples = []
        
        # 1. Random uniform sampling (40% of samples)
        n_uniform = int(0.4 * self.n_samples)
        uniform_samples = rng.uniform(
            feature_min,
            feature_max,
            size=(n_uniform, self.n_features_)
        )
        samples.append(uniform_samples)
        
        # 2. Perturbations around training samples (40% of samples)
        n_perturbed = int(0.4 * self.n_samples)
        indices = rng.choice(len(X_train), size=n_perturbed, replace=True)
        noise = rng.normal(0, 0.1 * feature_range, size=(n_perturbed, self.n_features_))
        perturbed_samples = X_train[indices] + noise
        samples.append(perturbed_samples)
        
        # 3. Include original training samples (20% of samples)
        n_original = self.n_samples - n_uniform - n_perturbed
        if n_original > len(X_train):
            indices = rng.choice(len(X_train), size=n_original, replace=True)
        else:
            indices = rng.choice(len(X_train), size=n_original, replace=False)
        samples.append(X_train[indices])
        
        # Combine all samples
        X_synthetic = np.vstack(samples)
        
        # Clip to valid ranges
        X_synthetic = np.clip(X_synthetic, feature_min, feature_max)
        
        return X_synthetic
    
    def get_fidelity(self, model: BaseEstimator, X_test: np.ndarray) -> float:
        """
        Calculate fidelity: agreement between surrogate and original model.
        
        For classification: accuracy (exact match)
        For regression: R^2 score (coefficient of determination)
        
        Parameters
        ----------
        model : BaseEstimator
            The original model.
        X_test : np.ndarray
            Test data.
        
        Returns
        -------
        float
            Fidelity score (0-1 for classification, -inf to 1 for regression).
            Higher is better.
        """
        if self.surrogate_tree_ is None:
            raise ValueError("Must call extract() first")
        
        y_original = model.predict(X_test)
        y_surrogate = self.surrogate_tree_.predict(X_test)
        
        # Detect if regression or classification
        is_regression = isinstance(self.surrogate_tree_, DecisionTreeRegressor)
        
        if is_regression:
            # For regression: use R^2 score
            from sklearn.metrics import r2_score
            return r2_score(y_original, y_surrogate)
        else:
            # For classification: use accuracy
            return np.mean(y_original == y_surrogate)
