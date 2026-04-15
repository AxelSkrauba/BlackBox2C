"""
Feature analysis and selection module for BlackBox2C.

This module provides tools to analyze feature importance and sensitivity,
helping to identify redundant features and optimize the feature set for
embedded deployment.
"""

import numpy as np
from sklearn.base import BaseEstimator
from typing import Dict, List, Tuple, Optional


class FeatureSensitivityAnalyzer:
    """
    Analyze feature importance and sensitivity using permutation importance.
    
    This analyzer helps identify:
    - Critical features that significantly impact predictions
    - Redundant features that can be removed
    - Optimal feature subset for minimal sensor requirements
    
    Parameters
    ----------
    n_repeats : int, default=10
        Number of times to permute each feature for robust estimates.
        
    random_state : int, default=42
        Random seed for reproducibility.
    
    Examples
    --------
    >>> from blackbox2c.analysis import FeatureSensitivityAnalyzer
    >>> analyzer = FeatureSensitivityAnalyzer()
    >>> results = analyzer.analyze(model, X_train, y_train)
    >>> print(results.summary())
    >>> optimal_features = results.get_optimal_subset(threshold=0.01)
    """
    
    def __init__(self, n_repeats: int = 10, random_state: int = 42):
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.results_ = None
    
    def analyze(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'SensitivityResults':
        """
        Analyze feature sensitivity using permutation importance.
        
        The method works by:
        1. Computing baseline model score
        2. For each feature, permute its values and measure score drop
        3. Repeat multiple times for robust estimates
        4. Return importance scores and recommendations
        
        Parameters
        ----------
        model : BaseEstimator
            Trained scikit-learn model to analyze.
            
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
            
        y : np.ndarray
            Target values of shape (n_samples,).
            
        feature_names : list of str, optional
            Names of features. If None, uses "feature_0", "feature_1", etc.
        
        Returns
        -------
        SensitivityResults
            Object containing importance scores, statistics, and recommendations.
        
        Examples
        --------
        >>> results = analyzer.analyze(model, X_train, y_train, 
        ...                            feature_names=['temp', 'humidity', 'pressure'])
        >>> print(f"Most important: {results.get_top_features(3)}")
        """
        # Validate inputs
        if not hasattr(model, 'score'):
            raise TypeError("Model must have a score() method")
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        
        if len(y) != len(X):
            raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
        
        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match "
                f"number of features ({X.shape[1]})"
            )
        
        # Baseline score
        baseline_score = model.score(X, y)
        
        # Permutation importance
        importances = {}
        importances_std = {}
        
        rng = np.random.RandomState(self.random_state)
        
        for feat_idx in range(X.shape[1]):
            scores = []
            
            for _ in range(self.n_repeats):
                # Permute feature
                X_permuted = X.copy()
                X_permuted[:, feat_idx] = rng.permutation(X_permuted[:, feat_idx])
                
                # Measure impact
                permuted_score = model.score(X_permuted, y)
                importance = baseline_score - permuted_score
                scores.append(importance)
            
            importances[feat_idx] = np.mean(scores)
            importances_std[feat_idx] = np.std(scores)
        
        self.results_ = SensitivityResults(
            feature_names=feature_names,
            importances=importances,
            importances_std=importances_std,
            baseline_score=baseline_score
        )
        
        return self.results_


class SensitivityResults:
    """
    Results from feature sensitivity analysis.
    
    Attributes
    ----------
    feature_names : list of str
        Names of analyzed features.
        
    importances : dict
        Feature index -> importance score mapping.
        
    importances_std : dict
        Feature index -> standard deviation of importance.
        
    baseline_score : float
        Model score before any permutation.
    """
    
    def __init__(
        self,
        feature_names: List[str],
        importances: Dict[int, float],
        importances_std: Dict[int, float],
        baseline_score: float
    ):
        self.feature_names = feature_names
        self.importances = importances
        self.importances_std = importances_std
        self.baseline_score = baseline_score
    
    def summary(self) -> str:
        """
        Generate human-readable summary of analysis.
        
        Returns
        -------
        str
            Formatted summary with importance scores and recommendations.
        
        Examples
        --------
        >>> print(results.summary())
        Feature Sensitivity Analysis
        ====================================
        Feature 0 (temp): Impact = 0.85 ± 0.02 (Critical)
        Feature 1 (humidity): Impact = 0.12 ± 0.03 (Medium)
        Feature 2 (pressure): Impact = 0.01 ± 0.01 (Very Low)
        
        Recommendations:
        - Consider removing 1 low-impact features
        - Keep 1 critical features
        """
        lines = ["Feature Sensitivity Analysis", "=" * 50]
        
        # Sort by importance (descending)
        sorted_features = sorted(
            self.importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Display each feature
        for feat_idx, importance in sorted_features:
            name = self.feature_names[feat_idx]
            std = self.importances_std[feat_idx]
            
            # Classify importance level
            if importance > 0.5:
                level = "Critical"
            elif importance > 0.2:
                level = "High"
            elif importance > 0.05:
                level = "Medium"
            elif importance > 0.01:
                level = "Low"
            else:
                level = "Very Low"
            
            lines.append(
                f"Feature {feat_idx} ({name}): "
                f"Impact = {importance:.4f} ± {std:.4f} ({level})"
            )
        
        # Generate recommendations
        lines.append("\nRecommendations:")
        
        # Low impact features
        low_impact = [
            (idx, imp) for idx, imp in sorted_features
            if imp < 0.01
        ]
        
        if low_impact:
            lines.append(f"- Consider removing {len(low_impact)} low-impact feature(s):")
            for idx, imp in low_impact[:5]:  # Show top 5
                lines.append(f"  * Feature {idx} ({self.feature_names[idx]})")
        
        # Critical features
        critical = [
            (idx, imp) for idx, imp in sorted_features
            if imp > 0.5
        ]
        
        if critical:
            lines.append(f"- Keep {len(critical)} critical feature(s):")
            for idx, imp in critical[:5]:
                lines.append(f"  * Feature {idx} ({self.feature_names[idx]})")
        
        # Medium/High features
        important = [
            (idx, imp) for idx, imp in sorted_features
            if 0.05 <= imp <= 0.5
        ]
        
        if important:
            lines.append(f"- {len(important)} feature(s) have moderate impact")
        
        return "\n".join(lines)
    
    def get_optimal_subset(
        self,
        threshold: float = 0.01,
        min_features: int = 1
    ) -> List[int]:
        """
        Get optimal feature subset based on importance threshold.
        
        Parameters
        ----------
        threshold : float, default=0.01
            Minimum importance to include a feature.
            Features with importance below this are excluded.
            
        min_features : int, default=1
            Minimum number of features to keep, even if below threshold.
            Ensures at least some features are selected.
        
        Returns
        -------
        list of int
            Indices of selected features, sorted in ascending order.
        
        Examples
        --------
        >>> # Keep features with >1% impact
        >>> selected = results.get_optimal_subset(threshold=0.01)
        >>> X_reduced = X[:, selected]
        
        >>> # Keep top 3 features minimum
        >>> selected = results.get_optimal_subset(threshold=0.05, min_features=3)
        """
        # Sort by importance (descending)
        sorted_features = sorted(
            self.importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select features above threshold
        selected = [idx for idx, imp in sorted_features if imp >= threshold]
        
        # Ensure minimum number of features
        if len(selected) < min_features:
            selected = [idx for idx, _ in sorted_features[:min_features]]
        
        # Return sorted indices
        return sorted(selected)
    
    def get_top_features(self, n: int = 5) -> List[Tuple[int, str, float]]:
        """
        Get top N most important features.
        
        Parameters
        ----------
        n : int, default=5
            Number of top features to return.
        
        Returns
        -------
        list of tuple
            List of (index, name, importance) for top N features.
        
        Examples
        --------
        >>> top_features = results.get_top_features(3)
        >>> for idx, name, importance in top_features:
        ...     print(f"{name}: {importance:.3f}")
        """
        sorted_features = sorted(
            self.importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            (idx, self.feature_names[idx], imp)
            for idx, imp in sorted_features[:n]
        ]
    
    def get_redundant_features(self, threshold: float = 0.01) -> List[int]:
        """
        Get features that can potentially be removed (low importance).
        
        Parameters
        ----------
        threshold : float, default=0.01
            Features with importance below this are considered redundant.
        
        Returns
        -------
        list of int
            Indices of redundant features.
        
        Examples
        --------
        >>> redundant = results.get_redundant_features(threshold=0.01)
        >>> print(f"Can remove {len(redundant)} features")
        """
        return [
            idx for idx, imp in self.importances.items()
            if imp < threshold
        ]
    
    def plot(self, figsize=(10, 6), save_path=None):
        """
        Plot feature importances with error bars.
        
        Parameters
        ----------
        figsize : tuple, default=(10, 6)
            Figure size (width, height) in inches.
            
        save_path : str, optional
            If provided, save plot to this path.
        
        Returns
        -------
        fig, ax
            Matplotlib figure and axes objects.
        
        Raises
        ------
        ImportError
            If matplotlib is not installed.
        
        Examples
        --------
        >>> fig, ax = results.plot(figsize=(12, 8))
        >>> plt.show()
        
        >>> # Save to file
        >>> results.plot(save_path='feature_importance.png')
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            )
        
        # Sort by importance
        sorted_features = sorted(
            self.importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        indices = [idx for idx, _ in sorted_features]
        importances = [imp for _, imp in sorted_features]
        stds = [self.importances_std[idx] for idx in indices]
        names = [self.feature_names[idx] for idx in indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(indices))
        ax.barh(y_pos, importances, xerr=stds, alpha=0.7, color='steelblue')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Importance (Permutation)', fontsize=12)
        ax.set_title('Feature Sensitivity Analysis', fontsize=14, fontweight='bold')
        
        # Add threshold lines
        ax.axvline(x=0.01, color='red', linestyle='--', linewidth=1, 
                   label='Low threshold (0.01)', alpha=0.7)
        ax.axvline(x=0.05, color='orange', linestyle='--', linewidth=1,
                   label='Medium threshold (0.05)', alpha=0.7)
        ax.axvline(x=0.2, color='green', linestyle='--', linewidth=1,
                   label='High threshold (0.2)', alpha=0.7)
        
        ax.legend(loc='best')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        return fig, ax
