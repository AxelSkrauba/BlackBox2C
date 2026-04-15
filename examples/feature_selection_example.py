"""
Feature Selection Example - BlackBox2C

This example demonstrates how to use the FeatureSensitivityAnalyzer to:
1. Identify important features
2. Remove redundant features
3. Optimize feature set for embedded deployment
"""

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from blackbox2c.analysis import FeatureSensitivityAnalyzer
from blackbox2c import convert, ConversionConfig


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70 + "\n")


def example_1_basic_analysis():
    """Example 1: Basic feature sensitivity analysis."""
    print_section("EXAMPLE 1: BASIC FEATURE SENSITIVITY ANALYSIS")
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print("Dataset: Iris")
    print(f"Features: {len(iris.feature_names)}")
    print(f"Feature names: {iris.feature_names}")
    print(f"Samples: {len(X)}\n")
    
    # Train model
    print("[1/3] Training Random Forest...")
    model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
    model.fit(X, y)
    print(f"  Training accuracy: {model.score(X, y):.4f}\n")
    
    # Analyze feature sensitivity
    print("[2/3] Analyzing feature sensitivity...")
    analyzer = FeatureSensitivityAnalyzer(n_repeats=10, random_state=42)
    results = analyzer.analyze(model, X, y, feature_names=iris.feature_names)
    print("  Analysis complete!\n")
    
    # Display results
    print("[3/3] Results:")
    print(results.summary())
    
    # Get top features
    print("\n" + "-" * 70)
    print("Top 3 Most Important Features:")
    print("-" * 70)
    for idx, name, importance in results.get_top_features(3):
        print(f"  {idx}. {name:25s} - Impact: {importance:.4f}")


def example_2_feature_selection():
    """Example 2: Feature selection for code size optimization."""
    print_section("EXAMPLE 2: FEATURE SELECTION FOR CODE SIZE OPTIMIZATION")
    
    # Create dataset with some redundant features
    print("Creating synthetic dataset with redundant features...")
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=4,
        n_redundant=2,
        n_repeated=0,
        n_classes=3,
        random_state=42
    )
    
    feature_names = [f"sensor_{i}" for i in range(8)]
    
    print(f"  Features: {len(feature_names)}")
    print(f"  Samples: {len(X)}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train full model
    print("[1/4] Training model with ALL features...")
    model_full = RandomForestClassifier(n_estimators=20, random_state=42)
    model_full.fit(X_train, y_train)
    score_full = model_full.score(X_test, y_test)
    print(f"  Test accuracy: {score_full:.4f}\n")
    
    # Convert to C (full model)
    print("[2/4] Converting full model to C...")
    config_full = ConversionConfig(max_depth=5)
    c_code_full = convert(model_full, X_train, config=config_full)
    size_full = len(c_code_full)
    print(f"  Code size: {size_full} bytes\n")
    
    # Analyze and select features
    print("[3/4] Analyzing features and selecting optimal subset...")
    analyzer = FeatureSensitivityAnalyzer(n_repeats=10)
    results = analyzer.analyze(model_full, X_train, y_train, feature_names=feature_names)
    
    # Select features with >5% impact
    selected_features = results.get_optimal_subset(threshold=0.05, min_features=3)
    print(f"  Selected {len(selected_features)} features: {selected_features}")
    print(f"  Removed {len(feature_names) - len(selected_features)} redundant features\n")
    
    # Train reduced model
    print("[4/4] Training model with SELECTED features...")
    X_train_reduced = X_train[:, selected_features]
    X_test_reduced = X_test[:, selected_features]
    
    model_reduced = RandomForestClassifier(n_estimators=20, random_state=42)
    model_reduced.fit(X_train_reduced, y_train)
    score_reduced = model_reduced.score(X_test_reduced, y_test)
    print(f"  Test accuracy: {score_reduced:.4f}\n")
    
    # Convert to C (reduced model)
    print("  Converting reduced model to C...")
    c_code_reduced = convert(model_reduced, X_train_reduced, config=config_full)
    size_reduced = len(c_code_reduced)
    print(f"  Code size: {size_reduced} bytes\n")
    
    # Compare results
    print("-" * 70)
    print("COMPARISON:")
    print("-" * 70)
    print(f"{'Metric':<30} {'Full Model':<20} {'Reduced Model':<20}")
    print("-" * 70)
    print(f"{'Features':<30} {len(feature_names):<20} {len(selected_features):<20}")
    print(f"{'Test Accuracy':<30} {score_full:<20.4f} {score_reduced:<20.4f}")
    print(f"{'Code Size (bytes)':<30} {size_full:<20} {size_reduced:<20}")
    print(f"{'Size Reduction':<30} {'-':<20} {(1 - size_reduced/size_full)*100:.1f}%")
    print(f"{'Accuracy Loss':<30} {'-':<20} {(score_full - score_reduced)*100:.2f}%")
    
    # Recommendations
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS:")
    print("-" * 70)
    
    if size_reduced < size_full * 0.8:
        print("[OK] Significant code size reduction achieved!")
    
    if score_reduced >= score_full - 0.05:
        print("[OK] Accuracy maintained within acceptable range!")
    
    redundant = results.get_redundant_features(threshold=0.01)
    if redundant:
        print(f"[OK] Can safely remove {len(redundant)} sensor(s) from hardware")


def example_3_threshold_comparison():
    """Example 3: Compare different selection thresholds."""
    print_section("EXAMPLE 3: COMPARING DIFFERENT SELECTION THRESHOLDS")
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Train model
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)
    
    # Analyze
    analyzer = FeatureSensitivityAnalyzer(n_repeats=10)
    results = analyzer.analyze(model, X, y, feature_names=iris.feature_names)
    
    # Try different thresholds
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.2]
    
    print("Threshold Analysis:")
    print("-" * 70)
    print(f"{'Threshold':<15} {'Features Selected':<20} {'Features Kept':<30}")
    print("-" * 70)
    
    for threshold in thresholds:
        selected = results.get_optimal_subset(threshold=threshold, min_features=1)
        kept_names = [iris.feature_names[i] for i in selected]
        
        print(f"{threshold:<15.3f} {len(selected):<20} {', '.join(kept_names):<30}")
    
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("-" * 70)
    print("- Lower threshold: More features, higher accuracy, larger code")
    print("- Higher threshold: Fewer features, lower accuracy, smaller code")
    print("- Recommended: Start with 0.01 threshold (1% impact)")


def example_4_visualization():
    """Example 4: Visualize feature importances."""
    print_section("EXAMPLE 4: FEATURE IMPORTANCE VISUALIZATION")
    
    try:
        import matplotlib.pyplot as plt
        
        # Load data
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Train and analyze
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)
        
        analyzer = FeatureSensitivityAnalyzer(n_repeats=10)
        results = analyzer.analyze(model, X, y, feature_names=iris.feature_names)
        
        # Create plot
        print("Generating plot...")
        fig, ax = results.plot(figsize=(10, 6))
        
        # Save plot
        output_path = "output/feature_importance.png"
        results.plot(figsize=(10, 6), save_path=output_path)
        
        print(f"\n[OK] Plot saved to: {output_path}")
        print("  Open the file to view feature importance visualization")
        
    except ImportError:
        print("[WARNING] matplotlib not installed. Skipping visualization.")
        print("  Install with: pip install matplotlib")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("FEATURE SELECTION EXAMPLES - BLACKBOX2C".center(70))
    print("=" * 70)
    
    # Run examples
    example_1_basic_analysis()
    example_2_feature_selection()
    example_3_threshold_comparison()
    example_4_visualization()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY".center(70))
    print("=" * 70)
    print("""
Feature sensitivity analysis helps you:

1. Identify which sensors/features are actually needed
2. Reduce hardware costs by eliminating redundant sensors
3. Minimize code size for embedded deployment
4. Maintain accuracy while optimizing resources

Next Steps:
- Use feature_threshold parameter in ConversionConfig
- Integrate analysis into your conversion workflow
- Test on your specific hardware constraints
""")


if __name__ == "__main__":
    main()
