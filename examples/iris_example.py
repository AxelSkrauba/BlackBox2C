"""
Iris Dataset Example - Demonstrates BlackBox2C conversion.

This example shows how to:
1. Train different types of models on Iris dataset
2. Convert them to C code using BlackBox2C
3. Evaluate the conversion quality
4. Compare different configurations
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blackbox2c import convert, Converter, ConversionConfig


def load_and_prepare_data():
    """Load and split the Iris dataset."""
    print("="*70)
    print("LOADING IRIS DATASET")
    print("="*70)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {iris.feature_names}")
    print(f"Classes: {iris.target_names.tolist()}\n")
    
    return X_train, X_test, y_train, y_test, iris.feature_names, iris.target_names


def train_models(X_train, y_train):
    """Train different types of models."""
    print("="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    models = {}
    
    # Decision Tree
    print("\n[1/4] Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    models['DecisionTree'] = dt
    print("  [OK] Complete")
    
    # Random Forest
    print("\n[2/4] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    print("  [OK] Complete")
    
    # SVM
    print("\n[3/4] Training SVM...")
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    models['SVM'] = svm
    print("  [OK] Complete")
    
    # Neural Network
    print("\n[4/4] Training Neural Network...")
    mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    models['NeuralNetwork'] = mlp
    print("  [OK] Complete\n")
    
    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model accuracy."""
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\n{model_name} Performance:")
    print(f"  Training accuracy: {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    
    return train_acc, test_acc


def convert_and_save(model, X_train, X_test, feature_names, class_names, 
                     model_name, config):
    """Convert model to C and save to file."""
    print(f"\n{'='*70}")
    print(f"CONVERTING {model_name.upper()} TO C CODE")
    print(f"{'='*70}")
    
    converter = Converter(config)
    c_code = converter.convert(
        model=model,
        X_train=X_train,
        X_test=X_test,
        feature_names=[f"features[{i}]" for i in range(len(feature_names))],
        class_names=[name.upper() for name in class_names]
    )
    
    # Save to file
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{output_dir}/{model_name.lower()}_model.c"
    with open(filename, 'w') as f:
        f.write(c_code)
    
    print(f"\n[OK] C code saved to: {filename}")
    
    # Get and display metrics
    metrics = converter.get_metrics()
    print(f"\nConversion Metrics:")
    print(f"  Fidelity: {metrics['fidelity']:.4f}")
    print(f"  Code size: ~{metrics['size_estimate']['flash_bytes']} bytes")
    print(f"  Tree depth: {metrics['complexity']['max_depth']}")
    print(f"  Number of rules: {metrics['complexity']['n_internal_nodes']}")
    
    return c_code, metrics


def compare_configurations(model, X_train, X_test, feature_names, class_names):
    """Compare different conversion configurations."""
    print(f"\n{'='*70}")
    print("COMPARING DIFFERENT CONFIGURATIONS")
    print(f"{'='*70}")
    
    configs = [
        ("Minimal (max_depth=3)", ConversionConfig(max_depth=3, optimize_rules='high')),
        ("Balanced (max_depth=5)", ConversionConfig(max_depth=5, optimize_rules='medium')),
        ("Detailed (max_depth=7)", ConversionConfig(max_depth=7, optimize_rules='low')),
    ]
    
    results = []
    
    for config_name, config in configs:
        print(f"\n{config_name}:")
        print("-" * 50)
        
        converter = Converter(config)
        c_code = converter.convert(
            model=model,
            X_train=X_train,
            X_test=X_test,
            feature_names=[f"features[{i}]" for i in range(len(feature_names))],
            class_names=[name.upper() for name in class_names]
        )
        
        metrics = converter.get_metrics()
        results.append({
            'config': config_name,
            'fidelity': metrics['fidelity'],
            'code_size': metrics['size_estimate']['flash_bytes'],
            'depth': metrics['complexity']['max_depth'],
            'nodes': metrics['complexity']['n_nodes']
        })
    
    # Display comparison table
    print(f"\n{'='*70}")
    print("CONFIGURATION COMPARISON")
    print(f"{'='*70}")
    print(f"{'Configuration':<25} {'Fidelity':<12} {'Size (bytes)':<15} {'Depth':<8} {'Nodes':<8}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['config']:<25} {r['fidelity']:<12.4f} {r['code_size']:<15} {r['depth']:<8} {r['nodes']:<8}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print(" "*15 + "BLACKBOX2C - IRIS EXAMPLE")
    print("="*70 + "\n")
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names, class_names = load_and_prepare_data()
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate all models
    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    for model_name, model in models.items():
        evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
    
    # Convert models to C
    print(f"\n{'='*70}")
    print("CONVERTING MODELS TO C CODE")
    print(f"{'='*70}\n")
    
    # Use default configuration for conversions
    default_config = ConversionConfig(
        max_depth=5,
        optimize_rules='medium',
        precision=8
    )
    
    converted_models = {}
    
    for model_name, model in models.items():
        c_code, metrics = convert_and_save(
            model, X_train, X_test, feature_names, class_names,
            model_name, default_config
        )
        converted_models[model_name] = {'code': c_code, 'metrics': metrics}
    
    # Compare configurations using Random Forest
    print(f"\n{'='*70}")
    print("CONFIGURATION ANALYSIS (using Random Forest)")
    print(f"{'='*70}")
    compare_configurations(
        models['RandomForest'],
        X_train, X_test,
        feature_names, class_names
    )
    
    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("\nAll models have been successfully converted to C code!")
    print("Generated files are in the 'output/' directory.")
    print("\nKey Findings:")
    print("  • Decision Tree: Direct conversion with perfect fidelity")
    print("  • Random Forest: Surrogate approximation with high fidelity")
    print("  • SVM & Neural Network: Black-box extraction works well")
    print("\nNext Steps:")
    print("  1. Review generated C files in output/ directory")
    print("  2. Compile and test on your target microcontroller")
    print("  3. Adjust max_depth and optimization level as needed")
    print("  4. Measure actual performance on hardware")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
