"""
Multi-Format Export Example

This example demonstrates exporting decision trees to multiple formats:
- C++ (modern C++11 with classes)
- Arduino (optimized for Arduino boards)
- MicroPython (for microcontrollers running Python)

Each format is optimized for its target platform.
"""

import numpy as np
from sklearn.datasets import load_iris, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from blackbox2c.exporters import CppExporter, ArduinoExporter, MicroPythonExporter, create_exporter


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def example_1_cpp_classification():
    """Example 1: Export classification model to C++."""
    print_section("EXAMPLE 1: C++ EXPORT (CLASSIFICATION)")
    
    # Load and train model
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"\nModel accuracy: {model.score(X_test, y_test):.4f}")
    
    # Export to C++
    print("\nExporting to C++...")
    exporter = CppExporter(
        function_name='predict',
        class_name='IrisPredictor',
        use_namespace=True,
        namespace='ml'
    )
    
    cpp_code = exporter.generate(
        model,
        feature_names=iris.feature_names,
        class_names=iris.target_names.tolist()
    )
    
    # Save to file
    output_file = 'output/iris_predictor.hpp'
    with open(output_file, 'w') as f:
        f.write(cpp_code)
    
    print(f"[OK] C++ code saved to: {output_file}")
    print(f"  Class: ml::IrisPredictor")
    print(f"  Method: predict()")
    print(f"  Features: Modern C++11")
    print(f"  Size: {len(cpp_code)} bytes")


def example_2_arduino_classification():
    """Example 2: Export classification model to Arduino."""
    print_section("EXAMPLE 2: ARDUINO EXPORT (CLASSIFICATION)")
    
    # Load and train model
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"\nModel accuracy: {model.score(X_test, y_test):.4f}")
    
    # Export to Arduino
    print("\nExporting to Arduino...")
    exporter = ArduinoExporter(
        function_name='predict_iris',
        use_progmem=True
    )
    
    arduino_code = exporter.generate(
        model,
        feature_names=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid'],
        class_names=['setosa', 'versicolor', 'virginica']
    )
    
    # Save to file
    output_file = 'output/iris_predictor.ino'
    with open(output_file, 'w') as f:
        f.write(arduino_code)
    
    print(f"[OK] Arduino code saved to: {output_file}")
    print(f"  Function: predict_iris()")
    print(f"  PROGMEM: Yes (saves RAM)")
    print(f"  Compatible: Arduino Uno, Nano, Mega, ESP32, ESP8266")
    print(f"  Size: {len(arduino_code)} bytes")


def example_3_micropython_classification():
    """Example 3: Export classification model to MicroPython."""
    print_section("EXAMPLE 3: MICROPYTHON EXPORT (CLASSIFICATION)")
    
    # Load and train model
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"\nModel accuracy: {model.score(X_test, y_test):.4f}")
    
    # Export to MicroPython
    print("\nExporting to MicroPython...")
    exporter = MicroPythonExporter(
        function_name='predict',
        class_name='IrisPredictor',
        use_const=True
    )
    
    python_code = exporter.generate(
        model,
        feature_names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        class_names=['setosa', 'versicolor', 'virginica']
    )
    
    # Save to file
    output_file = 'output/iris_predictor.py'
    with open(output_file, 'w') as f:
        f.write(python_code)
    
    print(f"[OK] MicroPython code saved to: {output_file}")
    print(f"  Class: IrisPredictor")
    print(f"  Method: predict()")
    print(f"  Compatible: ESP32, ESP8266, Raspberry Pi Pico, PyBoard")
    print(f"  Size: {len(python_code)} bytes")


def example_4_cpp_regression():
    """Example 4: Export regression model to C++."""
    print_section("EXAMPLE 4: C++ EXPORT (REGRESSION)")
    
    # Generate regression data
    X, y = make_regression(n_samples=200, n_features=3, noise=10.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"\nModel R^2 score: {model.score(X_test, y_test):.4f}")
    
    # Export to C++
    print("\nExporting to C++...")
    exporter = CppExporter(
        function_name='predict_value',
        class_name='ValuePredictor',
        use_namespace=True,
        namespace='regression'
    )
    
    cpp_code = exporter.generate(
        model,
        feature_names=['feature_0', 'feature_1', 'feature_2']
    )
    
    # Save to file
    output_file = 'output/value_predictor.hpp'
    with open(output_file, 'w') as f:
        f.write(cpp_code)
    
    print(f"[OK] C++ code saved to: {output_file}")
    print(f"  Class: regression::ValuePredictor")
    print(f"  Method: predict_value()")
    print(f"  Return type: float")
    print(f"  Size: {len(cpp_code)} bytes")


def example_5_arduino_regression():
    """Example 5: Export regression model to Arduino."""
    print_section("EXAMPLE 5: ARDUINO EXPORT (REGRESSION)")
    
    # Generate temperature-like data
    np.random.seed(42)
    hour = np.random.uniform(0, 24, 200)
    humidity = np.random.uniform(30, 90, 200)
    temperature = 15 + 10 * np.sin((hour - 6) * np.pi / 12) - 0.05 * (humidity - 60)
    temperature += np.random.normal(0, 2, 200)
    
    X = np.column_stack([hour, humidity])
    y = temperature
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    model = DecisionTreeRegressor(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"\nModel R^2 score: {model.score(X_test, y_test):.4f}")
    
    # Export to Arduino
    print("\nExporting to Arduino...")
    exporter = ArduinoExporter(
        function_name='predict_temperature',
        use_progmem=False  # Regression doesn't need PROGMEM for strings
    )
    
    arduino_code = exporter.generate(
        model,
        feature_names=['hour', 'humidity']
    )
    
    # Save to file
    output_file = 'output/temperature_predictor.ino'
    with open(output_file, 'w') as f:
        f.write(arduino_code)
    
    print(f"[OK] Arduino code saved to: {output_file}")
    print(f"  Function: predict_temperature()")
    print(f"  Return type: float")
    print(f"  Use case: Weather station, IoT sensor")
    print(f"  Size: {len(arduino_code)} bytes")


def example_6_factory_pattern():
    """Example 6: Using the factory pattern for easy export."""
    print_section("EXAMPLE 6: FACTORY PATTERN")
    
    # Train a simple model
    iris = load_iris()
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(iris.data, iris.target)
    
    print("\nExporting to multiple formats using factory pattern...\n")
    
    formats = ['cpp', 'arduino', 'micropython']
    
    for fmt in formats:
        print(f"Exporting to {fmt.upper()}...")
        
        # Create exporter using factory
        exporter = create_exporter(fmt)
        
        # Generate code
        code = exporter.generate(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names.tolist()
        )
        
        # Save to file
        extensions = {'cpp': '.hpp', 'arduino': '.ino', 'micropython': '.py'}
        output_file = f'output/iris_factory_{fmt}{extensions[fmt]}'
        
        with open(output_file, 'w') as f:
            f.write(code)
        
        print(f"  [OK] Saved to: {output_file} ({len(code)} bytes)\n")


def example_7_comparison():
    """Example 7: Compare code sizes across formats."""
    print_section("EXAMPLE 7: FORMAT COMPARISON")
    
    # Train model
    iris = load_iris()
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(iris.data, iris.target)
    
    print("\nGenerating code in all formats...\n")
    print(f"{'Format':<20} {'Size (bytes)':<15} {'Lines':<10} {'Features':<30}")
    print("-" * 75)
    
    # C++ export
    cpp_exporter = CppExporter()
    cpp_code = cpp_exporter.generate(
        model, iris.feature_names, iris.target_names.tolist()
    )
    cpp_lines = cpp_code.count('\n')
    print(f"{'C++':<20} {len(cpp_code):<15} {cpp_lines:<10} {'Classes, templates, STL':<30}")
    
    # Arduino export
    arduino_exporter = ArduinoExporter()
    arduino_code = arduino_exporter.generate(
        model, iris.feature_names, iris.target_names.tolist()
    )
    arduino_lines = arduino_code.count('\n')
    print(f"{'Arduino':<20} {len(arduino_code):<15} {arduino_lines:<10} {'PROGMEM, low memory':<30}")
    
    # MicroPython export
    micropython_exporter = MicroPythonExporter()
    micropython_code = micropython_exporter.generate(
        model, iris.feature_names, iris.target_names.tolist()
    )
    micropython_lines = micropython_code.count('\n')
    print(f"{'MicroPython':<20} {len(micropython_code):<15} {micropython_lines:<10} {'Pure Python, readable':<30}")
    
    print("\n" + "-" * 75)
    print("INTERPRETATION:")
    print("-" * 75)
    print("- C++: Best for performance-critical applications")
    print("- Arduino: Best for resource-constrained boards")
    print("- MicroPython: Best for rapid prototyping and readability")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("  MULTI-FORMAT EXPORT EXAMPLES")
    print("  BlackBox2C - Export to C++, Arduino, and MicroPython")
    print("=" * 70)
    
    # Create output directory
    import os
    os.makedirs('output', exist_ok=True)
    
    # Run examples
    example_1_cpp_classification()
    example_2_arduino_classification()
    example_3_micropython_classification()
    example_4_cpp_regression()
    example_5_arduino_regression()
    example_6_factory_pattern()
    example_7_comparison()
    
    print("\n" + "=" * 70)
    print("  ALL EXAMPLES COMPLETED!")
    print("=" * 70)
    print("\nGenerated files in output/ directory:")
    print("  C++:")
    print("    - iris_predictor.hpp")
    print("    - value_predictor.hpp")
    print("  Arduino:")
    print("    - iris_predictor.ino")
    print("    - temperature_predictor.ino")
    print("  MicroPython:")
    print("    - iris_predictor.py")
    print("\nNext steps:")
    print("  1. Review the generated code for your target platform")
    print("  2. Copy to your project directory")
    print("  3. Compile/upload and test")
    print("  4. Deploy to your device!")


if __name__ == '__main__':
    main()
