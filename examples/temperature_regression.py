"""
Temperature Prediction Example - Regression

This example demonstrates using BlackBox2C for regression tasks.
We'll predict temperature based on time of day, humidity, and pressure.

This is a practical example for embedded weather stations or IoT devices.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from blackbox2c import convert
from blackbox2c.config import ConversionConfig


def generate_temperature_data(n_samples=500):
    """
    Generate synthetic temperature data.
    
    Features:
    - hour: Hour of day (0-23)
    - humidity: Relative humidity (0-100%)
    - pressure: Atmospheric pressure (980-1040 hPa)
    
    Target:
    - temperature: Temperature in Celsius
    """
    np.random.seed(42)
    
    # Generate features
    hour = np.random.uniform(0, 24, n_samples)
    humidity = np.random.uniform(30, 90, n_samples)
    pressure = np.random.uniform(980, 1040, n_samples)
    
    # Generate temperature with realistic patterns
    # Base temperature varies with time of day (sinusoidal)
    base_temp = 15 + 10 * np.sin((hour - 6) * np.pi / 12)
    
    # Higher humidity -> slightly lower temperature
    humidity_effect = -0.05 * (humidity - 60)
    
    # Higher pressure -> slightly higher temperature
    pressure_effect = 0.1 * (pressure - 1010)
    
    # Add some noise
    noise = np.random.normal(0, 2, n_samples)
    
    temperature = base_temp + humidity_effect + pressure_effect + noise
    
    # Combine features
    X = np.column_stack([hour, humidity, pressure])
    
    return X, temperature


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def example_1_simple_regression():
    """Example 1: Simple decision tree regression."""
    print_section("EXAMPLE 1: SIMPLE TEMPERATURE PREDICTION")
    
    # Generate data
    X, y = generate_temperature_data(n_samples=300)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    print("\nTraining DecisionTreeRegressor...")
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"  Mean Absolute Error: {mae:.2f} C")
    print(f"  R^2 Score: {r2:.4f}")
    
    # Convert to C
    print("\nConverting to C code...")
    feature_names = ['hour', 'humidity', 'pressure']
    
    config = ConversionConfig(
        function_name='predict_temperature',
        max_depth=5
    )
    
    c_code = convert(
        model,
        X_train,
        X_test=X_test,
        feature_names=feature_names,
        config=config
    )
    
    # Save to file
    output_file = 'output/temperature_predictor.c'
    with open(output_file, 'w') as f:
        f.write(c_code)
    
    print(f"\n[OK] C code saved to: {output_file}")
    print(f"  Function signature: float predict_temperature(float features[3])")
    print(f"  Input: [hour, humidity, pressure]")
    print(f"  Output: temperature in Celsius")


def example_2_ensemble_regression():
    """Example 2: Random Forest regression with optimization."""
    print_section("EXAMPLE 2: ENSEMBLE MODEL WITH OPTIMIZATION")
    
    # Generate data
    X, y = generate_temperature_data(n_samples=500)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train ensemble model
    print("\nTraining RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=20,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Ensemble Model Performance:")
    print(f"  Mean Absolute Error: {mae:.2f} C")
    print(f"  R^2 Score: {r2:.4f}")
    
    # Convert with optimization
    print("\nConverting to C with high optimization...")
    feature_names = ['hour', 'humidity', 'pressure']
    
    config = ConversionConfig(
        function_name='predict_temp_optimized',
        max_depth=5,
        optimize_rules='high'
    )
    
    c_code = convert(
        model,
        X_train,
        X_test=X_test,
        feature_names=feature_names,
        config=config
    )
    
    # Save to file
    output_file = 'output/temperature_optimized.c'
    with open(output_file, 'w') as f:
        f.write(c_code)
    
    print(f"\n[OK] Optimized C code saved to: {output_file}")
    
    # Show code size
    lines = c_code.count('\n')
    print(f"  Code size: {lines} lines, {len(c_code)} bytes")


def example_3_fixed_point_regression():
    """Example 3: Fixed-point arithmetic for embedded systems."""
    print_section("EXAMPLE 3: FIXED-POINT FOR EMBEDDED SYSTEMS")
    
    # Generate data
    X, y = generate_temperature_data(n_samples=300)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train simple model
    print("\nTraining simple model for embedded deployment...")
    model = DecisionTreeRegressor(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Model MAE: {mae:.2f} C")
    
    # Convert with fixed-point arithmetic
    print("\nConverting to C with 16-bit fixed-point...")
    feature_names = ['hour', 'humidity', 'pressure']
    
    config = ConversionConfig(
        function_name='predict_temp_embedded',
        max_depth=4,
        use_fixed_point=True,
        precision=16
    )
    
    c_code = convert(
        model,
        X_train,
        feature_names=feature_names,
        config=config
    )
    
    # Save to file
    output_file = 'output/temperature_embedded.c'
    with open(output_file, 'w') as f:
        f.write(c_code)
    
    print(f"\n[OK] Embedded C code saved to: {output_file}")
    print(f"  Uses 16-bit fixed-point arithmetic")
    print(f"  Suitable for: Arduino, STM32, ESP32, etc.")
    print(f"  Memory footprint: Minimal (no floating-point)")


def example_4_model_comparison():
    """Example 4: Compare different models and configurations."""
    print_section("EXAMPLE 4: MODEL COMPARISON")
    
    # Generate data
    X, y = generate_temperature_data(n_samples=400)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    feature_names = ['hour', 'humidity', 'pressure']
    
    models = [
        ("Simple Tree (depth=3)", DecisionTreeRegressor(max_depth=3, random_state=42)),
        ("Medium Tree (depth=5)", DecisionTreeRegressor(max_depth=5, random_state=42)),
        ("Deep Tree (depth=7)", DecisionTreeRegressor(max_depth=7, random_state=42)),
        ("Random Forest", RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42))
    ]
    
    print("\nComparing models:\n")
    print(f"{'Model':<25} {'MAE (C)':<12} {'R^2':<10} {'Code Size':<12}")
    print("-" * 70)
    
    for name, model in models:
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Convert
        if "Forest" in name:
            config = ConversionConfig(max_depth=5)
            c_code = convert(model, X_train, feature_names=feature_names, config=config)
        else:
            c_code = convert(model, X_train, feature_names=feature_names)
        
        code_size = len(c_code)
        
        print(f"{name:<25} {mae:<12.2f} {r2:<10.4f} {code_size:<12} bytes")
    
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("-" * 70)
    print("- Deeper trees: Better accuracy, larger code size")
    print("- Random Forest: Best accuracy, but requires surrogate approximation")
    print("- Simple trees: Good balance for embedded systems")


def example_5_usage_example():
    """Example 5: Show how to use generated C code."""
    print_section("EXAMPLE 5: USING THE GENERATED C CODE")
    
    print("""
The generated C code can be used in your embedded project like this:

1. Include the generated file:
   
   #include "temperature_predictor.c"

2. Prepare your input features:
   
   float features[3];
   features[0] = 14.5;  // hour (2:30 PM)
   features[1] = 65.0;  // humidity (65%)
   features[2] = 1013.0; // pressure (1013 hPa)

3. Call the prediction function:
   
   float temperature = predict_temperature(features);
   printf("Predicted temperature: %.1f C\\n", temperature);

4. Example Arduino sketch:
   
   void loop() {
       float features[3];
       features[0] = getHour();      // From RTC
       features[1] = getHumidity();  // From DHT22 sensor
       features[2] = getPressure();  // From BMP280 sensor
       
       float temp = predict_temperature(features);
       
       Serial.print("Predicted: ");
       Serial.print(temp);
       Serial.println(" C");
       
       delay(60000);  // Update every minute
   }

5. The code is:
   - Self-contained (no external dependencies)
   - Fast (simple if-else logic)
   - Small (typically < 2KB)
   - Portable (standard C)
""")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("  TEMPERATURE PREDICTION - REGRESSION EXAMPLES")
    print("  BlackBox2C - Convert ML models to embedded C code")
    print("=" * 70)
    
    # Create output directory
    import os
    os.makedirs('output', exist_ok=True)
    
    # Run examples
    example_1_simple_regression()
    example_2_ensemble_regression()
    example_3_fixed_point_regression()
    example_4_model_comparison()
    example_5_usage_example()
    
    print("\n" + "=" * 70)
    print("  ALL EXAMPLES COMPLETED!")
    print("=" * 70)
    print("\nGenerated files in output/ directory:")
    print("  - temperature_predictor.c")
    print("  - temperature_optimized.c")
    print("  - temperature_embedded.c")
    print("\nNext steps:")
    print("  1. Review the generated C code")
    print("  2. Integrate into your embedded project")
    print("  3. Test with real sensor data")
    print("  4. Deploy to your device!")


if __name__ == '__main__':
    main()
