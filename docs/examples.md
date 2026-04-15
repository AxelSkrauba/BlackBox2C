# Examples

All examples are in the [`examples/`](https://github.com/AxelSkrauba/BlackBox2C/tree/main/examples) directory.

---

## iris_example.py

Full end-to-end demonstration using the Iris dataset:

- Trains Decision Tree, Random Forest, SVM, and Neural Network
- Converts each to C with different configurations
- Compares fidelity, code size, and complexity
- Shows feature analysis workflow

```bash
python examples/iris_example.py
```

---

## Code Recipes

### Fixed-point for AVR (Arduino Uno)

Arduino Uno's ATmega328P has no FPU. Use 16-bit fixed-point:

```python
from blackbox2c import convert, ConversionConfig

config = ConversionConfig(
    max_depth=4,
    use_fixed_point=True,
    precision=16,
    optimize_rules='high',
    memory_budget_kb=1.0,
)

arduino_code = convert(
    model, X_train,
    target='arduino',
    config=config,
)
```

### Minimal footprint for ATtiny

```python
config = ConversionConfig(
    max_depth=3,
    use_fixed_point=True,
    precision=8,
    optimize_rules='high',
)
```

### ESP32 / Raspberry Pi Pico (MicroPython)

```python
mp_code = convert(model, X_train, target='micropython', max_depth=6)

# Save and flash
with open('predictor.py', 'w') as f:
    f.write(mp_code)
```

On device:
```python
from predictor import Predictor
result = Predictor.predict([0.5, 1.2, 3.1, 0.8])
```

### Feature selection for sensor-constrained systems

When you have many features but limited sensors:

```python
from blackbox2c.analysis import FeatureSensitivityAnalyzer
from blackbox2c import convert

# Analyze
analyzer = FeatureSensitivityAnalyzer(n_repeats=20)
results = analyzer.analyze(model, X_train, y_train, feature_names=names)

# Get top 4 features
top_indices = [idx for idx, _, _ in results.get_top_features(4)]
X_reduced = X_train[:, top_indices]
reduced_names = [names[i] for i in top_indices]

# Retrain on reduced features
model_reduced = RandomForestClassifier(n_estimators=50, random_state=42)
model_reduced.fit(X_reduced, y_train)

# Convert
code = convert(model_reduced, X_reduced, feature_names=reduced_names)
```

### Batch conversion via CLI

```bash
for model in models/*.pkl; do
    name=$(basename "$model" .pkl)
    blackbox2c convert "$model" data/X_train.npy \
        -t arduino -o "output/${name}.h"
done
```
