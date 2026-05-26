# Examples

## Interactive Notebooks

For hands-on, fully-executed examples with rendered outputs, see the [Notebooks](notebooks/01_quickstart.ipynb) section:

| Notebook | Description |
|---|---|
| [Quickstart](notebooks/01_quickstart.ipynb) | Convert a model to C in 5 minutes |
| [Classification](notebooks/02_classification_iris.ipynb) | DT / RF / SVM / MLP comparison |
| [Regression IoT](notebooks/03_regression_temperature.ipynb) | Fixed-point, depth trade-offs |
| [Feature Analysis](notebooks/04_feature_analysis.ipynb) | Sensor reduction, BOM impact |
| [Multi-Format Export](notebooks/05_multi_format_export.ipynb) | C, C++, Arduino, MicroPython |
| [End-to-End IoT (ADL)](notebooks/06_end_to_end_iot.ipynb) | Full pipeline on a real gas-sensor dataset |
| [Advanced Optimization (v0.2)](notebooks/07_advanced_optimization.ipynb) | Quine-McCluskey, BDD, and `'auto'` compared on Iris+RandomForest |

---

## Script Examples

All script examples are in the [`examples/`](https://github.com/AxelSkrauba/BlackBox2C/tree/main/examples) directory.

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

### Advanced rule optimization (v0.2)

For classification models, the `'auto'` level evaluates Quine-McCluskey, BDD, and the
no-op baseline, and returns the variant with the smallest estimated FLASH footprint:

```python
from blackbox2c import convert, ConversionConfig

config = ConversionConfig(
    max_depth=5,
    optimize_rules='auto',     # try qm, bdd, no-op; pick smallest FLASH
    qm_max_literals=12,        # cap before QM falls back to identity
    bdd_max_literals=20,       # cap before BDD falls back to identity
)

c_code = convert(model, X_train, config=config)
```

Typical savings on Iris+RandomForest: **−47 % FLASH** vs `'medium'`. See
[`benchmarks/results/v0.2.md`](https://github.com/AxelSkrauba/BlackBox2C/blob/main/benchmarks/results/v0.2.md)
and the [Advanced Optimization notebook](notebooks/07_advanced_optimization.ipynb).

!!! note
    Advanced levels (`'qm'`, `'bdd'`, `'auto'`) are classification-only.
    On regression tasks they emit a single `UserWarning` and fall back to `'high'`.

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
from model import Predictor
pred = Predictor()
result = pred.predict([0.5, 1.2, 3.1, 0.8])
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
