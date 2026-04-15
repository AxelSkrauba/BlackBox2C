# Getting Started

## Installation

```bash
pip install blackbox2c
```

Requirements: Python 3.8+, NumPy >= 1.21, scikit-learn >= 1.0.

**Recommended:** use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate          # Windows
pip install blackbox2c
```

For contributors (development install from source):

```bash
git clone https://github.com/AxelSkrauba/BlackBox2C.git
cd BlackBox2C
pip install -e ".[dev]"
```

---

## Classification Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from blackbox2c import convert

# 1. Train your model
iris = load_iris()
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(iris.data, iris.target)

# 2. Convert to C
c_code = convert(
    model,
    iris.data,
    feature_names=list(iris.feature_names),
    class_names=list(iris.target_names),
    max_depth=5,
)
print(c_code)
```

Generated C code:

```c
/*
 * Auto-generated C code by BlackBox2C
 *   - Input features: 4
 *   - Output classes: 3
 */
#include <stdint.h>

#define setosa 0
#define versicolor 1
#define virginica 2

uint8_t predict(float features[4]) {
    if (features[2] <= 2.449999f) {
        return 0;
    } else {
        if (features[3] <= 1.750000f) {
            return 1;
        } else {
            return 2;
        }
    }
}
```

---

## Regression Example

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from blackbox2c import convert

data = load_diabetes()
model = GradientBoostingRegressor(random_state=42)
model.fit(data.data, data.target)

c_code = convert(model, data.data, max_depth=5)
# Returns: float predict(float features[10]) { ... }
```

---

## Exporting to Other Formats

```python
# All formats use the same interface — just change target=
arduino_code  = convert(model, X_train, target='arduino')
cpp_code      = convert(model, X_train, target='cpp')
mp_code       = convert(model, X_train, target='micropython')
c_code        = convert(model, X_train, target='c')  # default
```

---

## Using the Converter Class Directly

```python
from blackbox2c import Converter, ConversionConfig

config = ConversionConfig(
    max_depth=5,
    optimize_rules='high',
    use_fixed_point=True,
    precision=16,
)

converter = Converter(config)
code = converter.convert(
    model, X_train,
    X_test=X_test,
    feature_names=['temp', 'humidity', 'pressure'],
    class_names=['LOW', 'MEDIUM', 'HIGH'],
    target='arduino',
)

# Inspect conversion metrics
metrics = converter.get_metrics()
print(f"Fidelity: {metrics['fidelity']:.3f}")
print(f"Nodes: {metrics['complexity']['n_nodes']}")
print(f"Est. FLASH: {metrics['size_estimate']['flash_bytes']} bytes")
```

---

## Feature Analysis

```python
from blackbox2c.analysis import FeatureSensitivityAnalyzer

analyzer = FeatureSensitivityAnalyzer(n_repeats=10, random_state=42)
results = analyzer.analyze(model, X_train, y_train, feature_names=feature_names)

print(results.summary())
top3 = results.get_top_features(3)
redundant = results.get_redundant_features(threshold=0.001)
```

---

## CLI

```bash
# Convert
blackbox2c convert model.pkl X_train.npy -t arduino -o predict.h

# Analyze
blackbox2c analyze model.pkl X_train.npy --feature-names sl,sw,pl,pw --top-n 3

# Export (direct, no surrogate)
blackbox2c export model.pkl -f cpp -o predictor.hpp
```

Save your model and data:

```python
import pickle, numpy as np
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
np.save('X_train.npy', X_train)
```

---

## Configuration Reference

See [ConversionConfig API](api/config.md) for all parameters.
