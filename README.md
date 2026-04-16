# BlackBox2C

**Convert scikit-learn models to native embedded code — C, C++, Arduino, MicroPython**

[![Tests](https://github.com/AxelSkrauba/BlackBox2C/actions/workflows/ci.yml/badge.svg)](https://github.com/AxelSkrauba/BlackBox2C/actions)
[![PyPI](https://img.shields.io/pypi/v/blackbox2c)](https://pypi.org/project/blackbox2c/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

BlackBox2C converts any trained scikit-learn model into a minimal if-else decision tree in your
target language. The generated code has **zero runtime dependencies**, runs on any microcontroller
with a C compiler, and fits in a few hundred bytes of FLASH.

---

## How It Works

1. **Surrogate extraction** — A lightweight `DecisionTree` is trained to mimic any black-box model
   (Random Forest, SVM, MLP, etc.) by generating synthetic boundary samples and labeling them with
   the original model's predictions.
2. **Rule optimization** — Redundant branches are pruned and similar leaves are merged to minimize
   code size.
3. **Code generation** — The optimized tree is serialized as a pure if-else function in the target
   language.

---

## Supported Models and Targets

| Input models | Output formats |
|---|---|
| Any scikit-learn estimator with `predict()` | Pure C (C99) |
| Decision Tree, Random Forest, SVM, MLP... | C++11 (class + namespace) |
| Classification and Regression tasks | Arduino (`.h` with PROGMEM) |
| | MicroPython (`.py` module) |

---

## Installation

```bash
pip install blackbox2c
```

Requirements: Python 3.8+, NumPy >= 1.21, scikit-learn >= 1.0.

> **Tip:** Use a virtual environment to keep your project isolated:

```bash
# python -m venv .venv && source .venv/bin/activate  # Linux/macOS
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install blackbox2c
```

For development (from source):

```bash
git clone https://github.com/AxelSkrauba/BlackBox2C.git
cd BlackBox2C
pip install -e ".[dev]"
```

---

## Quick Start

### Classification

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from blackbox2c import convert

iris = load_iris()
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(iris.data, iris.target)

# Convert to C (default target)
c_code = convert(
    model,
    iris.data,
    feature_names=list(iris.feature_names),
    class_names=list(iris.target_names),
    max_depth=5,
)
print(c_code)
```

Generated output:

```c
/*
 * Auto-generated C code by BlackBox2C
 *   - Input features: 4
 *   - Output classes: 3
 *   - Precision: 8-bit
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

### Export to Other Formats

```python
# Arduino .ino file
arduino_code = convert(model, iris.data, target='arduino')

# C++ class
cpp_code = convert(model, iris.data, target='cpp')

# MicroPython module
mp_code = convert(model, iris.data, target='micropython')
```

### Regression

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from blackbox2c import convert

data = load_diabetes()
model = GradientBoostingRegressor(random_state=42)
model.fit(data.data, data.target)

c_code = convert(model, data.data, max_depth=5)
# Generates: float predict(float features[10]) { ... }
```

### Feature Analysis

```python
from blackbox2c.analysis import FeatureSensitivityAnalyzer

analyzer = FeatureSensitivityAnalyzer(n_repeats=10, random_state=42)
results = analyzer.analyze(model, X_train, y_train, feature_names=feature_names)
print(results.summary())

# Get top 3 most important features by index
top3 = results.get_top_features(3)
```

---

## Configuration

```python
from blackbox2c import Converter, ConversionConfig

config = ConversionConfig(
    max_depth=5,             # Surrogate tree depth (1-10, default 5)
    optimize_rules='medium', # 'low' | 'medium' | 'high'
    use_fixed_point=False,   # Use integer arithmetic instead of float
    precision=8,             # Bit width for fixed-point: 8 | 16 | 32
    function_name='predict', # Name of the generated function
    n_samples=10000,         # Synthetic samples for surrogate training
    feature_threshold=None,  # Auto-select N most important features
    memory_budget_kb=None,   # Auto-tune params to fit a KB budget
)

converter = Converter(config)
code = converter.convert(model, X_train, target='arduino')
metrics = converter.get_metrics()
# {'fidelity': 0.97, 'complexity': {...}, 'size_estimate': {...}}
```

---

## CLI

```bash
# Convert a pickled model to C
blackbox2c convert model.pkl X_train.npy -o output.c

# Export to Arduino
blackbox2c convert model.pkl X_train.npy -t arduino -o predict.h

# Analyze feature importance
blackbox2c analyze model.pkl X_train.npy --top-n 5

# Export a decision tree directly (no surrogate extraction)
blackbox2c export model.pkl -f cpp -o predictor.hpp

# Help
blackbox2c --help
blackbox2c convert --help
```

---

## Benchmarks

```bash
python benchmarks/benchmark_classic_datasets.py --output results.md
```

Covers Iris, Wine, Diabetes, and California Housing with Decision Trees, Random Forests, SVMs,
and Neural Networks. Metrics: fidelity, estimated FLASH size, tree depth, conversion time.

> **Note**: Code size figures are estimates from BlackBox2C's built-in size estimator,
> not measurements on real hardware.

---

## Project Structure

```
blackbox2c/
├── blackbox2c/
│   ├── __init__.py      # Public API: convert(), Converter, ConversionConfig
│   ├── converter.py     # Main orchestration pipeline
│   ├── config.py        # ConversionConfig dataclass
│   ├── surrogate.py     # Surrogate tree extraction
│   ├── codegen.py       # C code generation
│   ├── optimizer.py     # Rule pruning and merging
│   ├── exporters.py     # C++, Arduino, MicroPython exporters
│   ├── analysis.py      # Feature sensitivity analysis
│   └── cli.py           # Command-line interface
├── tests/               # 182 tests, >91% coverage
├── notebooks/           # Jupyter notebook examples (runnable on Colab)
├── benchmarks/          # Classic dataset benchmarks
├── examples/            # Script-based end-to-end examples
└── docs/                # MkDocs documentation source
```

---

## Comparison with Alternatives

| Feature | BlackBox2C | emlearn | MicroMLGen | TFLite Micro |
|---|---|---|---|---|
| Any sklearn model | ✅ | ⚠️ Trees only | ⚠️ Trees only | ❌ TF only |
| Pure if-else output | ✅ | ✅ | ✅ | ❌ |
| C++ / Arduino / MicroPython | ✅ | ⚠️ Partial | ❌ | ⚠️ Partial |
| Feature selection built-in | ✅ | ❌ | ❌ | ❌ |
| Memory budget control | ✅ | ❌ | ❌ | ⚠️ |
| Zero runtime dependencies | ✅ | ✅ | ✅ | ❌ |

---

## Roadmap (v0.2)

- Quine-McCluskey and BDD rule optimization
- Hardware-validated benchmarks on real MCUs
- Quantization-aware training integration

---

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Issues and PRs welcome at [github.com/AxelSkrauba/BlackBox2C](https://github.com/AxelSkrauba/BlackBox2C).
