# BlackBox2C

**Convert scikit-learn models to native embedded code — C, C++, Arduino, MicroPython**

BlackBox2C converts any trained scikit-learn model into a compact, dependency-free if-else
function in your target language — ready to flash on any microcontroller.

---

## Why BlackBox2C?

Most ML deployment tools for embedded systems require a runtime, specific hardware support, or
restrict you to certain model types. BlackBox2C takes a different approach:

- **Any scikit-learn model** → train with whatever works best (Random Forest, SVM, MLP...)
- **Any target platform** → output is pure if-else logic, no runtime needed
- **Exact resource control** → tune depth, precision, and memory budget explicitly

---

## Key Features

| Feature | Description |
|---|---|
| Universal model support | Any scikit-learn estimator with `predict()` |
| Multi-format export | C, C++11, Arduino, MicroPython |
| Classification & regression | Both task types fully supported |
| Feature sensitivity analysis | Identify and remove unimportant features |
| Memory budget control | Auto-tune parameters to fit a target KB |
| Zero runtime dependencies | Generated code is self-contained |

---

## Quick Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from blackbox2c import convert

iris = load_iris()
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(iris.data, iris.target)

c_code = convert(model, iris.data, target='c', max_depth=5)
print(c_code)
```

See [Getting Started](quickstart.md) for the full walkthrough.

---

## Installation

```bash
pip install -e .
```

Requires Python 3.8+, NumPy >= 1.21, scikit-learn >= 1.0.

---

## License

[MIT](https://github.com/AxelSkrauba/BlackBox2C/blob/main/LICENSE) — BlackBox2C Team
