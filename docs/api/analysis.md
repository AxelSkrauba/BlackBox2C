# `FeatureSensitivityAnalyzer`

```python
class blackbox2c.analysis.FeatureSensitivityAnalyzer(n_repeats=10, random_state=42)
```

Measures feature importance via permutation: shuffles each feature's values and measures
how much the model's score drops. Features that cause large drops are important.

## Constructor

| Parameter | Default | Description |
|---|---|---|
| `n_repeats` | `10` | Number of permutation repetitions per feature |
| `random_state` | `42` | Random seed |

## `analyze()`

```python
results = analyzer.analyze(
    model,
    X,
    y,
    feature_names=None,
    scoring=None
)
```

**Parameters**:

| Parameter | Description |
|---|---|
| `model` | Fitted sklearn estimator |
| `X` | Input data (n_samples, n_features) |
| `y` | Target values |
| `feature_names` | Optional list of names |
| `scoring` | Scoring function (default: auto-detect accuracy or R2) |

**Returns**: `SensitivityResults`

---

## `SensitivityResults`

### Methods

```python
results.summary()
# Returns a formatted string with importances for all features

results.get_top_features(n)
# Returns list of (index, name, importance) for the top-N features

results.get_optimal_subset(threshold=0.01, min_features=1)
# Returns list of feature indices above the importance threshold

results.get_redundant_features(threshold=0.001)
# Returns list of feature indices with importance near zero
```

### Plotting

```python
fig, ax = results.plot(figsize=(10, 6), save_path=None)
# Returns (fig, ax). Requires matplotlib.
# Displays a horizontal bar chart with permutation importances and error bars.
```

---

## Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from blackbox2c.analysis import FeatureSensitivityAnalyzer

iris = load_iris()
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(iris.data, iris.target)

analyzer = FeatureSensitivityAnalyzer(n_repeats=20, random_state=42)
results = analyzer.analyze(
    model, iris.data, iris.target,
    feature_names=list(iris.feature_names)
)

print(results.summary())
top2 = results.get_top_features(2)
print("Top 2 features:", top2)
```
