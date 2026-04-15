# `convert()` — Convenience Function

```python
blackbox2c.convert(
    model,
    X_train,
    feature_names=None,
    class_names=None,
    X_test=None,
    target='c',
    config=None,
    **config_kwargs
)
```

High-level convenience function. Creates a `Converter` with the given configuration and
calls `converter.convert()` in one step.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `BaseEstimator` | required | Trained scikit-learn model |
| `X_train` | `np.ndarray` | required | Training data (n_samples, n_features) |
| `feature_names` | `list[str]` | `None` | Feature names for code readability |
| `class_names` | `list[str]` | `None` | Class names (classification only) |
| `X_test` | `np.ndarray` | `None` | Test data for fidelity evaluation |
| `target` | `str` | `'c'` | Output format: `'c'`, `'cpp'`, `'arduino'`, `'micropython'` |
| `config` | `ConversionConfig` | `None` | Config object (if set, `**config_kwargs` ignored) |
| `**config_kwargs` | | | Passed to `ConversionConfig(...)` when `config=None` |

## Returns

`str` — Generated code in the requested format.

## Examples

```python
from blackbox2c import convert

# Minimal usage
c_code = convert(model, X_train)

# With names and test data
c_code = convert(
    model, X_train,
    feature_names=['temp', 'humidity'],
    class_names=['LOW', 'HIGH'],
    X_test=X_test,
    max_depth=4,
)

# Export to Arduino
arduino = convert(model, X_train, target='arduino')

# Use a config object
from blackbox2c import ConversionConfig
config = ConversionConfig(max_depth=7, optimize_rules='high')
c_code = convert(model, X_train, config=config)
```
