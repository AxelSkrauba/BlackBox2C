# `Converter`

```python
class blackbox2c.Converter(config=None)
```

Main class that orchestrates the full conversion pipeline. Use this class when you need
access to conversion metrics or want to reuse the surrogate tree.

## Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `ConversionConfig` | `None` | Conversion configuration. Uses defaults if `None`. |

## Methods

### `convert()`

```python
converter.convert(
    model, X_train,
    feature_names=None,
    class_names=None,
    X_test=None,
    target='c'
)
```

Run the full pipeline and return generated code.

**Parameters**: same as `convert()` convenience function, minus `config` and `**config_kwargs`.

**Returns**: `str`

### `get_metrics()`

```python
metrics = converter.get_metrics()
```

Returns a `dict` with conversion metrics:

```python
{
    'fidelity': 0.974,              # agreement with original model
    'complexity': {
        'n_nodes': 45,
        'n_leaves': 23,
        'n_internal_nodes': 22,
        'max_depth': 5,
        'avg_path_length': 4.2,
    },
    'size_estimate': {
        'flash_bytes': 382,         # estimated FLASH usage
        'ram_bytes': 96,            # estimated RAM usage
    }
}
```

## Attributes (after `convert()`)

| Attribute | Description |
|---|---|
| `surrogate_tree_` | The fitted `DecisionTree` (sklearn object) |
| `feature_names_` | Feature names used in code generation |
| `class_names_` | Class names (classification only) |
| `metrics_` | Same as `get_metrics()` |

## Example

```python
from blackbox2c import Converter, ConversionConfig

config = ConversionConfig(max_depth=5, optimize_rules='high')
converter = Converter(config)

code = converter.convert(
    model, X_train,
    X_test=X_test,
    feature_names=['sl', 'sw', 'pl', 'pw'],
    class_names=['setosa', 'versicolor', 'virginica'],
    target='cpp',
)

m = converter.get_metrics()
print(f"Fidelity: {m['fidelity']:.3f}")
print(f"Nodes: {m['complexity']['n_nodes']}")
```
