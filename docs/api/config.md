# `ConversionConfig`

```python
@dataclass
class blackbox2c.ConversionConfig
```

Dataclass holding all conversion parameters. Pass to `Converter(config)` or as
`**kwargs` to `convert()`.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_depth` | `int` | `5` | Max depth of the surrogate decision tree. Range: 1-10. Higher = more accurate but larger code. |
| `optimize_rules` | `str` | `'medium'` | Rule optimization level: `'low'` (light pruning), `'medium'` (pruning), `'high'` (pruning + leaf merging). |
| `use_fixed_point` | `bool` | `False` | Use integer arithmetic instead of float. Useful for MCUs without FPU. |
| `precision` | `int` | `8` | Bit width for fixed-point scaling: `8`, `16`, or `32`. Ignored if `use_fixed_point=False`. |
| `function_name` | `str` | `'predict'` | Name of the generated C/C++ function. |
| `n_samples` | `int` | `10000` | Number of synthetic samples generated for surrogate training. |
| `feature_threshold` | `int` | `None` | If set, automatically selects the N most important features before conversion. |
| `memory_budget_kb` | `float` | `None` | Target memory budget in KB. Auto-adjusts `max_depth`, `precision`, and `use_fixed_point`. |
| `random_state` | `int` | `42` | Random seed for reproducibility. |
| `include_probabilities` | `bool` | `False` | **Not yet implemented.** Will emit a warning if set to `True`. |

## Memory Budget Auto-tuning

When `memory_budget_kb` is set, parameters are adjusted automatically:

| Budget | Effect |
|---|---|
| < 1 KB | `max_depth <= 3`, `precision=8`, `use_fixed_point=True` |
| 1-2 KB | `max_depth <= 4`, `precision=8` |
| 2-4 KB | `max_depth <= 6` |

## Example

```python
from blackbox2c import ConversionConfig

# For a very constrained MCU (ATmega328P, 2KB RAM)
config = ConversionConfig(
    max_depth=3,
    use_fixed_point=True,
    precision=8,
    optimize_rules='high',
)

# Auto-tune for 1KB budget
config = ConversionConfig(memory_budget_kb=1.0)

# Feature selection: keep only 3 most important features
config = ConversionConfig(feature_threshold=3, max_depth=4)
```
