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
| `optimize_rules` | `str` | `'medium'` | Rule optimization level: `'low'` (no-op), `'medium'` (pruning), `'high'` (pruning + leaf merging), `'qm'` *(v0.2)* (Quine-McCluskey), `'bdd'` *(v0.2)* (Reduced Ordered BDD), `'auto'` *(v0.2)* (smallest-FLASH of all). |
| `use_fixed_point` | `bool` | `False` | Use integer arithmetic instead of float. Useful for MCUs without FPU. |
| `precision` | `int` | `8` | Bit width for fixed-point scaling: `8`, `16`, or `32`. Ignored if `use_fixed_point=False`. |
| `function_name` | `str` | `'predict'` | Name of the generated C/C++ function. |
| `n_samples` | `int` | `10000` | Number of synthetic samples generated for surrogate training. |
| `feature_threshold` | `int` | `None` | If set, automatically selects the N most important features before conversion. |
| `memory_budget_kb` | `float` | `None` | FLASH budget in KB. Pre-tunes small budgets and warns after conversion if estimated FLASH exceeds it; not a compiled-firmware guarantee. |
| `fidelity_warning_threshold` | `float` or `None` | `0.95` | Warn when surrogate fidelity is lower. Set to `None` to disable this warning. |
| `random_state` | `int` | `42` | Random seed for reproducibility. |
| `include_probabilities` | `bool` | `False` | **Not yet implemented.** Will emit a warning if set to `True`. |
| `qm_max_literals` | `int` | `12` | *(v0.2)* Cap on unique literals before `'qm'` falls back to identity (with `UserWarning`). |
| `bdd_max_literals` | `int` | `24` | *(v0.2)* Cap on unique literals before `'bdd'` falls back to identity (with `UserWarning`). |
| `max_bridge_nodes` | `int` | `4096` | Maximum advanced RuleSet reconstruction nodes before safe fallback to the legacy tree path. |

## Memory Budget Auto-tuning

When `memory_budget_kb` is set, parameters are adjusted automatically for
small budgets. After generation, BlackBox2C compares the estimated FLASH bytes
against `memory_budget_kb * 1024` and emits a `UserWarning` when it is exceeded.
The estimate is heuristic and does not replace a measurement from the target
compiler or MCU.

| Budget | Effect |
|---|---|
| < 1 KB | `max_depth <= 3`, `precision=8`, `use_fixed_point=True` |
| 1-2 KB | `max_depth <= 4`, `precision=8` |
| 2-4 KB | `max_depth <= 6` |

## Advanced optimization caveats (v0.2)

- `'qm'`, `'bdd'`, and `'auto'` are **classification-only**. On regression tasks they
  emit a single `UserWarning` and fall back to `'high'`.
- Legacy levels (`'low'`, `'medium'`, `'high'`) keep their exact v0.1 semantics and
  produce **byte-identical** C code.
- Functional equivalence with the surrogate tree is verified at 100 % by the test
  suite.
- Inputs above both literal caps, or RuleSets that exceed `max_bridge_nodes` during
  reconstruction, emit a `UserWarning` and use the legacy tree path without changing
  predictions.
- See [`benchmarks/results/v0.2.md`](https://github.com/AxelSkrauba/BlackBox2C/blob/main/benchmarks/results/v0.2.md)
  for measured FLASH savings and the [Optimizer (advanced)](optimizer.md) reference
  for direct programmatic access to QM, BDD, and the routing layer.

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
