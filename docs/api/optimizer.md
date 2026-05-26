# Optimizer (Advanced, v0.2)

The `blackbox2c.optimizer` package exposes the IR and the advanced rule-optimisation
pipeline introduced in v0.2. Most users never need to import from it directly — the
`'qm'`, `'bdd'`, and `'auto'` values of [`ConversionConfig.optimize_rules`](config.md)
already wire everything together. This page is for users who want to inspect or
extend the pipeline programmatically.

---

## Overview

```
sklearn tree  ──►  RuleSet (IR)  ──►  Optimizer  ──►  RuleSet  ──►  C code
                  (extraction)      (qm/bdd/auto)    (simplify)   (bridge codegen)
```

All optimisers operate on the same immutable IR (`RuleSet`) and produce another
`RuleSet`, so they compose freely.

---

## `blackbox2c.optimizer.ir`

Three frozen dataclasses model the rules.

### `Literal`

```python
Literal(feature: int, threshold: float, op: str)   # op in {'<=', '>'}
```

A single split predicate `features[feature] <= threshold` (or `>`). Implements
`evaluate(x)` against a single sample.

### `Conjunction`

```python
Conjunction(literals: tuple[Literal, ...], output: int | float)
```

Logical AND of literals leading to a class (classification) or value (regression).

- `Conjunction.evaluate(x) -> bool`
- `Conjunction.simplify() -> Conjunction | None` — collapse multiple literals on the
  same feature into a single `(lo, hi]` interval; returns `None` when the conjunction
  is unsatisfiable.

### `RuleSet`

```python
RuleSet(rules: tuple[Conjunction, ...], n_features: int, n_classes: int | None)
```

A complete classifier/regressor as a disjunction of conjunctions.

- `RuleSet.predict(X) -> np.ndarray`
- `RuleSet.complexity() -> dict` — `n_rules`, `n_literals`, `avg_literals_per_rule`.
- `RuleSet.unique_literals() -> set[tuple[int, float, str]]`
- `RuleSet.simplify() -> RuleSet` — apply `Conjunction.simplify` to every rule and
  drop unsatisfiable ones.

---

## `blackbox2c.optimizer.extraction`

```python
from blackbox2c.optimizer.extraction import from_sklearn_tree

ruleset = from_sklearn_tree(sklearn_tree, n_features)
```

Lossless conversion of any fitted scikit-learn `DecisionTreeClassifier` /
`DecisionTreeRegressor` into a `RuleSet`. The reverse direction (RuleSet → tree) is
performed implicitly by the bridge codegen.

---

## `blackbox2c.optimizer.qm.QMOptimizer`

Multi-valued Quine-McCluskey minimisation lifted to continuous splits.

```python
from blackbox2c.optimizer.qm import QMOptimizer

opt = QMOptimizer(
    max_literals=12,         # unique (feature, threshold) pairs cap
    max_minterms=4096,       # interval-product cap
    petrick_threshold=6,     # exact cover for ≤ this many prime implicants
)
optimized = opt.minimize(ruleset)
print(opt.last_diagnostics_)
```

- **Classification only.** A regression `RuleSet` is returned unchanged with a
  `UserWarning`.
- Over-cap inputs are returned unchanged with a `UserWarning`.
- The output is functionally equivalent to the input on every sample of the input
  domain (verified by Hypothesis-based property tests).

---

## `blackbox2c.optimizer.bdd.BDDOptimizer`

One Reduced Ordered BDD per output class, with frequency-ordered variables and a
unique table.

```python
from blackbox2c.optimizer.bdd import BDDOptimizer

opt = BDDOptimizer(
    max_literals=24,         # unique-literal cap
    max_bdd_nodes=200_000,   # soft ceiling on BDD size
)
optimized = opt.minimize(ruleset)
```

Same regression / over-cap semantics as `QMOptimizer`.

---

## `blackbox2c.optimizer.routing`

The routing layer is what `Converter` calls under the hood for advanced levels.

```python
from blackbox2c.optimizer.routing import (
    optimize_ruleset,
    is_advanced_level,
    VALID_LEVELS,         # ('low', 'medium', 'high', 'qm', 'bdd', 'auto')
)

assert is_advanced_level('auto')        # True
assert not is_advanced_level('high')    # False

best = optimize_ruleset(
    ruleset,
    level='auto',
    qm_max_literals=12,
    bdd_max_literals=20,
)
```

`'auto'` runs every applicable optimiser plus the no-op baseline, estimates FLASH
cost via the bridge codegen's tree-shape model, and returns the smallest result —
so it never regresses below the unoptimised input.

---

## End-to-end example

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from blackbox2c import Converter, ConversionConfig
from blackbox2c.optimizer.extraction import from_sklearn_tree
from blackbox2c.optimizer.routing import optimize_ruleset

# Train + extract surrogate via the standard pipeline
iris = load_iris()
model = RandomForestClassifier(n_estimators=50, random_state=42).fit(
    iris.data, iris.target
)

config = ConversionConfig(max_depth=5, optimize_rules='auto')
converter = Converter(config)
c_code = converter.convert(model, iris.data, target='c')

# The optimised RuleSet is exposed for downstream inspection
rs = converter.optimized_ruleset_
print(rs.complexity())

# You can also drive the optimiser yourself on any RuleSet
hand_built = from_sklearn_tree(converter.surrogate_tree_.tree_, n_features=4)
smaller = optimize_ruleset(hand_built, level='qm')
```

---

## See also

- [Advanced Optimization tutorial notebook](../notebooks/07_advanced_optimization.ipynb)
- [Benchmark results (v0.2)](https://github.com/AxelSkrauba/BlackBox2C/blob/main/benchmarks/results/v0.2.md)
- [`ConversionConfig`](config.md) — high-level entry point
- [Algorithm](../algorithm.md) — Stage 2 covers the maths behind QM, BDD and `'auto'`
