# Algorithm

## Overview

BlackBox2C uses a **black-box surrogate approach** to convert any ML model into executable
decision rules. The pipeline has four stages:

```
Original model  ->  [1] Surrogate extraction
                ->  [2] Rule optimization
                ->  [3] Code generation
                ->  [4] Size estimation
```

---

## Stage 1: Surrogate Tree Extraction

### Problem

Direct inspection of complex models (Random Forest, SVM, Neural Networks) is either impossible
or produces code that is too large for embedded systems. A Random Forest with 100 trees and
depth 10 would generate millions of nodes.

### Solution: Decision Boundary Synthesis

BlackBox2C trains a shallow `DecisionTreeClassifier` or `DecisionTreeRegressor` to **mimic**
the original model. The process:

1. **Synthetic sample generation** - generates `n_samples` points covering the input space:
   - 40% uniform random samples within feature ranges
   - 40% perturbations around training data points (boundary-focused)
   - 20% original training samples

2. **Labeling** - the original model labels each synthetic sample with `model.predict(X_synthetic)`

3. **Surrogate training** - a `DecisionTree` with `max_depth` constraint is fitted on the
   labeled synthetic data

4. **Fidelity measurement** - agreement between original and surrogate predictions:
   - Classification: accuracy (exact match rate)
   - Regression: R2 score

### Fidelity vs. Size Trade-off

| `max_depth` | Approx. nodes | Typical fidelity |
|---|---|---|
| 3 | 7-15 | 0.80-0.92 |
| 5 | 15-63 | 0.90-0.98 |
| 7 | 30-127 | 0.95-0.99 |

A deeper surrogate captures more decision boundaries at the cost of larger generated code.

---

## Stage 2: Rule Optimization

The optimizer simplifies the surrogate tree without retraining. Six levels are available
via `optimize_rules`:

### Branch Pruning (`'medium'`, `'high'`)

Scans for internal nodes where **both children are leaves with the same prediction**.
These are collapsed into a single leaf, reducing the if-else depth.

### Leaf Merging (`'high'`)

Merges sibling leaves when their class probability distributions are highly similar
(cosine similarity > 0.95), further reducing the number of output paths.

### Literal Simplification *(v0.2)*

Applied automatically by every advanced level. Within each conjunction, multiple
literals on the same feature are collapsed into a single `(lo, hi]` interval; rules
with empty intervals are detected as unsatisfiable and dropped.
Implemented in `Conjunction.simplify` / `RuleSet.simplify`.

### Quine-McCluskey (`'qm'`, v0.2)

Multi-valued Quine-McCluskey boolean minimisation lifted to continuous splits. The
surrogate tree is converted into a `RuleSet`, each unique threshold becomes a
multi-valued literal, and the QM tabular method merges minterms class-by-class.
Gated by `qm_max_literals` (default 12) and an internal `max_minterms=4096` cap;
over-cap inputs return unchanged with a `UserWarning`.

### Reduced Ordered BDD (`'bdd'`, v0.2)

One ROBDD per output class is built with frequency-ordered variables and a unique
table. Re-emission enumerates true-paths back into a `RuleSet`. Capped by
`bdd_max_literals` (default 20).

### Auto Routing (`'auto'`, v0.2)

Runs every applicable optimiser plus the no-op baseline, estimates FLASH cost via
the bridge codegen's tree-shape model, and returns the smallest. Never regresses
below the `'medium'` baseline on the in-tree benchmark.

> **NOTE**:
    Advanced levels (`'qm'`, `'bdd'`, `'auto'`) are classification-only. On
    regression tasks they emit a single `UserWarning` and fall back to `'high'`.
    Legacy levels (`'low'`, `'medium'`, `'high'`) produce byte-identical output to v0.1.

---

## Stage 3: Code Generation

The optimized tree is serialized as a recursive if-else function. Each internal node
becomes an `if` condition; each leaf becomes a `return` statement.

### Hierarchical bridge codegen *(v0.2)*

When an advanced level is requested, the optimised `RuleSet` is fed to a hierarchical
rebuilder (`RuleSetCodeGenerator`) that reconstructs a decision tree using a
split-on-most-frequent-literal heuristic. This preserves prefix sharing and emits the
same nested if/else surface as the legacy generator, so downstream tooling sees no
difference.

### Feature name injection

```c
// With feature_names=['sepal_length', ...]
if (sepal_length <= 5.449999f) { ...

// Without feature_names (default)
if (features[0] <= 5.449999f) { ...
```

### Fixed-point arithmetic

When `use_fixed_point=True`, thresholds are scaled by `2^(precision-1)`:

```c
// float (default):  if (features[0] <= 5.449999f)
// fixed 8-bit:      if (features[0] <= 697)       // 5.45 * 2^7 = 697
// fixed 16-bit:     if (features[0] <= 178586)     // 5.45 * 2^15
```

This avoids floating-point hardware requirements on MCUs like AVR (Arduino Uno).

---

## Stage 4: Size Estimation

BlackBox2C estimates FLASH and RAM usage heuristically:

- **FLASH**: `54 + (n_leaves * 8) + (n_internal_nodes * 16)` bytes (approximate)
- **RAM**: `4 + n_leaves * 4` bytes (approximate)

!!! warning
    These are rough estimates. Actual size depends on the compiler, optimization flags,
    and target architecture. Always verify with your toolchain.

---

## Differentiators vs. Existing Tools

### vs. emlearn

emlearn generates C arrays with lookup logic. BlackBox2C generates pure if-else, which is
more readable, compiler-friendly, and avoids array-access overhead. BlackBox2C also supports
non-tree models (SVM, MLP) via surrogate extraction.

### vs. MicroMLGen

MicroMLGen is limited to decision trees and outputs a single C file. BlackBox2C supports
any sklearn model, multiple export formats, and adds feature selection and memory budgeting.

### vs. TensorFlow Lite Micro

TFLite Micro requires a runtime library (~20-100 KB), a specific model format, and
quantization tooling. BlackBox2C generates zero-dependency code that compiles with any
C compiler and fits in under 1 KB for simple models.

### vs. STM32Cube.AI

STM32Cube.AI is vendor-locked to STMicroelectronics hardware. BlackBox2C is hardware-agnostic
and works on any MCU with a C compiler: AVR, ARM Cortex-M, ESP32, RISC-V, etc.
