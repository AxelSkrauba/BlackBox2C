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

The optimizer simplifies the surrogate tree without retraining. Two strategies:

### Branch Pruning (`optimize_rules='medium'` or `'high'`)

Scans for internal nodes where **both children are leaves with the same prediction**.
These are collapsed into a single leaf, reducing the if-else depth.

### Leaf Merging (`optimize_rules='high'`)

Merges sibling leaves when their class probability distributions are highly similar
(cosine similarity > 0.95), further reducing the number of output paths.

---

## Stage 3: Code Generation

The optimized tree is serialized as a recursive if-else function. Each internal node
becomes an `if` condition; each leaf becomes a `return` statement.

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
