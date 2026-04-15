# Comparison with Alternatives

This page compares BlackBox2C with other tools for deploying ML models to embedded systems.
All comparisons are based on publicly available documentation and source code.

---

## Feature Matrix

| Feature | BlackBox2C | emlearn | MicroMLGen | TFLite Micro | STM32Cube.AI |
|---|---|---|---|---|---|
| Any sklearn model | Yes | Trees only | Trees only | TF/Keras only | TF/Keras only |
| SVM, MLP support | Yes (surrogate) | No | No | Limited | Limited |
| Pure if-else output | Yes | No (arrays) | Yes | No (runtime) | No (runtime) |
| C output | Yes | Yes | Yes | Yes | Yes |
| C++ output | Yes | No | No | No | No |
| Arduino output | Yes | Partial | No | No | No |
| MicroPython output | Yes | No | No | No | No |
| Zero runtime deps | Yes | Yes | Yes | No | No |
| Hardware-agnostic | Yes | Yes | Yes | Partial | No (ST only) |
| Feature selection | Yes | No | No | No | No |
| Memory budget control | Yes | No | No | Partial | Partial |
| Regression support | Yes | Partial | No | Yes | Yes |
| Open source | Yes | Yes | Yes | Yes | No |

---

## emlearn

[emlearn](https://emlearn.org) is a mature library focused on efficient C implementations
of decision trees and ensemble models.

**emlearn strengths**:
- Very efficient array-based C for tree ensembles
- Supports more tree types natively
- More mature and battle-tested

**BlackBox2C advantages over emlearn**:
- Converts **any** sklearn model (SVM, MLP, etc.) via surrogate extraction
- Generates readable if-else code instead of arrays
- Multiple output formats (C++, Arduino, MicroPython)
- Built-in feature sensitivity analysis

---

## MicroMLGen

[MicroMLGen](https://github.com/eloquentarduino/micromlgen) generates Arduino-ready C code
from sklearn decision trees and SVMs.

**MicroMLGen strengths**:
- Good Arduino/IDE integration
- Simple interface for tree models

**BlackBox2C advantages over MicroMLGen**:
- Supports all sklearn models (not just trees/SVM)
- Multiple output formats beyond Arduino
- Memory budget control
- Feature analysis and selection

---

## TensorFlow Lite Micro

[TFLite Micro](https://www.tensorflow.org/lite/microcontrollers) is Google's solution for
running TensorFlow models on microcontrollers.

**TFLite Micro strengths**:
- Supports complex neural networks
- Optimized for ARM Cortex-M with CMSIS-NN
- Google-backed, large ecosystem

**BlackBox2C advantages over TFLite Micro**:
- No runtime library required (saves 20-100+ KB)
- Works with any C compiler, any MCU
- Accepts sklearn models directly
- Much simpler integration

---

## When to Use Each

| Use case | Recommended tool |
|---|---|
| sklearn model, any MCU, minimal footprint | **BlackBox2C** |
| Large tree ensemble, performance-critical | emlearn |
| Quick Arduino sketch from decision tree | MicroMLGen |
| Complex neural network, ARM Cortex-M | TFLite Micro |
| ST hardware, Keras model | STM32Cube.AI |
