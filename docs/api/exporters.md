# Exporters

Exporters generate platform-specific code from a fitted `DecisionTree`. They are used
internally by `Converter.convert(target=...)` but can also be used directly.

## `create_exporter()` Factory

```python
from blackbox2c.exporters import create_exporter

exporter = create_exporter(format, **kwargs)
```

| `format` value | Exporter class |
|---|---|
| `'cpp'` or `'c++'` | `CppExporter` |
| `'arduino'` | `ArduinoExporter` |
| `'micropython'` or `'python'` | `MicroPythonExporter` |

---

## `CppExporter`

Generates a C++11 class inside an `ml` namespace.

```python
from blackbox2c.exporters import CppExporter

exporter = CppExporter(
    function_name='predict',
    class_name='Predictor',
    use_namespace=True,
    namespace='ml',
    use_fixed_point=False,
    precision=8,
)
code = exporter.generate(tree, feature_names=['sl', 'sw', 'pl', 'pw'])
```

Output structure:
```cpp
namespace ml {
class Predictor {
public:
    static uint8_t predict(float features[4]) { ... }
};
} // namespace ml
```

---

## `ArduinoExporter`

Generates an Arduino `.h` file with optional `PROGMEM` constants.

```python
from blackbox2c.exporters import ArduinoExporter

exporter = ArduinoExporter(
    function_name='predict',
    use_progmem=True,
    use_fixed_point=False,
    precision=8,
)
code = exporter.generate(tree, feature_names=['temp', 'humidity'])
```

Include in your sketch:
```cpp
#include "predict.h"

void loop() {
    float features[2] = { readTemp(), readHumidity() };
    uint8_t result = predict(features);
}
```

---

## `MicroPythonExporter`

Generates a Python module for MicroPython-enabled boards (ESP32, Pico, etc.).

```python
from blackbox2c.exporters import MicroPythonExporter

exporter = MicroPythonExporter(
    function_name='predict',
    class_name='Predictor',
    use_const=True,
)
code = exporter.generate(tree, feature_names=['x', 'y'])
```

Usage on device:
```python
from predictor import Predictor
result = Predictor.predict([0.5, 1.2])
```

---

## Direct Usage (after Converter)

```python
converter = Converter(config)
converter.convert(model, X_train)  # runs pipeline, builds surrogate_tree_

exporter = create_exporter('arduino')
code = exporter.generate(
    converter.surrogate_tree_,
    feature_names=converter.feature_names_,
    class_names=converter.class_names_,
)
```
