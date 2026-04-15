# CLI Reference

The `blackbox2c` command is installed as a console script when the package is installed.

```
blackbox2c <command> [options]
```

---

## `convert`

Convert a pickled model to embedded code.

```bash
blackbox2c convert MODEL DATA [options]
```

**Positional arguments**:

| Argument | Description |
|---|---|
| `MODEL` | Path to pickled sklearn model (`.pkl`) |
| `DATA` | Path to training data (`.npy` or `.npz`) |

**Options**:

| Option | Default | Description |
|---|---|---|
| `-t`, `--target` | `c` | Output format: `c`, `cpp`, `arduino`, `micropython` |
| `-o`, `--output` | stdout | Output file path |
| `--test-data PATH` | — | Test data for fidelity evaluation |
| `--feature-names NAMES` | — | Comma-separated feature names |
| `--class-names NAMES` | — | Comma-separated class names |
| `--max-depth N` | `5` | Max surrogate tree depth |
| `--optimize LEVEL` | `medium` | Rule optimization: `low`, `medium`, `high` |
| `--fixed-point` | off | Use fixed-point arithmetic |
| `--precision N` | `8` | Fixed-point bit width: `8`, `16`, `32` |
| `--function-name NAME` | `predict` | Name of the generated function |
| `--n-samples N` | `10000` | Synthetic samples for surrogate training |

**Examples**:

```bash
# Basic conversion to C
blackbox2c convert model.pkl X_train.npy -o predict.c

# Arduino with named features and classes
blackbox2c convert model.pkl X_train.npy \
    -t arduino \
    --feature-names temp,humidity,pressure \
    --class-names LOW,MED,HIGH \
    -o predict.h

# MicroPython with high optimization
blackbox2c convert model.pkl X_train.npy \
    -t micropython \
    --optimize high \
    --max-depth 4 \
    -o model.py
```

---

## `analyze`

Run feature sensitivity analysis on a model.

```bash
blackbox2c analyze MODEL DATA [options]
```

**Options**:

| Option | Default | Description |
|---|---|---|
| `--feature-names NAMES` | — | Comma-separated feature names |
| `--top-n N` | — | Print top N most important features |
| `--n-repeats N` | `10` | Permutation repeats |
| `-o`, `--output PATH` | stdout | Write report to file |

**Example**:

```bash
blackbox2c analyze model.pkl X_train.npy \
    --feature-names sl,sw,pl,pw \
    --top-n 3
```

---

## `export`

Export a decision tree model directly (no surrogate extraction).

Use this when your model **is already a decision tree**. For other model types, use `convert`.

```bash
blackbox2c export MODEL [options]
```

**Options**:

| Option | Default | Description |
|---|---|---|
| `-f`, `--format` | `cpp` | Output format: `cpp`, `arduino`, `micropython` |
| `-o`, `--output PATH` | stdout | Output file path |
| `--feature-names NAMES` | — | Comma-separated feature names |
| `--class-names NAMES` | — | Comma-separated class names |
| `--function-name NAME` | `predict` | Name of the generated function |

**Example**:

```bash
blackbox2c export dt_model.pkl -f arduino -o predict.h
```

---

## Preparing Model and Data Files

```python
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=42
)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
```
