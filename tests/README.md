# BlackBox2C - Test Suite

**326 tests · 93% coverage · 100% passing** *(reference: v0.2.3)*

[![Tests](https://github.com/AxelSkrauba/BlackBox2C/actions/workflows/ci.yml/badge.svg)](https://github.com/AxelSkrauba/BlackBox2C/actions)

---

## Test Suite Layout

```
tests/
├── __init__.py                    # Test package initialization
├── test_advanced_fallbacks.py     # Safe fallbacks for advanced optimizers (6 tests)
├── test_analysis.py               # Feature sensitivity analysis (20 tests)
├── test_cli.py                    # Command-line interface (19 tests)
├── test_codegen.py                # C code generation (25 tests)
├── test_config.py                 # ConversionConfig validation (9 tests)
├── test_conversion_warnings.py    # Fidelity / FLASH budget warnings (5 tests)
├── test_converter.py              # Converter end-to-end integration (14 tests)
├── test_exporters.py              # C++, Arduino, MicroPython exporters (40 tests)
├── test_optimizer.py              # Legacy rule optimization (9 tests)
├── test_prune_negative_index_bug.py  # Regression suite: features[-2] prune bug (18 tests)
├── test_regression.py             # Regression task conversion (16 tests)
├── test_reproducibility.py        # Reproducibility across runs (6 tests)
├── test_surrogate.py              # Surrogate tree extraction (7 tests)
├── test_target_param.py           # target= parameter behavior (20 tests)
├── optimizer/                     # Advanced optimization pipeline (v0.2)
│   ├── test_bdd.py                # Reduced Ordered BDD optimizer (17 tests)
│   ├── test_extraction.py         # Rule extraction from sklearn trees (15 tests)
│   ├── test_integration.py        # End-to-end advanced pipeline (18 tests)
│   ├── test_ir.py                 # Intermediate representation (17 tests)
│   ├── test_qm.py                 # Quine-McCluskey minimisation (20 tests)
│   ├── test_routing.py            # Optimizer routing and size estimation (14 tests)
│   └── test_simplify.py           # Boolean simplification primitives (11 tests)
└── README.md                      # This file
```

## Statistics

| Metric | Value |
|---|---|
| Total tests | **326** |
| Status | 100% passing |
| Coverage | **93%** (1734 statements, 126 missed) |
| Runtime | ~24 s (full suite) |
| Reference version | v0.2.3 |

> Figures above were measured with `pytest --cov` on the reference version. Counts
> drift as the suite grows — re-run `pytest --collect-only -q` to refresh them.

---

## Running Tests

### Install dev dependencies

```bash
# From the project root (canonical install, includes pytest + pytest-cov)
pip install -e ".[dev]"
```

> **Tip:** Use a dedicated virtual environment (e.g. `python -m venv .venv`) rather
> than your global interpreter.

### Run all tests

```bash
# From the project root (paths are configured in pyproject.toml)
pytest

# With coverage
pytest tests/ --cov=blackbox2c --cov-report=term
```

### Run specific tests

```bash
# A single module
pytest tests/test_config.py -vv

# A specific class
pytest tests/test_config.py::TestConversionConfig

# A single test
pytest tests/test_config.py::TestConversionConfig::test_default_config
```

---

## Coverage Map

| Module | Coverage | Notes |
|---|---|---|
| `exporters.py` | 99% | C++, Arduino, MicroPython exporters |
| `tree_constants.py` | 100% | Shared leaf-detection constants |
| `optimizer/extraction.py` | 100% | Rule extraction from sklearn trees |
| `optimizer/ir.py` | 99% | Intermediate representation |
| `codegen.py` | 96% | C code generation |
| `optimizer/legacy.py` | 96% | Legacy pruning / merging |
| `optimizer/bdd.py` | 95% | BDD optimiser |
| `optimizer/qm.py` | 94% | Quine-McCluskey optimiser |
| `optimizer/routing.py` | 92% | Optimizer routing and auto-selection |
| `surrogate.py` | 97% | Surrogate tree extraction |
| `codegen_bridge.py` | 84% | RuleSet → hierarchical C bridge |
| `converter.py` | 90% | Main orchestration pipeline |
| `cli.py` | 87% | Command-line interface |
| `analysis.py` | 77% | Feature sensitivity analysis |
| **TOTAL** | **93%** | |

---

## CI

Tests run automatically on every push to `main`/`develop` and on PRs against `main`
via [`.github/workflows/ci.yml`](../.github/workflows/ci.yml):

- **Matrix**: Python 3.8, 3.9, 3.10, 3.11, 3.12 (ubuntu-latest)
- **Install**: `pip install -e ".[dev]"`
- **Run**: `pytest tests/ --tb=short -q`
- **Coverage**: collected on Python 3.11 and uploaded to Codecov

[![Tests](https://github.com/AxelSkrauba/BlackBox2C/actions/workflows/ci.yml/badge.svg)](https://github.com/AxelSkrauba/BlackBox2C/actions)

---

## Writing New Tests

### Template

```python
"""
Tests for the surrogate extractor.
"""

import pytest
import numpy as np
from blackbox2c import Converter, ConversionConfig


class TestSurrogateExtractor:
    """Test SurrogateExtractor behavior."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        rng = np.random.default_rng(42)
        return rng.random((100, 4))

    def test_extract_from_random_forest(self, sample_data):
        # Arrange
        converter = Converter(ConversionConfig())

        # Act
        code = converter.convert(model, sample_data)

        # Assert
        assert "uint8_t predict" in code
```

Use `@pytest.mark.parametrize` for matrix-style cases, as done throughout the
suite (e.g. `test_all_targets_all_levels_no_negative_index[medium-c]`).

### Best Practices

1. **Descriptive names**: `test_convert_random_forest_with_high_fidelity`
2. **One concept per test**: each test validates one specific behavior
3. **Fixtures for setup**: reuse data preparation code
4. **Clear assertions**: informative error messages
5. **Independence**: tests must not depend on each other

### Regression Policy

When you find a bug, add a regression test before fixing it — see
[`test_prune_negative_index_bug.py`](test_prune_negative_index_bug.py) for the
pattern used for the `features[-2]` prune bug. The suite also enforces the
project's leaf-detection convention (`feature == -2` plus both children
undefined) across codegen, exporters and extraction.

---

## Roadmap (Future Test Work)

- [ ] Compile generated C code in CI (real toolchain validation)
- [ ] Performance/benchmark regression tests
- [ ] Integration tests on real hardware
- [ ] Architecture-specific optimization tests

---

## Maintenance

### Update tests when

- A new feature is added
- A bug is found (add a regression test first)
- The public API changes
- Existing code is optimized

### Review tests when

- Tests fail after changes
- Coverage drops
- The suite becomes slow (>1 minute)

---

**Last updated**: 2026-09-22 · **Reference version**: v0.2.3 · **Tests**: 326 · **Coverage**: 93%
