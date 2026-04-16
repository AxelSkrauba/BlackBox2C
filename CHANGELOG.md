# Changelog

All notable changes to BlackBox2C will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Advanced rule optimization: Quine-McCluskey boolean minimization, BDDs
- Hardware-validated benchmarks on real MCUs (Arduino Uno, ESP32, Pico)
- CLI batch conversion from config file
- More export formats
- Quantization-aware training integration
- SIMD / vectorization hints for Cortex-M4/M7

---

## [0.1.1] - 2026-04-16

### Fixed

- **`codegen.py` — critical bug: invalid C code generated for C target**
  (`CCodeGenerator._generate_tree_logic`): feature names were emitted verbatim
  as C identifiers inside `if`-conditions (e.g. `if (petal length (cm) <= 2.45f)`),
  producing code that fails to compile whenever feature names contain spaces,
  parentheses, or other characters that are invalid in C identifiers.
  The fix always emits `features[<index>]` array indexing, matching the
  behaviour already present in the `CppExporter`, `ArduinoExporter`, and
  `MicroPythonExporter`.  This bug affected **all C-target outputs** when
  `feature_names` with special characters were supplied (e.g. the standard
  scikit-learn Iris, Wine, Diabetes datasets).

- **`codegen.py` — latent bug: unsanitized class names in `#define` macros**
  (`CCodeGenerator._generate_defines`): class names were used verbatim as
  C preprocessor identifiers, which is invalid when they contain spaces,
  hyphens, or start with a digit.  The new `_sanitize_c_identifier()` helper
  (already present in `exporters.py` for the C++ exporter) is now shared and
  applied consistently.  Sanitized names are uppercased
  (e.g. `"class-A (ok)"` → `CLASS_A__OK_`).

### Tests

- **`tests/test_codegen.py`** fully rewritten:
  - All fixtures now use **real feature names** (Iris dataset, sensor names
    with units and spaces) instead of the previous workaround that pre-built
    `features[i]` strings to mask the bug.
  - Added `TestFeatureNameBugRegression` class with 5 regression tests that
    directly verify the bug is absent.
  - Added regression tree tests, `_sanitize_c_identifier` unit tests, and a
    monotonicity test for the size estimator.
- **`tests/test_target_param.py`**: updated `#define setosa 0` assertion to
  `#define SETOSA 0` to reflect the now-correct identifier sanitization.

---

## [0.1.0] - 2026-04-15

### Fixed
- **`pyproject.toml`**: `project.dependencies` was incorrectly formatted as a TOML table instead of an array of PEP 508 strings, preventing installation (`pip install` failed with `configuration error`).
- **Regression detection**: `SurrogateExtractor` and `Converter` used a fragile `_estimator_type` instance-attribute check that failed for ensemble regressors (`RandomForestRegressor`, `GradientBoostingRegressor`) in scikit-learn ≥ 1.8. Replaced with `sklearn.base.is_regressor()` with a safe fallback for non-conformant mock models.
- **Build system**: Removed `setuptools_scm` from build dependencies (was unused; no git tag versioning configured) and dropped legacy `setup.py` that duplicated `pyproject.toml` metadata.

### Changed
- Promoted from beta (`0.1.0b1`) to stable (`0.1.0`).
- Development Status classifier updated to `5 - Production/Stable`.
- Added `[tool.setuptools.packages.find]` section to `pyproject.toml` for explicit package discovery.

### Documentation (post-release)
- Added 6 Jupyter notebook examples (`notebooks/`) covering quickstart, classification, regression, feature analysis, multi-format export, and a full end-to-end IoT pipeline on the ADL Air Quality dataset.
- Integrated notebooks into MkDocs docs via `mkdocs-jupyter` plugin with a pre-build hook.
- Updated installation instructions across all docs: `pip install blackbox2c` as primary, with virtual environment guidance.
- Fixed `docs/api/analysis.md` `plot()` signature, `optimize_rules` description in `config.md`.
- Removed stale "PyPI release" item from roadmap (already released).

### Infrastructure (post-release)
- Added `publish.yml` GitHub Actions workflow: tag-triggered, runs tests → build → PyPI Trusted Publisher → GitHub Release.
- Configured PyPI Trusted Publisher (OIDC) — no API token stored in secrets.

---

## [0.1.0b1] - 2025-07-01

### Added
- **Multi-format export**: Support for C, C++11, Arduino, and MicroPython
- **Regression support**: Full support for regression models (DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor)
- **Feature analysis**: Automatic feature importance analysis with `FeatureSensitivityAnalyzer`
- **Factory pattern**: Easy exporter creation with `create_exporter()`
- **Comprehensive tests**: 128 tests covering all functionality
- **Complete examples**: 4 full examples demonstrating all features

### Features
- Convert scikit-learn models to optimized embedded code
- Support for classification and regression tasks
- Automatic task detection (classification vs regression)
- Surrogate model extraction for complex models (SVM, Neural Networks)
- Rule optimization with configurable levels (low, medium, high)
- Fixed-point arithmetic support for resource-constrained devices
- Memory budget management
- Code size estimation
- Fidelity calculation

### Exporters
- **C Exporter**: Standard C code with if-else logic
- **C++ Exporter**: Modern C++11 with classes, templates, and STL
- **Arduino Exporter**: Optimized for Arduino boards with PROGMEM support
- **MicroPython Exporter**: Pure Python for microcontrollers

### Examples
- `iris_example.py`: Classification with multiple models
- `temperature_regression.py`: Regression with 5 use cases
- `feature_selection_example.py`: Feature importance analysis
- `multi_format_export.py`: Export to all formats

### Documentation
- Comprehensive README with quick start guide
- API documentation in docstrings
- Complete examples with explanations
- Troubleshooting guide

### Tests
- 128 tests (100% passing)
- ~90% code coverage
- Unit tests for all modules
- Integration tests
- Edge case tests

[Unreleased]: https://github.com/AxelSkrauba/BlackBox2C/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/AxelSkrauba/BlackBox2C/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/AxelSkrauba/BlackBox2C/releases/tag/v0.1.0
[0.1.0b1]: https://github.com/AxelSkrauba/BlackBox2C/releases/tag/v0.1.0b1
