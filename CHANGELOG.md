# Changelog

All notable changes to BlackBox2C will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-15

### Fixed
- **`pyproject.toml`**: `project.dependencies` was incorrectly formatted as a TOML table instead of an array of PEP 508 strings, preventing installation (`pip install` failed with `configuration error`).
- **Regression detection**: `SurrogateExtractor` and `Converter` used a fragile `_estimator_type` instance-attribute check that failed for ensemble regressors (`RandomForestRegressor`, `GradientBoostingRegressor`) in scikit-learn ≥ 1.8. Replaced with `sklearn.base.is_regressor()` with a safe fallback for non-conformant mock models.
- **Build system**: Removed `setuptools_scm` from build dependencies (was unused; no git tag versioning configured) and dropped legacy `setup.py` that duplicated `pyproject.toml` metadata.

### Changed
- Promoted from beta (`0.1.0b1`) to stable (`0.1.0`).
- Development Status classifier updated to `5 - Production/Stable`.
- Added `[tool.setuptools.packages.find]` section to `pyproject.toml` for explicit package discovery.

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

## [Unreleased]

### Planned
- Command-line interface (CLI) enhancements
- Additional optimization algorithms (Quine-McCluskey, BDD)
- Hardware-specific optimizations (SIMD, vectorization)
- Quantization-aware training integration
- Automated benchmarking on real hardware

---

[0.1.0]: https://github.com/AxelSkrauba/BlackBox2C/releases/tag/v0.1.0
[0.1.0b1]: https://github.com/AxelSkrauba/BlackBox2C/releases/tag/v0.1.0b1
