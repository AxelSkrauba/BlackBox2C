# Changelog

All notable changes to BlackBox2C will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Hardware-validated benchmarks on real MCUs (Arduino Uno, ESP32, Pico)
- IR-aware exporters (C++, Arduino, MicroPython): currently advanced
  optimisation only feeds the C target; the platform-specific exporters
  still consume the surrogate sklearn tree.
- CLI batch conversion from config file
- More export formats
- Quantization-aware training integration
- SIMD / vectorization hints for Cortex-M4/M7

---

## [0.2.0] - 2026-05-23

Advanced rule-optimisation pipeline.  Backward-compatible: every
existing `optimize_rules` value (`'low'`, `'medium'`, `'high'`)
produces byte-identical C code to v0.1.

### Added

- **Immutable IR layer** (`blackbox2c.optimizer.ir`): `Literal`,
  `Conjunction`, `RuleSet` frozen dataclasses with a strict
  evaluation contract.  Includes `RuleSet.predict`,
  `RuleSet.complexity` and `RuleSet.unique_literals` for downstream
  consumption.
- **`Conjunction.simplify` / `RuleSet.simplify`**: collapse
  redundant per-feature literals into a single `(lo, hi]` interval
  and drop unsatisfiable rules.  Applied automatically by QM and
  BDD before re-emission.
- **`from_sklearn_tree`** (`blackbox2c.optimizer.extraction`):
  lossless conversion of any scikit-learn tree (classifier or
  regressor) into a `RuleSet`.
- **Quine-McCluskey optimiser**
  (`blackbox2c.optimizer.qm.QMOptimizer`): multi-valued boolean
  minimisation lifted to continuous splits.  Hard caps
  (`max_literals=12`, `max_minterms=4096`) gate the search; over-cap
  inputs return unchanged with a `UserWarning`.
- **ROBDD optimiser** (`blackbox2c.optimizer.bdd.BDDOptimizer`):
  reduced-ordered BDD with frequency-ordered variables and a
  unique-table.  One BDD per output class, then re-emit by
  enumerating true-paths.
- **Routing layer** (`blackbox2c.optimizer.routing.optimize_ruleset`,
  `is_advanced_level`): three new `optimize_rules` values
  (`'qm'`, `'bdd'`, `'auto'`) plus the cap parameters
  `qm_max_literals` and `bdd_max_literals` on `ConversionConfig`.
  `'auto'` runs every applicable optimiser, estimates the FLASH cost
  via the bridge codegen's tree-shape model, and returns the
  smallest – never regressing below the `'medium'` baseline on the
  in-tree benchmark.
- **Hierarchical bridge codegen**
  (`blackbox2c.codegen_bridge.RuleSetCodeGenerator`): rebuilds a
  decision tree from the optimised RuleSet using a
  split-on-most-frequent-literal heuristic and emits the same nested
  if/else surface as the legacy generator, preserving prefix
  sharing.
- **Regression-safety net**: every advanced level emits a single
  `UserWarning` and falls back to the legacy `'high'` path on
  regression tasks; documented at `ConversionConfig`,
  `Converter.convert`, `QMOptimizer`, `BDDOptimizer`, and the
  router.
- **Benchmark suite v0.2**
  (`benchmarks/benchmark_optimization_levels.py`): sweep across
  Iris/Wine + 4 model families + 6 levels, captured in
  `benchmarks/results/v0.2.md`.
- **Tutorial notebook** (`notebooks/07_advanced_optimization.ipynb`,
  mirrored under `docs/notebooks/`): illustrates the new API on
  Iris+RandomForest with a FLASH bar chart and a discussion of
  caveats.

### Changed

- `ConversionConfig.optimize_rules`'s type annotation widened from
  `Literal['low', 'medium', 'high']` to include `'qm'`, `'bdd'`,
  `'auto'`.  Existing values keep their exact semantics.
- `Converter.convert` now stores the optimised `RuleSet` (when an
  advanced level is requested) on `Converter.optimized_ruleset_`
  for downstream inspection.
- The legacy `RuleOptimizer` accepts the new level strings: it
  treats them as `'medium'` internally so QM/BDD always see a
  pruned input tree.

### Performance

Estimated FLASH (heuristic, not yet validated against a compiled
binary) on the Iris+Wine benchmark, best-of-{qm, bdd, auto} vs
legacy `'medium'`:

| Case | `'medium'` | best v0.2 | Δ |
|---|---|---|---|
| Iris + RandomForest | 222 B | 118 B (qm/bdd/auto) | **−47 %** |
| Iris + MLP | 262 B | 214 B (qm) | **−18 %** |
| Wine + RandomForest | 254 B | 214 B (bdd) | **−16 %** |
| Iris + SVM | 262 B | 230 B (qm) | **−12 %** |
| Iris + DecisionTree | 166 B | 166 B (auto) | tie |

Functional equivalence preserved at 100 % across every case.

### Tests

- 79 new tests (212 → 293), grouped under `tests/optimizer/`:
  `test_ir.py`, `test_extraction.py`, `test_qm.py`, `test_bdd.py`,
  `test_simplify.py`, `test_routing.py`, `test_integration.py`.
- Property-based equivalence via Hypothesis on QM and BDD outputs
  vs `tree.predict`.
- An end-to-end mini-interpreter (`test_integration.py::_interpret_c`)
  that parses the generated C body and validates it against the
  original sklearn tree on a held-out grid.

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
