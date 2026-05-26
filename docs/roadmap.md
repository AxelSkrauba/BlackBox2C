# Roadmap

## v0.1.0 (released)

- [x] Core conversion pipeline (surrogate extraction, rule optimization, C codegen)
- [x] Classification and regression support
- [x] Multi-format export: C, C++11, Arduino, MicroPython
- [x] Feature sensitivity analysis
- [x] Memory budget auto-tuning
- [x] CLI: `convert`, `analyze`, `export`
- [x] Benchmark suite (Iris, Wine, Diabetes, California Housing)
- [x] 167 tests, >91% coverage

## v0.2.0 (released)

- [x] **Advanced rule optimization**
  - [x] Quine-McCluskey multi-valued boolean minimization (`optimize_rules='qm'`)
  - [x] Reduced Ordered BDDs with frequency-ordered variables (`optimize_rules='bdd'`)
  - [x] Smallest-FLASH `'auto'` routing across no-op / QM / BDD
- [x] **Immutable RuleSet IR** (`Literal`, `Conjunction`, `RuleSet`) with literal simplification
- [x] **Hierarchical bridge codegen** preserving prefix sharing in nested if/else
- [x] **Regression-safety net** with `UserWarning` + fallback to `'high'` on regression tasks
- [x] **Benchmark suite v0.2** (`benchmarks/benchmark_optimization_levels.py`, results in `benchmarks/results/v0.2.md`)
- [x] **Tutorial notebook** `07_advanced_optimization.ipynb`
- [x] 293 tests, full backward compatibility with `'low'`/`'medium'`/`'high'`

## v0.3 (planned)

- [ ] **Hardware-validated benchmarks** on real MCUs (Arduino Uno, ESP32, Pico)
- [ ] **CLI enhancements**
  - Batch conversion from a config file
  - `--watch` mode for development
- [ ] **Advanced optimization for regression** (currently classification-only)

## v0.4 (future)

- [ ] **Quantization-aware training integration**
- [ ] **SIMD / vectorization hints** for Cortex-M4/M7

---

## Contributing

Have a feature request or found a bug?
Open an issue at [github.com/AxelSkrauba/BlackBox2C/issues](https://github.com/AxelSkrauba/BlackBox2C/issues).
