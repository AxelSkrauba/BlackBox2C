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

## v0.2 (planned)

- [ ] **Advanced rule optimization**
  - Quine-McCluskey boolean minimization
  - Binary Decision Diagrams (BDD)
- [ ] **Hardware-validated benchmarks** on real MCUs (Arduino Uno, ESP32, Pico)
- [ ] **CLI enhancements**
  - Batch conversion from a config file
  - `--watch` mode for development

## v0.3 (future)

- [ ] **More export formats**: Rust, MicroC
- [ ] **Quantization-aware training integration**
- [ ] **SIMD / vectorization hints** for Cortex-M4/M7
- [ ] **pypi package** (public release)

---

## Contributing

Have a feature request or found a bug?
Open an issue at [github.com/AxelSkrauba/BlackBox2C/issues](https://github.com/AxelSkrauba/BlackBox2C/issues).
