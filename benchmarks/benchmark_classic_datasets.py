"""
BlackBox2C Benchmark Suite — Classic Datasets.

Evaluates conversion quality, code size, and timing on well-known
public datasets using multiple model types and all export formats.

Usage:
    python benchmarks/benchmark_classic_datasets.py
    python benchmarks/benchmark_classic_datasets.py --output results.md
"""

import argparse
import time
import sys
import os

from sklearn.datasets import (
    load_iris,
    load_wine,
    fetch_california_housing,
    load_diabetes,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from blackbox2c import Converter, ConversionConfig
from blackbox2c.exporters import create_exporter


# ── Dataset definitions ────────────────────────────────────────────────────

CLASSIFICATION_DATASETS = [
    ("Iris",   load_iris,   3),
    ("Wine",   load_wine,   3),
]

REGRESSION_DATASETS = [
    ("Diabetes",           load_diabetes,           None),
    ("California Housing", fetch_california_housing, None),
]


# ── Model definitions ──────────────────────────────────────────────────────

def get_clf_models():
    return {
        "DecisionTree":  DecisionTreeClassifier(max_depth=5, random_state=42),
        "RandomForest":  RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42),
        "SVM":           SVC(kernel="rbf", random_state=42),
        "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42),
    }


def get_reg_models():
    return {
        "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42),
        "SVR":          SVR(kernel="rbf"),
    }


# ── Benchmark runners ──────────────────────────────────────────────────────

def run_classification_benchmark(dataset_name, load_fn, n_classes):
    print(f"\n{'='*70}")
    print(f"  CLASSIFICATION — {dataset_name}")
    print(f"{'='*70}")

    data = load_fn()
    X, y = data.data, data.target
    feature_names = list(data.feature_names) if hasattr(data, "feature_names") else None
    class_names = list(data.target_names) if hasattr(data, "target_names") else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"  Samples: {len(X_train)} train / {len(X_test)} test | "
          f"Features: {X.shape[1]} | Classes: {n_classes}")

    config = ConversionConfig(max_depth=5, optimize_rules="medium", n_samples=5000)
    rows = []

    for model_name, model in get_clf_models().items():
        model.fit(X_train, y_train)
        orig_acc = accuracy_score(y_test, model.predict(X_test))

        t0 = time.perf_counter()
        converter = Converter(config)
        c_code = converter.convert(
            model, X_train, X_test=X_test,
            feature_names=feature_names,
            class_names=[str(c) for c in (class_names or range(n_classes))],
        )
        elapsed = time.perf_counter() - t0

        metrics = converter.get_metrics()
        rows.append({
            "model":       model_name,
            "orig_acc":    orig_acc,
            "fidelity":    metrics["fidelity"],
            "depth":       metrics["complexity"]["max_depth"],
            "nodes":       metrics["complexity"]["n_nodes"],
            "flash_bytes": metrics["size_estimate"]["flash_bytes"],
            "time_s":      elapsed,
        })

        print(f"  {model_name:<16s}  orig={orig_acc:.3f}  fidelity={metrics['fidelity']:.3f}  "
              f"depth={metrics['complexity']['max_depth']}  "
              f"flash={metrics['size_estimate']['flash_bytes']}B  "
              f"time={elapsed:.2f}s")

    return dataset_name, "classification", rows


def run_regression_benchmark(dataset_name, load_fn):
    print(f"\n{'='*70}")
    print(f"  REGRESSION — {dataset_name}")
    print(f"{'='*70}")

    data = load_fn()
    X, y = data.data, data.target
    feature_names = list(data.feature_names) if hasattr(data, "feature_names") else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    print(f"  Samples: {len(X_train)} train / {len(X_test)} test | Features: {X.shape[1]}")

    config = ConversionConfig(max_depth=5, optimize_rules="medium", n_samples=5000)
    rows = []

    for model_name, model in get_reg_models().items():
        model.fit(X_train, y_train)
        orig_r2 = r2_score(y_test, model.predict(X_test))

        t0 = time.perf_counter()
        converter = Converter(config)
        c_code = converter.convert(
            model, X_train, X_test=X_test,
            feature_names=feature_names,
        )
        elapsed = time.perf_counter() - t0

        metrics = converter.get_metrics()
        rows.append({
            "model":       model_name,
            "orig_r2":     orig_r2,
            "fidelity":    metrics["fidelity"],
            "depth":       metrics["complexity"]["max_depth"],
            "nodes":       metrics["complexity"]["n_nodes"],
            "flash_bytes": metrics["size_estimate"]["flash_bytes"],
            "time_s":      elapsed,
        })

        print(f"  {model_name:<16s}  orig_R²={orig_r2:.3f}  fidelity={metrics['fidelity']:.3f}  "
              f"depth={metrics['complexity']['max_depth']}  "
              f"flash={metrics['size_estimate']['flash_bytes']}B  "
              f"time={elapsed:.2f}s")

    return dataset_name, "regression", rows


def run_format_benchmark():
    """Compare code size across all export formats using Iris + RandomForest."""
    print(f"\n{'='*70}")
    print(f"  FORMAT COMPARISON — Iris + RandomForest")
    print(f"{'='*70}")

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target
    )
    model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    config = ConversionConfig(max_depth=5, optimize_rules="medium", n_samples=5000)
    converter = Converter(config)
    converter.convert(
        model, X_train, X_test=X_test,
        feature_names=list(iris.feature_names),
        class_names=list(iris.target_names),
    )

    rows = []
    for fmt in ["c", "cpp", "arduino", "micropython"]:
        if fmt == "c":
            from blackbox2c.codegen import CCodeGenerator
            gen = CCodeGenerator(function_name="predict")
            code = gen.generate(
                converter.surrogate_tree_,
                converter.feature_names_,
                converter.class_names_,
            )
        else:
            exporter = create_exporter(fmt, function_name="predict")
            code = exporter.generate(
                converter.surrogate_tree_,
                feature_names=converter.feature_names_,
                class_names=converter.class_names_,
            )
        rows.append({"format": fmt, "chars": len(code), "lines": code.count("\n")})
        print(f"  {fmt:<12s}  chars={len(code):6d}  lines={code.count(chr(10)):4d}")

    return rows


# ── Report generation ──────────────────────────────────────────────────────

def build_markdown_report(all_results, format_rows) -> str:
    lines = [
        "# BlackBox2C Benchmark Results",
        "",
        "Benchmarks run on classic public datasets using BlackBox2C v0.1.0.",
        "All conversions use `max_depth=5`, `optimize_rules='medium'`.",
        "",
        "> **Note**: Code sizes are *estimates* produced by BlackBox2C's built-in",
        "> size estimator, not measured from compiled binaries.",
        "",
    ]

    for dataset_name, task, rows in all_results:
        lines.append(f"## {dataset_name} ({task.capitalize()})")
        lines.append("")
        if task == "classification":
            lines.append("| Model | Orig. Accuracy | Fidelity | Depth | Nodes | Est. FLASH (bytes) | Time (s) |")
            lines.append("|-------|---------------|----------|-------|-------|--------------------|----------|")
            for r in rows:
                lines.append(
                    f"| {r['model']} | {r['orig_acc']:.3f} | {r['fidelity']:.3f} | "
                    f"{r['depth']} | {r['nodes']} | {r['flash_bytes']} | {r['time_s']:.2f} |"
                )
        else:
            lines.append("| Model | Orig. R² | Fidelity | Depth | Nodes | Est. FLASH (bytes) | Time (s) |")
            lines.append("|-------|---------|----------|-------|-------|--------------------|----------|")
            for r in rows:
                lines.append(
                    f"| {r['model']} | {r['orig_r2']:.3f} | {r['fidelity']:.3f} | "
                    f"{r['depth']} | {r['nodes']} | {r['flash_bytes']} | {r['time_s']:.2f} |"
                )
        lines.append("")

    lines.append("## Export Format Comparison (Iris + RandomForest)")
    lines.append("")
    lines.append("| Format | Characters | Lines |")
    lines.append("|--------|-----------|-------|")
    for r in format_rows:
        lines.append(f"| {r['format']} | {r['chars']} | {r['lines']} |")
    lines.append("")

    return "\n".join(lines)


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BlackBox2C benchmark suite with classic datasets."
    )
    parser.add_argument(
        "-o", "--output",
        help="Write markdown report to this file (default: print to stdout)",
    )
    args = parser.parse_args()

    print("\nBlackBox2C Benchmark Suite")
    print("="*70)

    all_results = []

    for name, loader, n_cls in CLASSIFICATION_DATASETS:
        result = run_classification_benchmark(name, loader, n_cls)
        all_results.append(result)

    for name, loader, _ in REGRESSION_DATASETS:
        result = run_regression_benchmark(name, loader)
        all_results.append(result)

    format_rows = run_format_benchmark()

    report = build_markdown_report(all_results, format_rows)

    print(f"\n{'='*70}")
    print("MARKDOWN REPORT")
    print("="*70)
    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[OK] Report saved to: {args.output}")


if __name__ == "__main__":
    main()
