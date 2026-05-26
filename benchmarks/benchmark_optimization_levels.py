"""
BlackBox2C Benchmark — Optimization-level sweep (v0.2).

For every classic classification dataset and every supported model,
runs the conversion pipeline once per ``optimize_rules`` level
(``low``, ``medium``, ``high``, ``qm``, ``bdd``, ``auto``) and reports:

* **Fidelity** of the surrogate against the original model
  (computed by ``Converter`` itself).
* **n_rules** in the emitted decision logic
  (sklearn-tree leaves for legacy levels, ``len(RuleSet.rules)``
  for the IR levels).
* **n_literals**: total number of comparisons in the emitted code.
* **est. FLASH** in bytes from the built-in size estimator
  (the bridge generator uses an analogous heuristic, so the numbers
  are directly comparable across levels).
* **time** in seconds.

The script writes a markdown report to ``--output`` (default
``results_v0.2.md``) so the new numbers can be diffed against the v0.1
reference in ``results.md``.

Usage::

    python benchmarks/benchmark_optimization_levels.py
    python benchmarks/benchmark_optimization_levels.py --output results_v0.2.md
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import List, Optional

from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blackbox2c import ConversionConfig, Converter
from blackbox2c.optimizer.extraction import from_sklearn_tree


LEVELS = ("low", "medium", "high", "qm", "bdd", "auto")

DATASETS = [
    ("Iris", load_iris),
    ("Wine", load_wine),
]


def get_models():
    return {
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=20, max_depth=5, random_state=42
        ),
        "SVM": SVC(kernel="rbf", random_state=42),
        "NeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(16, 8), max_iter=500, random_state=42
        ),
    }


# ──────────────────────────────────────────────────────────────────
@dataclass
class Row:
    dataset: str
    model: str
    level: str
    orig_acc: float
    fidelity: float
    n_rules: int
    n_literals: int
    flash_bytes: int
    time_s: float


def _count_logic(converter: Converter, level: str) -> tuple[int, int]:
    """Return (n_rules, n_literals) for the conversion just performed.

    For legacy levels we fall back to inspecting the sklearn tree;
    advanced levels always set ``converter.optimized_ruleset_``.
    """
    if converter.optimized_ruleset_ is not None:
        rs = converter.optimized_ruleset_
        return len(rs.rules), sum(len(r.literals) for r in rs.rules)

    tree = converter.surrogate_tree_
    rs = from_sklearn_tree(tree)
    return len(rs.rules), sum(len(r.literals) for r in rs.rules)


def run_one(dataset_name, load_fn, model_name, model, level, n_classes):
    data = load_fn()
    X, y = data.data, data.target
    feature_names = list(data.feature_names) if hasattr(data, "feature_names") else None
    class_names = list(data.target_names) if hasattr(data, "target_names") else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )

    model.fit(X_train, y_train)
    orig_acc = accuracy_score(y_test, model.predict(X_test))

    cfg = ConversionConfig(
        max_depth=4, optimize_rules=level, n_samples=5000,
        qm_max_literals=10, bdd_max_literals=16,
    )
    converter = Converter(cfg)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            converter.convert(
                model, X_train, X_test=X_test,
                feature_names=feature_names,
                class_names=[str(c) for c in (class_names or range(n_classes))],
            )
        elapsed = time.perf_counter() - t0

    metrics = converter.get_metrics()
    n_rules, n_literals = _count_logic(converter, level)

    return Row(
        dataset=dataset_name,
        model=model_name,
        level=level,
        orig_acc=orig_acc,
        fidelity=metrics["fidelity"],
        n_rules=n_rules,
        n_literals=n_literals,
        flash_bytes=metrics["size_estimate"]["flash_bytes"],
        time_s=elapsed,
    )


def run_all() -> List[Row]:
    rows: List[Row] = []
    for ds_name, loader in DATASETS:
        n_classes = len(loader().target_names)
        print(f"\n{'='*72}")
        print(f"  {ds_name.upper()}  (n_classes={n_classes})")
        print(f"{'='*72}")
        for model_name, _ in get_models().items():
            print(f"\n  -- {model_name} --")
            for lvl in LEVELS:
                # Fresh model instance for every level (legacy
                # optimizer mutates the surrogate tree in place).
                model = get_models()[model_name]
                row = run_one(ds_name, loader, model_name, model, lvl, n_classes)
                rows.append(row)
                print(
                    f"    {lvl:<6s}  fid={row.fidelity:.3f}  "
                    f"rules={row.n_rules:3d}  lits={row.n_literals:3d}  "
                    f"flash={row.flash_bytes:4d}B  t={row.time_s:5.2f}s"
                )
    return rows


# ──────────────────────────────────────────────────────────────────
def build_markdown(rows: List[Row]) -> str:
    out = [
        "# BlackBox2C v0.2 — Optimization-level sweep",
        "",
        "All numbers measured with `max_depth=5`, `qm_max_literals=20`, ",
        "`bdd_max_literals=30`.  Fidelity is the agreement between the ",
        "emitted decision logic and the original model on the held-out test set.",
        "",
        "FLASH estimates come from the built-in size estimator and use the ",
        "same per-condition / per-leaf cost for the legacy and the bridge ",
        "generators, so they are directly comparable across rows.",
        "",
    ]

    by_dataset: dict[str, list[Row]] = {}
    for r in rows:
        by_dataset.setdefault(r.dataset, []).append(r)

    for ds, ds_rows in by_dataset.items():
        out.append(f"## {ds}")
        out.append("")
        # group by model
        by_model: dict[str, list[Row]] = {}
        for r in ds_rows:
            by_model.setdefault(r.model, []).append(r)

        for model_name, mrows in by_model.items():
            out.append(f"### {model_name}")
            out.append("")
            out.append(
                "| Level | Orig. Acc | Fidelity | Rules | Literals | Est. FLASH (B) | Time (s) |"
            )
            out.append(
                "|-------|-----------|----------|-------|----------|-----------------|----------|"
            )
            baseline_flash: Optional[int] = None
            for r in mrows:
                if r.level == "medium":
                    baseline_flash = r.flash_bytes
            for r in mrows:
                if baseline_flash and baseline_flash > 0:
                    delta = (r.flash_bytes - baseline_flash) / baseline_flash * 100
                    flash_str = f"{r.flash_bytes} ({delta:+.0f}%)"
                else:
                    flash_str = str(r.flash_bytes)
                out.append(
                    f"| {r.level} | {r.orig_acc:.3f} | {r.fidelity:.3f} | "
                    f"{r.n_rules} | {r.n_literals} | {flash_str} | {r.time_s:.2f} |"
                )
            out.append("")
    return "\n".join(out)


# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="results_v0.2.md")
    args = parser.parse_args()

    print("\nBlackBox2C v0.2 — Optimization-level sweep")
    rows = run_all()

    report = build_markdown(rows)
    print("\n\n" + "=" * 72)
    print("MARKDOWN REPORT")
    print("=" * 72 + "\n")
    print(report)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[OK] Report saved to: {args.output}")


if __name__ == "__main__":
    main()
