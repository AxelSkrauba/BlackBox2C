"""
Forensic inspection of the only configuration where the v0.2 advanced
pipeline actually beat the legacy codegen on estimated FLASH:
RandomForest + Iris + qm.

The script reproduces that exact case and prints, side-by-side:
- the legacy-pruned surrogate tree (after 'medium'),
- the RuleSet extracted from it,
- the RuleSet *after* QM,
- the legacy C output and the bridge C output,
- a per-line FLASH cost decomposition,
- and the same comparison for a couple of *losing* cases (RF-Iris with
  bdd, SVM-Iris with qm) to triangulate what makes the winner special.

Read it as a worksheet, not as a benchmark.
"""

from __future__ import annotations

import contextlib
import io

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from blackbox2c import ConversionConfig, Converter
from blackbox2c.codegen import CCodeGenerator
from blackbox2c.codegen_bridge import RuleSetCodeGenerator
from blackbox2c.optimizer.extraction import from_sklearn_tree
from blackbox2c.optimizer.qm import QMOptimizer
from blackbox2c.optimizer.bdd import BDDOptimizer


def _fmt_rule(rule, idx):
    if not rule.literals:
        return f"  R{idx}: (true) -> {rule.prediction}"
    lits = " AND ".join(
        f"x{l.feature_idx}{l.op}{l.threshold:.2f}" for l in rule.literals
    )
    return f"  R{idx}: {lits} -> {rule.prediction}"


def _print_ruleset(label, rs):
    print(f"\n--- {label} ---")
    print(f"  n_rules     : {len(rs.rules)}")
    n_lits_total = sum(len(r.literals) for r in rs.rules)
    n_lits_unique = len(rs.unique_literals())
    print(f"  total lits  : {n_lits_total}")
    print(f"  unique lits : {n_lits_unique}")
    for i, rule in enumerate(rs.rules):
        print(_fmt_rule(rule, i))


def _flash_breakdown(label, n_lits, n_rules):
    overhead = 50
    per_lit = 12
    per_ret = 4
    flash = overhead + n_lits * per_lit + n_rules * per_ret
    print(
        f"  {label:<32s} "
        f"overhead=50 + {n_lits}*12 + {n_rules}*4 = {flash} B"
    )
    return flash


def run_one(label, model, X_train, X_test, y_train, feature_names, class_names, level):
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    cfg = ConversionConfig(
        max_depth=4, optimize_rules=level, n_samples=5000,
        qm_max_literals=10, bdd_max_literals=16,
    )
    conv = Converter(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        c_code = conv.convert(
            model, X_train, X_test=X_test,
            feature_names=feature_names,
            class_names=class_names,
        )
    metrics = conv.get_metrics()

    print(f"\n  fidelity={metrics['fidelity']:.3f}  "
          f"flash_estimate={metrics['size_estimate']['flash_bytes']} B")

    surrogate_tree = conv.surrogate_tree_
    rs_legacy = from_sklearn_tree(surrogate_tree)
    _print_ruleset("RuleSet from legacy-pruned surrogate tree", rs_legacy)

    if conv.optimized_ruleset_ is not None:
        _print_ruleset(
            f"RuleSet after advanced level={level!r}",
            conv.optimized_ruleset_,
        )
    else:
        print("\n  (no advanced ruleset — legacy path)")

    # Cost breakdown.
    print("\n  FLASH breakdown:")
    tree = surrogate_tree.tree_
    # legacy: per-node cost — internal nodes count as conditions, leaves as returns.
    n_internal = int((tree.feature != -2).sum())
    n_leaves = int((tree.feature == -2).sum())
    _flash_breakdown(
        f"legacy tree codegen ({n_internal}/{n_leaves})",
        n_internal, n_leaves,
    )
    if conv.optimized_ruleset_ is not None:
        rs = conv.optimized_ruleset_
        n_lits = sum(len(r.literals) for r in rs.rules)
        n_rules = len(rs.rules)
        _flash_breakdown(
            f"bridge if-chain      ({n_lits}/{n_rules})",
            n_lits, n_rules,
        )

    return c_code


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )

    rf = RandomForestClassifier(
        n_estimators=20, max_depth=5, random_state=42,
    ).fit(X_tr, y_tr)
    svm = SVC(kernel="rbf", random_state=42).fit(X_tr, y_tr)
    feat_names = [f"f{i}" for i in range(4)]
    class_names = [str(c) for c in iris.target_names]

    # ── The winner ─────────────────────────────────────────────
    run_one(
        "WINNER: RandomForest + Iris + qm",
        RandomForestClassifier(
            n_estimators=20, max_depth=5, random_state=42,
        ).fit(X_tr, y_tr),
        X_tr, X_te, y_tr, feat_names, class_names, "qm",
    )

    # Same case with medium for direct comparison
    run_one(
        "BASELINE: RandomForest + Iris + medium",
        RandomForestClassifier(
            n_estimators=20, max_depth=5, random_state=42,
        ).fit(X_tr, y_tr),
        X_tr, X_te, y_tr, feat_names, class_names, "medium",
    )

    # Same case with bdd (loser)
    run_one(
        "LOSER: RandomForest + Iris + bdd",
        RandomForestClassifier(
            n_estimators=20, max_depth=5, random_state=42,
        ).fit(X_tr, y_tr),
        X_tr, X_te, y_tr, feat_names, class_names, "bdd",
    )

    # Worst loser: SVM + Iris + bdd
    run_one(
        "WORST LOSER: SVM + Iris + bdd",
        SVC(kernel="rbf", random_state=42).fit(X_tr, y_tr),
        X_tr, X_te, y_tr, feat_names, class_names, "bdd",
    )


if __name__ == "__main__":
    main()
