"""
Dispatch logic for ``optimize_rules`` levels.

The router maps the public string identifier to the concrete optimizer
implementation, applying two policies:

1. **Regression safety**: ``'qm'`` and ``'bdd'`` are Boolean
   minimization techniques and have no meaning when leaves carry
   continuous values.  When invoked on a regression RuleSet they emit
   a :class:`UserWarning` and silently degrade to ``'high'`` (legacy
   medium + cosine-merge).

2. **Auto routing**: ``'auto'`` chooses the most aggressive level the
   problem can afford, based on the number of unique literals in the
   RuleSet:

   ============================== =====================================
   Condition                      Algorithm
   ============================== =====================================
   regression                     ``'high'`` + warning
   ``n_lits <= qm_max_literals``  ``'qm'``
   ``n_lits <= bdd_max_literals`` ``'bdd'``
   else                           ``'high'`` + warning
   ============================== =====================================

The router never raises for legitimate inputs; it either runs the
chosen optimizer, falls back gracefully, or — for unknown identifiers
— raises ``ValueError`` with a helpful message.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

from .bdd import BDDOptimizer, DEFAULT_MAX_LITERALS as BDD_DEFAULT_MAX_LITERALS
from .ir import RuleSet
from .qm import QMOptimizer, DEFAULT_MAX_LITERALS as QM_DEFAULT_MAX_LITERALS

_logger = logging.getLogger(__name__)

#: All public optimizer levels recognised by ``ConversionConfig.optimize_rules``.
VALID_LEVELS = ("low", "medium", "high", "qm", "bdd", "auto")


def optimize_ruleset(
    rs: RuleSet,
    level: str,
    *,
    qm_max_literals: int = QM_DEFAULT_MAX_LITERALS,
    bdd_max_literals: int = BDD_DEFAULT_MAX_LITERALS,
) -> RuleSet:
    """Apply the optimizer chosen by ``level`` to ``rs``.

    Parameters
    ----------
    rs : RuleSet
        Input RuleSet.  May come from any of the legacy or advanced
        optimization paths.
    level : str
        One of :data:`VALID_LEVELS`.  ``'low'`` / ``'medium'`` /
        ``'high'`` are handled by the legacy sklearn-tree pipeline and
        are *not* routed through this function — only the IR-level
        levels (``'qm'``, ``'bdd'``, ``'auto'``) reach here.
    qm_max_literals, bdd_max_literals : int
        Caps used by the underlying optimizers and by ``'auto'``
        routing.

    Returns
    -------
    RuleSet
        Optimized (or unchanged, in the regression-fallback case) RuleSet.
    """
    if level not in VALID_LEVELS:
        raise ValueError(
            f"Unknown optimize_rules level {level!r}; "
            f"valid options are {VALID_LEVELS}."
        )

    if level == "low":
        return rs

    if level in ("medium", "high"):
        # Legacy levels operate on the sklearn tree, not on the IR.
        # The Converter is responsible for not routing them here; we
        # raise to make any accidental misuse loud during development.
        raise ValueError(
            f"Level {level!r} is handled by the legacy sklearn-tree "
            "pipeline; it should not reach the IR router."
        )

    # ── Regression safety net ──────────────────────────────────────
    if rs.n_classes is None:
        warnings.warn(
            f"optimize_rules={level!r} is only defined for classification "
            "tasks; falling back to 'high'.  The legacy optimizer is "
            "applied earlier in the pipeline, so this call is a no-op.",
            UserWarning,
            stacklevel=2,
        )
        return rs

    n_lits = len(rs.unique_literals())

    if level == "qm":
        return QMOptimizer(max_literals=qm_max_literals).minimize(rs)

    if level == "bdd":
        return BDDOptimizer(max_literals=bdd_max_literals).minimize(rs)

    # ── auto routing ───────────────────────────────────────────────
    assert level == "auto"
    return _auto_route(rs, n_lits, qm_max_literals, bdd_max_literals)


def _auto_route(
    rs: RuleSet,
    n_lits: int,
    qm_max_literals: int,
    bdd_max_literals: int,
) -> RuleSet:
    """Pick the smallest-FLASH option among {no-op, QM, BDD}.

    Earlier ``auto`` was a strict cap-based dispatcher: try QM first,
    BDD as a backup, no-op only if both rejected.  In practice that
    sometimes preferred BDD over keeping the input RuleSet, even when
    BDD enumeration produced a *larger* tree (Iris-SVM was a notable
    case: +122% over the legacy baseline).

    The new policy runs every applicable optimizer, estimates the
    code-size of each alternative through the same shape estimator the
    bridge codegen uses at emission time, and returns the smallest.
    Lazy-imported to avoid a circular dependency between
    :mod:`routing` and :mod:`codegen_bridge`.
    """
    # Baseline: emit the input RuleSet as-is.
    candidates: list[tuple[int, str, RuleSet]] = []
    baseline_size = _estimate_size(rs)
    candidates.append((baseline_size, "noop", rs))

    if n_lits <= qm_max_literals:
        qm_rs = QMOptimizer(max_literals=qm_max_literals).minimize(rs)
        candidates.append((_estimate_size(qm_rs), "qm", qm_rs))
    if n_lits <= bdd_max_literals:
        bdd_rs = BDDOptimizer(max_literals=bdd_max_literals).minimize(rs)
        candidates.append((_estimate_size(bdd_rs), "bdd", bdd_rs))

    # Stable: ties broken by insertion order (noop, qm, bdd).
    best_size, best_name, best_rs = min(candidates, key=lambda c: c[0])
    _logger.info(
        "optimize_rules='auto' with %d unique literals — picked %s "
        "(estimated FLASH: %d B)",
        n_lits, best_name, best_size,
    )

    if best_name == "noop" and not (
        n_lits <= qm_max_literals or n_lits <= bdd_max_literals
    ):
        warnings.warn(
            f"optimize_rules='auto' could not pick QM or BDD: "
            f"{n_lits} unique literals exceed both "
            f"qm_max_literals={qm_max_literals} and "
            f"bdd_max_literals={bdd_max_literals}.  Returning the "
            "RuleSet unchanged; the upstream legacy 'high' pruning "
            "still applies.",
            UserWarning,
            stacklevel=3,
        )
    return best_rs


def _estimate_size(rs: RuleSet) -> int:
    """Estimated FLASH bytes when ``rs`` is emitted by the bridge codegen.

    Imported lazily to avoid a circular ``routing → codegen_bridge →
    optimizer.routing`` dependency at module load.
    """
    from ..codegen_bridge import RuleSetCodeGenerator  # noqa: WPS433
    gen = RuleSetCodeGenerator()
    # ``_tree_shape`` only inspects rule structure; ``task_type`` not
    # required for size estimation.
    n_internal, n_leaves = gen._tree_shape(list(rs.rules))
    return 50 + n_internal * 12 + n_leaves * 4


def is_advanced_level(level: str) -> bool:
    """Return ``True`` if ``level`` requires the IR pipeline."""
    return level in ("qm", "bdd", "auto")
