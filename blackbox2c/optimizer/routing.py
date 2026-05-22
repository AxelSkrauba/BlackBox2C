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
    if n_lits <= qm_max_literals:
        _logger.info(
            "optimize_rules='auto' with %d unique literals → using 'qm'.",
            n_lits,
        )
        return QMOptimizer(max_literals=qm_max_literals).minimize(rs)
    if n_lits <= bdd_max_literals:
        _logger.info(
            "optimize_rules='auto' with %d unique literals → using 'bdd'.",
            n_lits,
        )
        return BDDOptimizer(max_literals=bdd_max_literals).minimize(rs)

    warnings.warn(
        f"optimize_rules='auto' could not pick QM or BDD: {n_lits} "
        f"unique literals exceed both qm_max_literals={qm_max_literals} "
        f"and bdd_max_literals={bdd_max_literals}.  Returning the "
        "RuleSet unchanged; the upstream legacy 'high' pruning still "
        "applies.",
        UserWarning,
        stacklevel=2,
    )
    return rs


def is_advanced_level(level: str) -> bool:
    """Return ``True`` if ``level`` requires the IR pipeline."""
    return level in ("qm", "bdd", "auto")
