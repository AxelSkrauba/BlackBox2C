"""
Rule optimization for surrogate decision trees.

This package provides multiple optimization levels controlled via
``ConversionConfig.optimize_rules``:

- ``'low'``    — no post-processing (returns the tree as-is).
- ``'medium'`` — prune internal nodes whose direct children are
  same-class leaves (legacy behaviour).
- ``'high'``   — ``'medium'`` + merge sibling leaves with very similar
  class-distribution vectors (legacy behaviour).
- ``'qm'``     — Quine–McCluskey boolean minimization adapted to
  continuous-threshold splits (introduced in v0.2).
- ``'bdd'``    — ROBDD-based reduction with shared-subgraph detection
  (introduced in v0.2).
- ``'auto'``   — dispatch automatically between ``'qm'``, ``'bdd'`` or
  ``'high'`` based on the number of unique literals in the tree
  (introduced in v0.2).

The public ``RuleOptimizer`` class preserves its v0.1 API; advanced
levels are implemented in dedicated submodules and routed transparently.
"""

from .legacy import RuleOptimizer

__all__ = ["RuleOptimizer"]
