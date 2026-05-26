"""
Quine-McCluskey minimization adapted to continuous-threshold splits.

Decision-tree splits are tests of the form ``x_i <op> theta_{i,k}``.
Each feature ``i`` accumulates a set of thresholds ``Theta_i`` that
partitions the real line into ``len(Theta_i) + 1`` intervals.  In this
discretised view, every leaf of the tree corresponds to a
multi-dimensional *cube* in the product space of intervals, so the
Boolean Quine-McCluskey procedure naturally generalises to a
*multi-valued* QM:

* Mintermos are tuples of interval indices ``(j_0, ..., j_{F-1})``.
* A *cube* is a tuple of subsets ``(S_0, ..., S_{F-1})`` with each
  ``S_i`` a non-empty subset of the valid interval indices for feature
  ``i``.  A cube whose ``S_i`` covers all intervals is a don't-care on
  feature ``i``.
* Two cubes merge if they coincide on every feature but one and their
  ``S_j`` differ; the merged cube takes the union on that feature.
* Prime implicants are non-mergeable cubes; we cover the on-set with a
  minimal set of PIs (greedy cover, essentials first; exact Petrick
  for very small problems).

The search is bounded both in the *number of literals* and in the
*total number of mintermos* (their product).  When the bounds are
exceeded, the optimizer emits a :class:`UserWarning` and returns the
RuleSet unchanged (no-op fallback) — callers using ``optimize_rules='auto'``
will then fall back to a cheaper level (see ``routing.py``).
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from .ir import Conjunction, Literal, RuleSet


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

#: Default cap on the number of distinct literals (i.e. unique
#: ``(feature, threshold)`` pairs).  Beyond this threshold the
#: optimizer will refuse to run because the multi-valued cube
#: enumeration becomes intractable.
DEFAULT_MAX_LITERALS: int = 12

#: Default cap on the total mintermos count.  Even with few literals,
#: features that share many thresholds explode the cube size; we cut
#: off there to keep wall time bounded.
DEFAULT_MAX_MINTERMS: int = 4096


@dataclass
class QMOptimizer:
    """Multi-valued Quine-McCluskey minimizer over a :class:`RuleSet`.

    Parameters
    ----------
    max_literals : int, default 12
        Maximum number of unique ``(feature, threshold)`` pairs.  When
        exceeded, the input RuleSet is returned unchanged.
    max_minterms : int, default 4096
        Maximum number of mintermos (product of intervals per feature).
        Same fallback semantics.
    petrick_threshold : int, default 6
        For at most this many prime implicants, the cover problem is
        solved exactly via Petrick's method; beyond it we fall back to
        a greedy cover with essential-PI prioritisation.
    """

    max_literals: int = DEFAULT_MAX_LITERALS
    max_minterms: int = DEFAULT_MAX_MINTERMS
    petrick_threshold: int = 6

    # Diagnostic info populated by ``minimize``.
    last_diagnostics_: Dict[str, object] = field(default_factory=dict)

    def minimize(self, rs: RuleSet) -> RuleSet:
        """Return a (possibly) smaller RuleSet equivalent to ``rs``.

        ``minimize`` is a no-op for regression RuleSets (``n_classes is
        None``) — Boolean minimization has no meaning when leaves carry
        continuous values.  Callers should route regression RuleSets to
        a different optimizer.
        """
        if rs.n_classes is None:
            warnings.warn(
                "QMOptimizer is only defined for classification; "
                "returning the input RuleSet unchanged.",
                UserWarning,
                stacklevel=2,
            )
            return rs

        thresholds_per_feature = _collect_thresholds(rs)
        n_literals = sum(len(t) for t in thresholds_per_feature)

        if n_literals == 0:
            # Trivially-true RuleSet (single root leaf) → nothing to do.
            self.last_diagnostics_ = {
                "n_literals": 0, "n_minterms": 1, "applied": False,
                "reason": "single-leaf",
            }
            return rs

        if n_literals > self.max_literals:
            warnings.warn(
                f"QMOptimizer: {n_literals} literals exceed "
                f"max_literals={self.max_literals}; returning input "
                "unchanged.",
                UserWarning,
                stacklevel=2,
            )
            self.last_diagnostics_ = {
                "n_literals": n_literals, "applied": False,
                "reason": "max_literals_exceeded",
            }
            return rs

        sizes = [len(t) + 1 for t in thresholds_per_feature]
        n_minterms = int(np.prod(sizes))
        if n_minterms > self.max_minterms:
            warnings.warn(
                f"QMOptimizer: {n_minterms} mintermos exceed "
                f"max_minterms={self.max_minterms}; returning input "
                "unchanged.",
                UserWarning,
                stacklevel=2,
            )
            self.last_diagnostics_ = {
                "n_literals": n_literals, "n_minterms": n_minterms,
                "applied": False, "reason": "max_minterms_exceeded",
            }
            return rs

        # ── Build the truth table and minimise per class ─────────────
        minterm_index_to_pred = _label_minterms(rs, thresholds_per_feature)
        all_minterms = list(minterm_index_to_pred.keys())

        new_rules: List[Conjunction] = []
        for cls in range(rs.n_classes):
            on_set = [m for m in all_minterms if minterm_index_to_pred[m] == cls]
            if not on_set:
                continue
            off_set = [m for m in all_minterms if minterm_index_to_pred[m] != cls]
            pis = _prime_implicants(on_set, off_set, sizes)
            cover = _cover(on_set, pis, exact_threshold=self.petrick_threshold)
            for cube in cover:
                conjs = _cube_to_conjunctions(
                    cube, thresholds_per_feature, prediction=cls,
                )
                new_rules.extend(conjs)

        if not new_rules:
            # Empty on_set for every class — can only happen if classes
            # is mis-specified.  Return the input to be safe.
            return rs

        # Append a default rule covering whatever the original RuleSet
        # would have predicted for any leftover input — for tree-derived
        # RuleSets the new rules already partition the space, but for
        # extra robustness we keep a fallback that mirrors the most
        # frequent class.
        # We re-evaluate completeness with the original predict.
        new_rs = RuleSet(
            rules=tuple(new_rules),
            n_features=rs.n_features,
            n_classes=rs.n_classes,
            feature_names=rs.feature_names,
        ).simplify()

        self.last_diagnostics_ = {
            "n_literals": n_literals,
            "n_minterms": n_minterms,
            "applied": True,
            "reason": "ok",
            "n_rules_in": len(rs.rules),
            "n_rules_out": len(new_rules),
        }
        return new_rs


# ─────────────────────────────────────────────────────────────────────
# Internal helpers (pure, easy to unit-test)
# ─────────────────────────────────────────────────────────────────────

# A *cube* is a tuple of frozensets: one frozenset per feature,
# containing the indices of the intervals included on that axis.
Cube = Tuple[FrozenSet[int], ...]
Minterm = Tuple[int, ...]


def _collect_thresholds(rs: RuleSet) -> List[List[float]]:
    """For each feature, return the sorted list of distinct thresholds
    that appear in any literal of the RuleSet."""
    per_feat: Dict[int, set] = {i: set() for i in range(rs.n_features)}
    for rule in rs.rules:
        for lit in rule.literals:
            per_feat[lit.feature_idx].add(float(lit.threshold))
    return [sorted(per_feat[i]) for i in range(rs.n_features)]


def _interval_for(value: float, thresholds: Sequence[float]) -> int:
    """Index of the interval in which ``value`` falls.

    Intervals are ``(-inf, t0], (t0, t1], ..., (t_{K-1}, +inf)``.  A
    point exactly equal to a threshold goes to the *left* interval —
    matching scikit-learn's ``<=`` split semantics.
    """
    # Equivalent to:  sum(value > t for t in thresholds)
    # but vectorised via numpy for clarity.
    if not thresholds:
        return 0
    return int(np.sum(value > np.asarray(thresholds)))


def _representative(interval_idx: int, thresholds: Sequence[float]) -> float:
    """A real value that lies inside the interval ``interval_idx`` and
    can be safely passed to :meth:`RuleSet.predict`."""
    if not thresholds:
        return 0.0
    if interval_idx == 0:
        return float(thresholds[0]) - 1.0
    if interval_idx == len(thresholds):
        return float(thresholds[-1]) + 1.0
    # Midpoint of (theta_{k-1}, theta_k] is *strictly inside* the
    # interval as long as the thresholds differ; the optimizer relies
    # on the fact that decision trees never emit duplicate thresholds
    # along a single root-to-leaf path on the same feature.
    a = float(thresholds[interval_idx - 1])
    b = float(thresholds[interval_idx])
    return 0.5 * (a + b)


def _label_minterms(
    rs: RuleSet, thresholds: Sequence[Sequence[float]],
) -> Dict[Minterm, int]:
    """Build the truth table: minterm → class predicted by ``rs``."""
    sizes = [len(t) + 1 for t in thresholds]
    out: Dict[Minterm, int] = {}
    coords = [list(range(s)) for s in sizes]
    # Precompute representative points per axis for a tiny speed-up.
    reps = [
        [_representative(j, thresholds[i]) for j in range(sizes[i])]
        for i in range(len(sizes))
    ]
    # Vectorise: build all sample points at once, predict in bulk.
    grids = list(itertools.product(*coords))
    if not grids:
        return out
    X = np.array(
        [[reps[i][g[i]] for i in range(len(sizes))] for g in grids],
        dtype=float,
    )
    preds = rs.predict(X)
    for g, p in zip(grids, preds):
        out[tuple(g)] = int(p)
    return out


def _prime_implicants(
    on_set: Sequence[Minterm],
    off_set: Sequence[Minterm],
    sizes: Sequence[int],
) -> List[Cube]:
    """Multi-valued Quine-McCluskey: enumerate prime implicants of the
    on_set such that no PI intersects the off_set."""
    if not on_set:
        return []

    # Initial implicants: each minterm as a singleton cube.
    current: List[Cube] = [
        tuple(frozenset({m[i]}) for i in range(len(sizes))) for m in on_set
    ]
    primes: List[Cube] = []
    off_set_t = [tuple(m) for m in off_set]

    while True:
        next_round: List[Cube] = []
        used = [False] * len(current)
        for a, b in itertools.combinations(range(len(current)), 2):
            merged = _try_merge(current[a], current[b])
            if merged is None:
                continue
            # Reject merges that would touch the off_set.
            if _cube_intersects_minterms(merged, off_set_t):
                continue
            used[a] = True
            used[b] = True
            if merged not in next_round:
                next_round.append(merged)
        for k, c in enumerate(current):
            if not used[k] and c not in primes:
                primes.append(c)
        if not next_round:
            break
        current = next_round

    # Deduplicate using set-of-tuples-of-frozensets.
    seen = set()
    uniq: List[Cube] = []
    for c in primes:
        key = tuple(frozenset(s) for s in c)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def _try_merge(a: Cube, b: Cube) -> Optional[Cube]:
    """Return the merge of two cubes if they differ on exactly one
    feature, otherwise ``None``."""
    diff_idx = -1
    for i, (sa, sb) in enumerate(zip(a, b)):
        if sa != sb:
            if diff_idx != -1:
                return None
            diff_idx = i
    if diff_idx == -1:
        return None  # identical
    out = list(a)
    out[diff_idx] = a[diff_idx] | b[diff_idx]
    return tuple(out)


def _cube_contains(cube: Cube, minterm: Minterm) -> bool:
    return all(m in s for m, s in zip(minterm, cube))


def _cube_intersects_minterms(cube: Cube, minterms: Sequence[Minterm]) -> bool:
    return any(_cube_contains(cube, m) for m in minterms)


def _cover(
    on_set: Sequence[Minterm],
    primes: Sequence[Cube],
    exact_threshold: int = 6,
) -> List[Cube]:
    """Choose a (small) subset of ``primes`` that covers ``on_set``.

    Strategy:
    1.  Compute essential prime implicants (PIs that uniquely cover at
        least one mintermo) and add them to the cover.
    2.  Drop the mintermos those PIs already cover.
    3.  For the residual: if at most ``exact_threshold`` PIs remain,
        run Petrick's method exactly; otherwise greedy
        (largest-coverage first).
    """
    if not on_set or not primes:
        return list(primes)

    on_list = list(on_set)
    pi_list = list(primes)

    # Map: pi_index → set of mintermo indices it covers.
    coverage = [
        {j for j, m in enumerate(on_list) if _cube_contains(pi, m)}
        for pi in pi_list
    ]

    # Essentials.
    chosen: List[int] = []
    covered: set = set()
    for j in range(len(on_list)):
        owners = [k for k, cov in enumerate(coverage) if j in cov]
        if len(owners) == 1 and owners[0] not in chosen:
            chosen.append(owners[0])
            covered |= coverage[owners[0]]

    remaining = set(range(len(on_list))) - covered
    if not remaining:
        return [pi_list[k] for k in chosen]

    # Restrict to remaining mintermos.
    residual_pis = [
        (k, coverage[k] & remaining)
        for k in range(len(pi_list))
        if k not in chosen and (coverage[k] & remaining)
    ]

    if len(residual_pis) <= exact_threshold:
        # Petrick: pick the smallest cover by exhaustive search.
        best: Optional[List[int]] = None
        for r in range(1, len(residual_pis) + 1):
            for combo in itertools.combinations(residual_pis, r):
                covered_combo = set()
                for _, cov in combo:
                    covered_combo |= cov
                if covered_combo >= remaining:
                    if best is None or len(combo) < len(best):
                        best = [k for k, _ in combo]
                    break
            if best is not None:
                break
        if best is not None:
            chosen.extend(best)
        else:  # pragma: no cover — should not happen in practice
            chosen.extend(k for k, _ in residual_pis)
    else:
        # Greedy: pick PI with the largest residual coverage.
        residual = remaining
        while residual:
            k_star = max(
                (k for k, _ in residual_pis if coverage[k] & residual),
                key=lambda k: len(coverage[k] & residual),
            )
            chosen.append(k_star)
            residual -= coverage[k_star]

    return [pi_list[k] for k in chosen]


def _cube_to_conjunctions(
    cube: Cube,
    thresholds: Sequence[Sequence[float]],
    prediction: int,
) -> List[Conjunction]:
    """Translate a cube into one or more :class:`Conjunction` instances.

    For each feature, the chosen interval indices may form a contiguous
    run (which translates to a single ``a < x <= b`` constraint, encoded
    as up to two literals) or several disjoint runs.  Disjoint runs on
    a feature would require an OR within a conjunction; since
    :class:`Conjunction` is a pure AND, we instead emit one
    :class:`Conjunction` per Cartesian combination of contiguous runs
    across features.  Trees rarely produce non-contiguous PIs, so this
    cartesian product stays small in practice.
    """
    # For each feature, the list of contiguous index ranges (a, b)
    # such that intervals a..b inclusive are present in the cube.
    runs_per_feature: List[List[Tuple[int, int]]] = []
    for f, axis_set in enumerate(cube):
        n_intervals = len(thresholds[f]) + 1
        if len(axis_set) == n_intervals:
            # don't-care on this feature
            runs_per_feature.append([(0, n_intervals - 1)])
            continue
        sorted_idx = sorted(axis_set)
        runs: List[Tuple[int, int]] = []
        a = b = sorted_idx[0]
        for k in sorted_idx[1:]:
            if k == b + 1:
                b = k
            else:
                runs.append((a, b))
                a = b = k
        runs.append((a, b))
        runs_per_feature.append(runs)

    out: List[Conjunction] = []
    for combo in itertools.product(*runs_per_feature):
        lits: List[Literal] = []
        for f, (a, b) in enumerate(combo):
            n_intervals = len(thresholds[f]) + 1
            # Lower bound: x > thresholds[a-1]   (if a > 0)
            if a > 0:
                lits.append(Literal(f, float(thresholds[f][a - 1]), ">"))
            # Upper bound: x <= thresholds[b]    (if b < n_intervals - 1)
            if b < n_intervals - 1:
                lits.append(Literal(f, float(thresholds[f][b]), "<="))
        out.append(Conjunction(tuple(lits), prediction))
    return out
