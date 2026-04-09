"""Shared utilities for compression modules."""


def _closest_factor_pair(d):
    """Find the factor pair (a, b) of d where a <= b and a*b == d, minimizing b - a."""
    root = int(d ** 0.5)
    best_a = 1
    best_b = d
    best_diff = best_b - best_a
    for a in range(1, root + 1):
        if d % a == 0:
            b = d // a
            diff = abs(b - a)
            if diff < best_diff:
                best_a, best_b, best_diff = a, b, diff
    return best_a, best_b
