from __future__ import annotations

from typing import List, Tuple


def _eval_poly(coeffs: List[int], x: int, prime: int) -> int:
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % prime
    return acc


def split_secret(secret: int, n_shares: int, threshold: int, prime: int) -> List[Tuple[int, int]]:
    import secrets

    coeffs = [secret] + [secrets.randbelow(prime) for _ in range(threshold - 1)]
    shares = []
    for x in range(1, n_shares + 1):
        y = _eval_poly(coeffs, x, prime)
        shares.append((x, y))
    return shares


def _lagrange_interpolate(x: int, x_s: List[int], y_s: List[int], prime: int) -> int:
    total = 0
    n = len(x_s)
    for i in range(n):
        xi, yi = x_s[i], y_s[i]
        prod = 1
        for j in range(n):
            if i == j:
                continue
            xj = x_s[j]
            num = (x - xj) % prime
            den = (xi - xj) % prime
            inv = pow(den, -1, prime)
            prod = (prod * num * inv) % prime
        total = (total + yi * prod) % prime
    return total


def reconstruct_secret(shares: List[Tuple[int, int]], prime: int) -> int:
    x_s = [x for x, _ in shares]
    y_s = [y for _, y in shares]
    return _lagrange_interpolate(0, x_s, y_s, prime)


def split_vector(secret_vec: List[int], n_shares: int, threshold: int, prime: int) -> List[List[Tuple[int, int]]]:
    """Split a small integer vector into SSS shares component-wise.

    Returns list of length len(secret_vec), each an SSS share list of (x,y).
    """
    return [split_secret(s, n_shares, threshold, prime) for s in secret_vec]


def reconstruct_vector(shares_vec: List[List[Tuple[int, int]]], prime: int) -> List[int]:
    return [reconstruct_secret(shares, prime) for shares in shares_vec]


