"""Sampling-wall calculator -- the coupon-collector law for fixed-distribution
sampling. The expected number of DISTINCT determinants after M independent
computational-basis shots of a fixed prepared state with amplitudes c_i is the
exact occupancy identity

    E[U(M)] = sum_i [ 1 - (1 - p_i)^M ],     p_i = |c_i|^2 .

This is the identity behind the paper's sampling wall: a concentrated eigenvector
distribution cannot deliver its own determinant support at any realistic shot
budget, so quantum sampling cannot out-collect a classical selector at matched
determinant count. The local exchange rate

    gamma(M) = d log E[U] / d log M

answers "10x more shots buys how many more configurations?" -- the answer is
10^gamma, and gamma < 1 for any concentrated distribution.

Scope: this governs independent computational-basis shots of a FIXED prepared
state (the operation QSCI defines). Adaptive preparations, measurement-basis
changes, and amplitude-estimation access lie outside the law.

Usage:
  python sampling_wall_calculator.py                 # demo: a concentrated dist
  python sampling_wall_calculator.py --measured      # the paper's [2Fe-2S] soup
                                                     #   vector (needs sqdship_vec.npz)
  python sampling_wall_calculator.py --zipf 1.3 --support 170448 --shots 1e6 1e8
"""
import argparse
import sys

import numpy as np


def expected_unique(p, M):
    """E[U(M)] for probabilities p (renormalised, zeros dropped) at shot counts M."""
    p = np.asarray(p, float)
    p = p[p > 0]
    p = p / p.sum()
    lp = np.log1p(-np.minimum(p, 1 - 1e-16))
    M = np.atleast_1d(np.asarray(M, float))
    return np.array([len(p) - np.exp(m * lp).sum() for m in M])


def gamma(p, M):
    """Local exchange rate d log E[U] / d log M at shot count M."""
    M = float(M)
    u = expected_unique(p, [M * 0.9, M * 1.1])
    return float((np.log(u[1]) - np.log(u[0])) / (np.log(M * 1.1) - np.log(M * 0.9)))


def report(name, p, shots):
    p = np.asarray(p, float)
    p = p[p > 0]
    p = p / p.sum()
    supp = len(p)
    print(f"\n== {name} ==")
    print(f"support (distinct determinants, p > 0): {supp:,}")
    for M in shots:
        u = expected_unique(p, [M])[0]
        g = gamma(p, M)
        print(f"  M = {M:>13,.0f} shots -> E[U] = {u:>12,.0f} distinct "
              f"({100 * u / supp:5.1f}% of support) | gamma = {g:.3f} "
              f"(10x shots -> {10 ** g:.1f}x determinants)")


def zipf(support, s):
    k = np.arange(1, support + 1, dtype=float)
    return k ** (-s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measured", action="store_true",
                    help="use the paper's [2Fe-2S] soup vector (sqdship_vec.npz)")
    ap.add_argument("--zipf", type=float, default=1.3,
                    help="demo Zipf exponent (concentration; default 1.3)")
    ap.add_argument("--support", type=int, default=170448,
                    help="demo support size (default 170448, the [2Fe-2S] soup)")
    ap.add_argument("--shots", type=float, nargs="+", default=[1e6, 1e7, 1e8])
    a = ap.parse_args()

    if a.measured:
        try:
            c = np.load("sqdship_vec.npz")["c"]
        except Exception as e:  # noqa: BLE001
            print(f"--measured needs sqdship_vec.npz in the current directory "
                  f"({e.__class__.__name__}); falling back to the --zipf demo.")
        else:
            report("measured [2Fe-2S] aufbau-HCI soup vector (sqdship_vec.npz)",
                   c ** 2, a.shots)
            u = expected_unique(c ** 2, [1e6])[0]
            print(f"\n  paper cross-check: E[U] @ 1e6 shots = {u:,.0f} "
                  f"(closed form 38,129; measured 38,118; 0.03%)")
            return 0

    report(f"demo: Zipf(s={a.zipf}) over {a.support:,} determinants "
           f"(a concentrated, eigenvector-like distribution)",
           zipf(a.support, a.zipf), a.shots)
    print("\nThe wall: a concentrated distribution recovers only a fraction of "
          "its own\nsupport even at 1e8 shots, and gamma < 1 everywhere -- shots "
          "buy determinants\nsub-linearly. Sampling cannot out-collect selection "
          "at matched determinant count.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
