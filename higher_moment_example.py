"""Higher-moment spin fingerprint -- why <S^2> alone cannot identify a state, and
how <S^4>/<S^6>/<S^8> plus a linear program over spin sectors pin the spin
decomposition. Self-contained (numpy + scipy); the ground truth is analytic.

A state that is an incoherent mixture of total-spin sectors S with weights w_S has

    <S^{2k}> = sum_S w_S * [S(S+1)]^k .

Two very different states can share <S^2> = 2.00:
  A -- a pure triplet         w = {S=1: 1}
  B -- a 2:1 singlet:quintet  w = {S=0: 2/3, S=2: 1/3}
Both give <S^2> = 2.00, but their higher moments diverge, and a linear program
over the sector simplex constrained by the measured moments bounds the triplet
weight of B to ~0. This is exactly the paper's [2Fe-2S] set-draw result: two
audited arms at <S^2> ~ 2.00, one a real triplet and one a singlet:quintet
mixture whose <S^6> = 72.30 (a 60%-triplet alternative would give <S^6> = 120)
and whose triplet weight the LP bounds below 1e-4.

  python higher_moment_example.py
"""
import numpy as np
from scipy.optimize import linprog

S = np.arange(0, 6)            # spin sectors S = 0, 1, 2, 3, 4, 5
CAS = S * (S + 1)              # Casimir S(S+1) = 0, 2, 6, 12, 20, 30


def moments(w, kmax=4):
    """<S^2>, <S^4>, <S^6>, <S^8> for sector weights w."""
    w = np.asarray(w, float)
    return np.array([(w * CAS ** k).sum() for k in range(1, kmax + 1)])


def lp_triplet_bounds(m, tol=1e-7, use=3):
    """min / max triplet (S=1) weight over the sector simplex, subject to
    sum w = 1 and <S^{2k}> = m_k (k = 1..use) within +/- tol, w >= 0."""
    A_eq, b_eq = [np.ones(6)], [1.0]
    A_ub, b_ub = [], []
    for k in range(use):
        row = CAS ** (k + 1)
        A_ub += [row, -row]
        b_ub += [m[k] + tol, -(m[k] - tol)]
    bounds = [(0.0, 1.0)] * 6
    lo = linprog([0, 1, 0, 0, 0, 0], A_ub, b_ub, A_eq, b_eq, bounds, method="highs")
    hi = linprog([0, -1, 0, 0, 0, 0], A_ub, b_ub, A_eq, b_eq, bounds, method="highs")
    if not (lo.success and hi.success):
        return None
    return lo.fun, -hi.fun


def as_weights(d):
    w = np.zeros(6)
    for k, v in d.items():
        w[k] = v
    return w


cases = [
    ("A: pure triplet          w = {S1: 1}", {1: 1.0}),
    ("B: 2:1 singlet:quintet   w = {S0: 2/3, S2: 1/3}", {0: 2 / 3, 2: 1 / 3}),
]

print("spin sectors S           :", [int(s) for s in S])
print("Casimir  S(S+1)          :", [int(c) for c in CAS])
for name, wd in cases:
    m = moments(as_weights(wd))
    lo, hi = lp_triplet_bounds(m)
    print(f"\n{name}")
    print(f"  <S^2> = {m[0]:.4f}   <S^4> = {m[1]:.4f}   "
          f"<S^6> = {m[2]:.4f}   <S^8> = {m[3]:.4f}")
    print(f"  LP triplet-weight bound from (<S^2>,<S^4>,<S^6>): "
          f"[{lo:.5f}, {hi:.5f}]")

print("\nBoth states have <S^2> = 2.0000 -- identical on the scalar mean.")
print("<S^6> alone separates them (8 vs 72), and the LP pins B's triplet weight")
print("to ~0: a 2:1 singlet:quintet, not a triplet. A scalar <S^2> cannot")
print("identify a state; the higher moments and the sector LP can.")
