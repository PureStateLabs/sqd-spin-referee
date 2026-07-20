"""Cross-instrument certification for the archive-audit legs: recompute
<S^2> of every saved final SCIState (product space: ci_strs_a x ci_strs_b,
amplitudes matrix) with OUR independent S^2-Gram kernel (from _s2audit.py,
the instrument validated against pyscf labels and used for the flagship
matrix) and compare against THEIR SCIState.spin_square() value stored in the
same npz. Gate: |ours - theirs| < 1e-6.

Usage: run after any *_ship leg lands; checks whatever files exist.
Sentinel: GRAMXCHECK_DONE (exit 1 on gate failure)
"""
import os
import sys

import numpy as np


def popcount64(x):
    return np.bitwise_count(np.asarray(x, np.uint64)).astype(np.uint64)


def s2_gram_ab(a, b, C, ncas):
    a = np.asarray(a, np.uint64)
    b = np.asarray(b, np.uint64)
    C = np.asarray(C, float)
    if C.ndim == 1:
        C = C[:, None]
    oa, ob, ov = [], [], []
    for p in range(ncas):
        bit = np.uint64(1 << p)
        low = np.uint64((1 << p) - 1)
        sel = ((b & bit) != 0) & ((a & bit) == 0)
        if not sel.any():
            continue
        a0, b0 = a[sel], b[sel]
        p1 = ((popcount64(a0) + popcount64(b0 & low)) & np.uint64(1)).astype(np.int64)
        p2 = (popcount64(a0 & low) & np.uint64(1)).astype(np.int64)
        sgn = ((1 - 2 * p1) * (1 - 2 * p2)).astype(float)
        oa.append(a0 | bit)
        ob.append(b0 & ~bit)
        ov.append(C[sel] * sgn[:, None])
    k = C.shape[1]
    if not oa:
        return np.zeros((k, k))
    A = np.concatenate(oa)
    B = np.concatenate(ob)
    V = np.concatenate(ov, axis=0)
    _, inv = np.unique(np.stack([A, B], 1), axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    nu = int(inv.max()) + 1
    W = np.column_stack([np.bincount(inv, weights=V[:, j], minlength=nu)
                         for j in range(k)])
    return W.T @ W


FAILS = []
CAND = [("lucj2ship_symm1.npz", 20), ("lucj2ship_symm0.npz", 20),
        ("ibmship_symm1.npz", 20), ("ibmship_symm0.npz", 20),
        ("lucjoptship_symm1.npz", 20), ("lucjoptship_symm0.npz", 20),
        ("ibm4ship_symm1.npz", 36), ("ibm4ship_symm0.npz", 36)]
for f, nc in CAND:
    if not os.path.exists(f):
        continue
    d = np.load(f, allow_pickle=True)
    if "ci_strs_a" not in d.files:
        print(f"{f}: no saved state, skip")
        continue
    sa = np.asarray(d["ci_strs_a"], np.uint64)
    sb = np.asarray(d["ci_strs_b"], np.uint64)
    amp = np.asarray(d["amplitudes"], float)
    na, nb = len(sa), len(sb)
    assert amp.shape == (na, nb), f"{f}: amp shape {amp.shape} vs {na}x{nb}"
    aa = np.repeat(sa, nb)
    bb = np.tile(sb, na)
    c = amp.ravel()
    nrm = float(c @ c)
    ours = float(s2_gram_ab(aa, bb, c, nc)[0, 0]) / nrm
    theirs = float(d["best_s2"])
    ok = abs(ours - theirs) < 1e-6
    print(f"[{'PASS' if ok else 'FAIL'}] {f}: OUR Gram {ours:.6f} vs THEIR "
          f"spin_square {theirs:.6f} (|d| {abs(ours-theirs):.2e}, "
          f"dim {na}x{nb})")
    if not ok:
        FAILS.append(f)

print(f"\n{len(FAILS)} failures")
print("GRAMXCHECK_DONE", flush=True)
sys.exit(1 if FAILS else 0)
