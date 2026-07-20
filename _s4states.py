"""S^4 second-moment audit of the eight saved final pipeline states
(review round 3, must-fix #1): the [2Fe-2S] hardware/template/set-draw legs
and the [4Fe-4S] hardware leg, mitigation on and off — the same eight states
certified in _gramxcheck.py. For each: <S^2> (gated against THEIR stored
spin_square to 1e-6), <S^4>, Var(S^2) = <S^4> - <S^2>^2, and the constrained
{S=0,1,2} two-moment sector solve. A spin eigenstate has Var = 0; this is
the measurement that decides whether any scalar-mean state ("spin-pure
triplet", <S^2> = 2.000) is actually pure.

Moment identity (M_s = 0): W = S+ v, G = W^T W; W2 = S+ W;
G4 = W2^T W2 + 2 G. Gauged on H4/STO-3G FCI in-process before any state
(eigenstate Var < 1e-9, analytic mixture Var = 4 s^2 c^2), same gate as
_gram7500s4.py (G0).

Out: s4states.npz  Sentinel: S4STATES_DONE (exit 1 on any gate failure)
"""
import os
import sys
import time

import numpy as np

from pyscf import fci, gto, scf
from pyscf.fci import cistring

T0 = time.time()

M1 = np.uint64(0x5555555555555555)
M2 = np.uint64(0x3333333333333333)
M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
H01 = np.uint64(0x0101010101010101)


def popcount64(a):
    a = a - ((a >> np.uint64(1)) & M1)
    a = (a & M2) + ((a >> np.uint64(2)) & M2)
    a = (a + (a >> np.uint64(4))) & M4
    return (a * H01) >> np.uint64(56)


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def splus_image(a, b, C, ncas):
    """Verbatim from _gram7500s4.py (phase logic from the shipped
    s2_gram_ab); returns the deduped S+ image (A, B, W)."""
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
        p1 = ((popcount64(a0) + popcount64(b0 & low)) & 1).astype(np.int64)
        p2 = (popcount64(a0 & low) & 1).astype(np.int64)
        sgn = ((1 - 2 * p1) * (1 - 2 * p2)).astype(float)
        oa.append(a0 | bit)
        ob.append(b0 & ~bit)
        ov.append(C[sel] * sgn[:, None])
    k = C.shape[1]
    if not oa:
        return (np.zeros(0, np.uint64), np.zeros(0, np.uint64),
                np.zeros((0, k)))
    A = np.concatenate(oa)
    B = np.concatenate(ob)
    V = np.concatenate(ov, axis=0)
    pair = np.stack([A, B], 1)
    del A, B
    uniq, inv = np.unique(pair, axis=0, return_inverse=True)
    del pair
    inv = np.asarray(inv).reshape(-1)
    nu = uniq.shape[0]
    W = np.column_stack([np.bincount(inv, weights=V[:, j], minlength=nu)
                         for j in range(k)])
    return uniq[:, 0].copy(), uniq[:, 1].copy(), W


def spin_moments(a, b, C, ncas):
    A, B, W = splus_image(a, b, C, ncas)
    G = W.T @ W
    _, _, W2 = splus_image(A, B, W, ncas)
    G4 = W2.T @ W2 + 2.0 * G
    return G, G4


def sector_weights(m, q):
    w2 = (q - 2.0 * m) / 24.0
    w1 = (6.0 * m - q) / 8.0
    return 1.0 - w1 - w2, w1, w2


# ---- G0: H4/STO-3G moment gauge (identical to _gram7500s4.py) ----
mol = gto.M(atom="H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6",
            basis="sto-3g", verbose=0)
mf = scf.RHF(mol).run()
cis = fci.FCI(mol, mf.mo_coeff)
cis.nroots = 6
es_h4, vs_h4 = cis.kernel()
sa_h4 = np.asarray(cistring.make_strings(range(4), 2), np.uint64)
aw_h4 = np.repeat(sa_h4, len(sa_h4))
bw_h4 = np.tile(sa_h4, len(sa_h4))
vm_h4 = np.column_stack([np.asarray(v).ravel() for v in vs_h4])
Gh, G4h = spin_moments(aw_h4, bw_h4, vm_h4, 4)
var_h4 = np.diagonal(G4h) - np.diagonal(Gh) ** 2
ok = bool((np.abs(var_h4) < 1e-9).all())
i0 = int(np.argmin(np.diagonal(Gh)))
i1 = int(np.argmin(np.abs(np.diagonal(Gh) - 2.0)))
mix = (np.sqrt(0.7) * vm_h4[:, i0] + np.sqrt(0.3) * vm_h4[:, i1])
mix /= np.linalg.norm(mix)
Gm, G4m = spin_moments(aw_h4, bw_h4, mix, 4)
varm = float(G4m[0, 0] - Gm[0, 0] ** 2)
ok &= abs(float(Gm[0, 0]) - 0.6) < 1e-9 and abs(varm - 0.84) < 1e-9
log(f"G0 H4 moment gauge: |Var|max {np.abs(var_h4).max():.1e}, mix mean "
    f"{float(Gm[0,0]):.6f} Var {varm:.6f} [{'PASS' if ok else 'FAIL'}]")
assert ok, "G0 FAIL: moment kernel does not reproduce exact spin answers"

# ---- the eight saved final pipeline states ----
CAND = [("ibmship_symm1.npz", 20, "[2Fe-2S] hardware, symm on"),
        ("ibmship_symm0.npz", 20, "[2Fe-2S] hardware, symm off"),
        ("lucj2ship_symm1.npz", 20, "[2Fe-2S] template, symm on"),
        ("lucj2ship_symm0.npz", 20, "[2Fe-2S] template, symm off"),
        ("lucjoptship_symm1.npz", 20, "[2Fe-2S] set-draw, symm on"),
        ("lucjoptship_symm0.npz", 20, "[2Fe-2S] set-draw, symm off"),
        ("ibm4ship_symm1.npz", 36, "[4Fe-4S] hardware, symm on"),
        ("ibm4ship_symm0.npz", 36, "[4Fe-4S] hardware, symm off")]

rows = []
fails = []
for f, nc, label in CAND:
    if not os.path.exists(f):
        fails.append(f"{f}: MISSING")
        continue
    d = np.load(f, allow_pickle=True)
    if "ci_strs_a" not in d.files:
        fails.append(f"{f}: no saved state")
        continue
    sa = np.asarray(d["ci_strs_a"], np.uint64)
    sb = np.asarray(d["ci_strs_b"], np.uint64)
    amp = np.asarray(d["amplitudes"], float)
    na, nb = len(sa), len(sb)
    assert amp.shape == (na, nb), f"{f}: amp shape {amp.shape}"
    c = amp.ravel()
    c = c / np.linalg.norm(c)
    aa = np.repeat(sa, nb)
    bb = np.tile(sb, na)
    t0 = time.time()
    G, G4 = spin_moments(aa, bb, c, nc)
    m = float(G[0, 0])
    q = float(G4[0, 0])
    var = q - m * m
    theirs = float(d["best_s2"])
    gate = abs(m - theirs) < 1e-6
    w0, w1, w2 = sector_weights(m, q)
    beyond = min(w0, w1, w2) < -1e-3
    rows.append((f, label, na * nb, m, theirs, q, var, w0, w1, w2))
    log(f"[{'PASS' if gate else 'FAIL'}] {label} ({f}, dim {na}x{nb}): "
        f"<S2> {m:.6f} (theirs {theirs:.6f}), <S4> {q:.4f}, "
        f"Var {var:.6f}, w=({w0:+.4f},{w1:+.4f},{w2:+.4f})"
        f"{'  [beyond-S=2 support]' if beyond else ''}"
        f"  [{time.time()-t0:.0f}s]")
    if not gate:
        fails.append(f"{f}: mean gate |{m:.8f}-{theirs:.8f}|")

np.savez_compressed(
    "s4states.npz",
    files=np.array([r[0] for r in rows]),
    labels=np.array([r[1] for r in rows]),
    dims=np.array([r[2] for r in rows], np.int64),
    s2=np.array([r[3] for r in rows]),
    s2_theirs=np.array([r[4] for r in rows]),
    s4=np.array([r[5] for r in rows]),
    var=np.array([r[6] for r in rows]),
    w=np.array([[r[7], r[8], r[9]] for r in rows]))
log(f"saved s4states.npz ({len(rows)} states)")
if fails:
    print("FAILS:", fails)
print("S4STATES_DONE", flush=True)
sys.exit(1 if fails else 0)
