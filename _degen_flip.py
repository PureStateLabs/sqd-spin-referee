"""Probe the one no-pair completion checkpoint: fe4s4 D=1106 -> 2097.

A unique ground of a flip-closed space under a flip-invariant H must be its
own spin image (flip(v) = +/- v): any state with Rayleigh quotient E0 is a
ground eigenvector, and the flip of the ground has RQ E0 exactly. Measure it.
Sentinel: FLIP_DONE
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pyci  # noqa: E402

from _degen import s2_gram_ab, rowkey  # noqa: E402  (_degen has a __main__ guard)

FE4 = ("_qsci_benchmarks/fe2s2/Active-space-model-for-Iron-Sulfur-"
      "Clusters/Fe2S2_and_Fe4S4/Fe4S4/fe4s4")

ham = pyci.hamiltonian(FE4)
nel = (27, 27)  # from the FCIDUMP header (NELEC=54, Ms=0), as _degen validates

rows = np.load("s2audit_fe4s4.npz", allow_pickle=True)["results"]
r = next(x for x in rows if int(x["ndets"]) == 1106)
da = np.asarray(r["dets_a"], np.uint64)
db = np.asarray(r["dets_b"], np.uint64)
Bd = np.stack([da, db], 1)
comp = np.unique(np.concatenate([Bd, Bd[:, ::-1]]), axis=0)
print(f"D {len(Bd)} -> completed {len(comp)} "
      f"(self-symmetric/shared: {2*len(Bd)-len(comp)})", flush=True)

w = pyci.fullci_wfn(ham.nbasis, *nel)
for row_ in comp:
    w.add_det(row_)
op = pyci.sparse_op(ham, w)
es, vs = op.solve(n=3, tol=1e-9)
es = np.asarray(es)
o = np.argsort(es)
es, vs = es[o], np.asarray(vs)[o]
B = w.to_det_array().reshape(len(w), 2)
print(f"lowest 3: {es[:3]}")
print(f"gap01 {es[1]-es[0]:.6e}  gap12 {es[2]-es[1]:.6e}")
print(f"stored e_symm[0:2] {np.asarray(r['e_symm'])[:2]}")

# flip overlap via searchsorted on row keys
kk = rowkey(B)
order = np.argsort(kk)
flipk = rowkey(B[:, ::-1])
pos = order[np.searchsorted(kk[order], flipk)]
assert np.array_equal(kk[pos], flipk), "flip image not closed?!"
v0 = vs[0]
vf = np.zeros_like(v0)
vf[pos] = v0
f = float(v0 @ vf)
print(f"naive (sign-free) flip overlap = {f:+.12f}")

# proper spin-flip carries a fermionic per-det sign; in the interleaved JW
# convention it is (-1)^{n_docc} (swap alpha/beta ops inside each doubly-
# occupied orbital). Blocked convention gives a global sign instead. Try both.
from _fairfight import popcount64  # noqa: E402
docc = popcount64(np.asarray(B[:, 0], np.uint64) & np.asarray(B[:, 1], np.uint64))
sgn = np.where(docc % 2, -1.0, 1.0)
vf2 = np.zeros_like(v0)
vf2[pos] = v0 * sgn          # F|a,b> = (-1)^{docc}|b,a>
f2 = float(v0 @ vf2)
print(f"docc-signed flip overlap       = {f2:+.12f}  (+/-1 => own spin image)")
print(f"overlap of signed flip with root1: {float(vs[1] @ vf2):+.6e}, root2: {float(vs[2] @ vf2):+.6e}")

# --- measure the true flip signs from H itself ---------------------------
# For the flip F = P.S (P the (a,b)->(b,a) permutation, S diagonal signs),
# [H,F]=0  <=>  H[F(j),F(i)] = s_j s_i H[j,i].  Extract s_j s_i0 from two
# H columns (i0 and F(i0)), then form the sign-corrected overlap.
n = len(B)


def hcol(i):
    e = np.zeros(n)
    e[i] = 1.0
    return op.matvec(e)


i0 = int(np.argmax(np.abs(v0)))
c1 = hcol(i0)            # H[:, i0]
c2 = hcol(int(pos[i0]))  # H[:, F(i0)]
c2p = c2[pos]            # row-permuted: c2p[j] = H[F(j), F(i0)]
mask = (np.abs(c1) > 1e-10) & (np.abs(c2p) > 1e-10)
ratio = c2p[mask] / c1[mask]
print(f"connected entries {mask.sum()}; |ratio| spread "
      f"{np.abs(ratio).min():.6f}..{np.abs(ratio).max():.6f}")
s_rel = np.sign(ratio)   # = s_j * s_i0 on connected j
# sign-corrected overlap restricted to the measured-sign support
idx = np.where(mask)[0]
num = float(np.sum(v0[idx] * s_rel * v0[pos[idx]]))
den = float(np.sum(v0[idx] * v0[pos[idx]] * 0 + np.abs(v0[idx] * v0[pos[idx]])))
w_meas = float(np.sum(v0[idx] ** 2))
print(f"sign-corrected partial overlap over {len(idx)} measured dets "
      f"(carrying {w_meas:.4f} of |v0|^2): {num:+.6f}")
print(f"  (sum of |v0_j v0_Fj| over same dets: {den:.6f} -- ratio "
      f"{num/den if den else float('nan'):+.6f}; +/-1 => own image on support)")

# ground weight on original D
inD = np.isin(kk, rowkey(Bd))
print(f"ground weight on original D: {float((v0[inD]**2).sum()):.6f}")

# --- independent check: dense-diagonalize the full 2097-dim block --------
print(f"wfn size {len(w)} vs completed {len(comp)}", flush=True)
H = np.column_stack([hcol(i) for i in range(n)])
print(f"dense H built; symmetry |H-H^T|_max = {np.abs(H - H.T).max():.3e}")
ew, Vd = np.linalg.eigh(H)
print(f"dense lowest 5: {ew[:5]}")
print(f"dense gap01 {ew[1]-ew[0]:.6e}  (davidson said {es[1]-es[0]:.6e})")
g0 = Vd[:, 0]
print(f"dense ground weight on D: {float((g0[inD]**2).sum()):.6f}  "
      f"|<dense g0|davidson v0>| = {abs(float(g0 @ v0)):.6f}")

# S2 of the ground
G = s2_gram_ab(B[:, 0], B[:, 1], v0, ham.nbasis)
print(f"<S2> ground: {float(G[0,0]):.4f}   (stored s2 r0 {float(np.asarray(r['s2_pyci'])[0]):.4f})")
print("FLIP_DONE", flush=True)
