"""Dense adjudication of the one completion checkpoint where the iterative
solver (pyci op.solve Davidson, as stored in s2audit_fe4s4.npz AND as re-run
by _degen.py) returns only one member of each degenerate pair: fe4s4
D=1106 -> 2097.

Dense diagonalization of the full 2097-dim block shows the ground IS an
exactly degenerate pair. This script computes the paper-grade, gauge-proof
mechanism row from the dense spectrum:
  dE    = E0(completed, dense) - E0(asym, dense)
  split = ew[1] - ew[0] (dense)
  p     = eigenvalues of the D-block projector compressed over the dense pair
  w_add = 1 - p_max  (weight of the block-localized gauge member on added dets)
  coup  = || P_new H v_asym ||  (gauge-free: defined on the asym ground)
  S2blk = 2x2 S^2 Gram over the dense pair (invariant eigenvalues)
Writes degen_dense.npz. Sentinel: DENSE_DONE
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pyci  # noqa: E402

from _degen import s2_gram_ab, rowkey  # noqa: E402

FE4 = ("_qsci_benchmarks/fe2s2/Active-space-model-for-Iron-Sulfur-"
       "Clusters/Fe2S2_and_Fe4S4/Fe4S4/fe4s4")
NEL = (27, 27)

ham = pyci.hamiltonian(FE4)
rows = np.load("s2audit_fe4s4.npz", allow_pickle=True)["results"]
r = next(x for x in rows if int(x["ndets"]) == 1106)
da = np.asarray(r["dets_a"], np.uint64)
db = np.asarray(r["dets_b"], np.uint64)
Bd = np.stack([da, db], 1)
comp = np.unique(np.concatenate([Bd, Bd[:, ::-1]]), axis=0)


def dense_h(dets):
    w = pyci.fullci_wfn(ham.nbasis, *NEL)
    for row_ in dets:
        w.add_det(row_)
    assert len(w) == len(dets)
    op = pyci.sparse_op(ham, w)
    n = len(w)
    H = np.zeros((n, n))
    e = np.zeros(n)
    for i in range(n):
        e[:] = 0.0
        e[i] = 1.0
        H[:, i] = op.matvec(e)
    B = w.to_det_array().reshape(n, 2)
    return H, B


# asymmetric space, dense
Ha, Ba = dense_h(Bd)
assert np.abs(Ha - Ha.T).max() == 0.0
ea, Va = np.linalg.eigh(Ha)
E_asym = float(ea[0])
print(f"asym dense E0 {E_asym:.10f} (stored e_pyci[0] "
      f"{float(np.asarray(r['e_pyci'])[0]):.10f}), asym gap01 {ea[1]-ea[0]:.3e}")

# completed space, dense
Hc, Bc = dense_h(comp)
assert np.abs(Hc - Hc.T).max() == 0.0
ec, Vc = np.linalg.eigh(Hc)
E0, split = float(ec[0]), float(ec[1] - ec[0])
gap2 = float(ec[2] - ec[1])
print(f"completed dense lowest 4: {ec[:4]}")
print(f"dE {E0-E_asym:+.3e}  split {split:.3e}  gap to root2 {gap2:.3e}")

# gauge-proof pair analysis
kk = rowkey(Bc)
inD = np.isin(kk, rowkey(Bd))
pair = Vc[:, :2]
P = pair.T @ (pair * inD[:, None])          # D-block projector over the pair
pev = np.sort(np.linalg.eigvalsh(P))[::-1]
print(f"projector spectrum over pair: [{pev[0]:.6f}, {pev[1]:.6f}]  "
      f"w_add(localized gauge) = {1-pev[0]:.3e}")

# coup on the asym ground (gauge-free): pad v_asym into completed space
order = np.argsort(kk)
posD = order[np.searchsorted(kk[order], rowkey(Ba))]
v_pad = np.zeros(len(Bc))
v_pad[posD] = Va[:, 0]
hv = Hc @ v_pad
coup = float(np.linalg.norm(hv[~inD]))
print(f"coup ||P_new H v_asym|| = {coup:.3e}")

# invariant S2 block over the dense pair
G = s2_gram_ab(Bc[:, 0], Bc[:, 1], pair, ham.nbasis)
gev = np.linalg.eigvalsh(G)
print(f"S2 block eigenvalues [{gev[0]:.4f}, {gev[1]:.4f}]  "
      f"offdiag {abs(G[0,1]):.1e}  (stored asym s2 r0 "
      f"{float(np.asarray(r['s2_pyci'])[0]):.4f})")

# iterative-solver artifact record (what op.solve returned on this space)
w2 = pyci.fullci_wfn(ham.nbasis, *NEL)
for row_ in comp:
    w2.add_det(row_)
op2 = pyci.sparse_op(ham, w2)
ei, vi = op2.solve(n=3, tol=1e-9)
ei = np.sort(np.asarray(ei))
print(f"iterative (artifact) lowest 3: {ei[:3]}  split {ei[1]-ei[0]:.3e}")

np.savez("degen_dense.npz",
         n0=len(Bd), n1=len(comp), E_asym=E_asym, E0=E0,
         dE=E0 - E_asym, split=split, gap2=gap2,
         p_spec=pev, w_add=1 - pev[0], coup=coup,
         s2blk=gev, s2off=abs(G[0, 1]),
         e_dense=ec[:6], e_iter=ei[:3], iter_split=float(ei[1] - ei[0]),
         s2_asym_r0=float(np.asarray(r["s2_pyci"])[0]))
print("DENSE_DONE", flush=True)
