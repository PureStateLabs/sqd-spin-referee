"""Probe every pyci API call used by _n2xval.py on a small N2 HCI space,
then check our Engine reproduces pyci's energy on the SAME det set.
Sentinel: PROBE_OK
"""
import os
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pyci
import pyscf
import pyscf.mcscf

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


NCAS, NELEC = 22, (5, 5)
if not os.path.exists("n2_ints.npz"):
    m = pyscf.M(atom="N 0 0 -0.545; N 0 0 0.545", basis="cc-pVDZ")
    mf = pyscf.scf.RHF(m).run()
    cas = pyscf.mcscf.CASCI(mf, NCAS, 10)
    h1, ecore = cas.h1e_for_cas()
    eri = pyscf.ao2mo.full(m, mf.mo_coeff[:, 2:2 + NCAS],
                           aosym="1").reshape(NCAS, NCAS, NCAS, NCAS)
    np.savez("n2_ints.npz", h1=h1, eri=eri, ecore=float(ecore),
             e_hf=float(mf.e_tot))
d = np.load("n2_ints.npz")
h1, eri, ecore = d["h1"], d["eri"], float(d["ecore"])
log(f"ints ok; E_HF {float(d['e_hf']):.6f}")

ham = pyci.hamiltonian(ecore, h1, eri.transpose(0, 2, 1, 3))
wfn = pyci.fullci_wfn(ham.nbasis, *NELEC)
wfn.add_hartreefock_det()
op = pyci.sparse_op(ham, wfn)
e, v = op.solve(n=1, tol=1e-9)
log(f"HF det energy via pyci: {e[0]:.6f} (vs pyscf {float(d['e_hf']):.6f})")
assert abs(e[0] - float(d["e_hf"])) < 1e-6, "HF energy mismatch"

for _ in range(2):
    added = pyci.add_hci(ham, wfn, v[0], eps=5e-2)
    op.update(ham, wfn)
    e, v = op.solve(n=1)
    log(f"add_hci eps=5e-2: added={added} len={len(wfn)} E={e[0]:.6f}")

A = wfn.to_det_array()
log(f"to_det_array: shape {A.shape} dtype {A.dtype}; "
    f"has add_dets {hasattr(wfn,'add_dets')}; "
    f"has compute_enpt2 {hasattr(pyci,'compute_enpt2')}")

# rebuild a wfn from the det array (round trip used in stages 3/4)
w2 = pyci.fullci_wfn(ham.nbasis, *NELEC)
if hasattr(w2, "add_dets"):
    w2.add_dets(A)
else:
    for row in A:
        w2.add_det(row)
op2 = pyci.sparse_op(ham, w2)
e2, _ = op2.solve(n=1, tol=1e-9)
log(f"round-trip wfn: len={len(w2)} E={e2[0]:.6f} (delta "
    f"{(e2[0]-e[0])*1e6:+.2f} uHa)")
assert abs(e2[0] - e[0]) < 1e-8

# our Engine on the same dets
import _fairfight as ff

t1 = time.time()
eng = ff.Engine(h1, eri, ecore, NCAS, *NELEC)
log(f"Engine build: {time.time()-t1:.1f}s "
    f"(paulis diag {len(eng.cd)} offdiag {len(eng.co)})")
dets64 = np.unique(A[:, 0].astype(np.int64)
                   | (A[:, 1].astype(np.int64) << NCAS))
t1 = time.time()
e3, _, _ = eng.solve(dets64)
log(f"Engine solve {len(dets64)} dets: {time.time()-t1:.1f}s  E={e3:.6f} "
    f"(delta vs pyci {(e3-e[0])*1e6:+.2f} uHa)")
assert abs(e3 - e[0]) < 2e-6, "Engine vs pyci disagree"
print("PROBE_OK", flush=True)
