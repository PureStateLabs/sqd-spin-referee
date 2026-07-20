"""Sanity probe for the S2 singlet-floor result: every diagonal (s,s) product
determinant in the captured spaces is a closed-shell S=0 eigenstate, so the
minimum of their diagonal energies is an UPPER bound on the true singlet-
sector minimum. If that bound were below the eigsh large-lambda endpoint,
eigsh missed the singlet ground; if it is at/above the endpoint, the
+2.9-Ha floor is consistent. Read-only; NT=1."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from pyscf import ao2mo, fci
from pyscf.tools import fcidump

FCID = ("_qsci_benchmarks/fe2s2/Active-space-model-for-Iron-Sulfur-"
        "Clusters/Fe2S2_and_Fe4S4/Fe2S2/fe2s2")
E_EXACT = -116.6056091
NC, NELEC = 20, (15, 15)
fd = fcidump.read(FCID)
H1 = fd["H1"]
ERI = ao2mo.restore(1, fd["H2"], NC)

z = np.load("sqdpen_spaces.npz", allow_pickle=True)
cap = list(z["cap"])
it1 = min((r for r in cap if r["iter"] == 1), key=lambda r: r["e_tot"])
best = min(cap, key=lambda r: r["e_tot"])
for tag, r in [("it1", it1), ("it4", best)]:
    sa = np.asarray(r["strs_a"], dtype=np.int64)
    sb = np.asarray(r["strs_b"], dtype=np.int64)
    na, nb = len(sa), len(sb)
    myci = fci.selected_ci.SelectedCI()
    hdiag = myci.make_hdiag(H1, ERI, (sa, sb), NC, NELEC, compress=True)
    hd = np.asarray(hdiag).reshape(na, nb)
    # diagonal (s,s) entries: strings sorted identically (symmetrized space)
    assert np.array_equal(sa, sb)
    cs = np.diagonal(hd)
    k = int(np.argmin(cs))
    print(f"{tag}: {na}x{nb}; closed-shell dets {len(cs)}")
    print(f"  min closed-shell diag: {cs[k]:.6f} Ha "
          f"(err {(cs[k]-E_EXACT)*1e3:+.1f} mHa) at string index {k} "
          f"(string {int(sa[k]):#07x})")
    print(f"  next 4: {np.sort(cs)[1:5].round(4)}")
    print(f"  global min of full hdiag: {hd.min():.6f} "
          f"(err {(hd.min()-E_EXACT)*1e3:+.1f} mHa)")
    aufbau = int("1" * NELEC[0], 2)
    if aufbau in set(int(x) for x in sa):
        i = int(np.where(sa == aufbau)[0][0])
        print(f"  aufbau closed-shell diag: {hd[i, i]:.6f} "
              f"(err {(hd[i, i]-E_EXACT)*1e3:+.1f} mHa)")
print("PROBE_DONE")
