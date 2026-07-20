"""Run IBM's qiskit-addon-sqd pipeline AS SHIPPED on samples drawn from the
converged [2Fe-2S] soup vector.

Question: given samples from exactly the kind of state these benchmarks
actually follow (aufbau-HCI sector root0 at 170k dets, <S^2> = 4.669,
E = -116.591986), does the shipped configuration-recovery loop -- with and
without their spin mitigation (symmetrize_spin) -- return a singlet?
S^2 is computed by THEIR OWN code (SCIState.spin_square, pyscf-backed);
subspaces are THEIR construction (Cartesian product of half-string sets).

Stages (idempotent):
  1. sqdship_vec.npz    -- re-solve the 170k-det aufbau space (pyci) for c
  2. sqdship_counts.npz -- SHOTS multinomial samples from |c|^2 as bit counts
  3. sqdship_<arm>.npz  -- their diagonalize_fermionic_hamiltonian loop,
                           arm symm1 (symmetrize_spin=True) / symm0 (False)
Env: SHOTS (1e6), SPB (500), NBATCH (3), DMAX (500), MAXIT (5), SEED2 (17),
     NT. Sentinel: SQDSHIP_DONE
"""
import os
import time

import numpy as np

NT = int(os.environ.get("NT", "6"))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))

from pyscf import ao2mo  # noqa: E402
from pyscf.tools import fcidump  # noqa: E402

T0 = time.time()
FCID = ("_qsci_benchmarks/fe2s2/Active-space-model-for-Iron-Sulfur-"
        "Clusters/Fe2S2_and_Fe4S4/Fe2S2/fe2s2")
E_EXACT = -116.6056091
NC = 20
SHOTS = int(float(os.environ.get("SHOTS", "1000000")))
SPB = int(os.environ.get("SPB", "500"))
NBATCH = int(os.environ.get("NBATCH", "3"))
DMAX = int(os.environ.get("DMAX", "500"))
MAXIT = int(os.environ.get("MAXIT", "5"))
SEED2 = int(os.environ.get("SEED2", "17"))


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


fd = fcidump.read(FCID)
ECORE = float(fd["ECORE"])
H1 = fd["H1"]
ERI = ao2mo.restore(1, fd["H2"], NC)
log(f"integrals loaded (ecore {ECORE:.6f})")

# ---- stage 1: the soup vector ------------------------------------------------
if not os.path.exists("sqdship_vec.npz"):
    import pyci
    rows = list(np.load("s2audit_fe2s2deep.npz", allow_pickle=True)["results"])
    last = rows[-1]
    log(f"stage1: re-solving aufbau space ndets={last['ndets']} "
        f"(recorded S2 {last['s2_pyci'][0]:.3f})")
    ham = pyci.hamiltonian(FCID)
    wfn = pyci.fullci_wfn(ham.nbasis, 15, 15)
    for r_ in np.stack([np.asarray(last["dets_a"], np.uint64),
                        np.asarray(last["dets_b"], np.uint64)], 1):
        wfn.add_det(r_)
    op = pyci.sparse_op(ham, wfn)
    ev, evec = op.solve(n=1, tol=1e-9)
    dd = wfn.to_det_array().reshape(len(wfn), 2)
    np.savez("sqdship_vec.npz", a=dd[:, 0], b=dd[:, 1], c=evec[0],
             e=float(ev[0]))
    log(f"stage1: E {ev[0]:.6f} (recorded {last['e_pyci'][0]:.6f}) saved")

# ---- stage 2: sample counts --------------------------------------------------
if not os.path.exists("sqdship_counts.npz"):
    d = np.load("sqdship_vec.npz")
    a, b, c = d["a"], d["b"], d["c"]
    p = c ** 2
    p /= p.sum()
    rng = np.random.default_rng(SEED2)
    hits = rng.multinomial(SHOTS, p)
    nz = np.nonzero(hits)[0]
    log(f"stage2: {SHOTS} shots -> {len(nz)} unique dets "
        f"(top w {p.max():.4f})")
    np.savez("sqdship_counts.npz", a=a[nz], b=b[nz], n=hits[nz])

# ---- stage 3: their pipeline -------------------------------------------------
from qiskit.primitives import BitArray  # noqa: E402
from qiskit_addon_sqd.fermion import (  # noqa: E402
    diagonalize_fermionic_hamiltonian,
)

d = np.load("sqdship_counts.npz")
counts = {f"{int(bb):020b}{int(aa):020b}": int(nn)
          for aa, bb, nn in zip(d["a"], d["b"], d["n"])}
bit_array = BitArray.from_counts(counts, num_bits=2 * NC)
log(f"stage3: BitArray built ({len(counts)} unique keys, "
    f"{sum(counts.values())} shots)")

SMOKE = os.environ.get("SMOKE", "0") == "1"
ARMS = [("symm1", True)] if SMOKE else [("symm1", True), ("symm0", False)]
for arm, symm in ARMS:
    out = f"sqdship_{arm}.npz"
    if not SMOKE and os.path.exists(out):
        log(f"{arm}: exists, skip")
        continue
    log(f"=== arm {arm} (symmetrize_spin={symm}) SPB={SPB} NBATCH={NBATCH} "
        f"DMAX={DMAX} MAXIT={MAXIT} ===")
    hist = []
    it_box = [0]

    def cb(results):
        it_box[0] += 1
        for j, r in enumerate(results):
            s2 = float(r.sci_state.spin_square())  # THEIR diagnostic
            da = len(r.sci_state.ci_strs_a)
            db = len(r.sci_state.ci_strs_b)
            e_tot = r.energy + ECORE
            hist.append({"iter": it_box[0], "batch": j, "dim_a": da,
                         "dim_b": db, "e_tot": e_tot,
                         "err_mHa": (e_tot - E_EXACT) * 1e3, "s2": s2})
            log(f"{arm} it{it_box[0]} b{j}: dim {da}x{db}={da*db:,} "
                f"E {e_tot:.6f} (err {(e_tot-E_EXACT)*1e3:+8.3f} mHa) "
                f"THEIR S2 {s2:.3f}")

    res = diagonalize_fermionic_hamiltonian(
        H1, ERI, bit_array,
        samples_per_batch=SPB, norb=NC, nelec=(15, 15),
        num_batches=NBATCH, max_iterations=MAXIT,
        symmetrize_spin=symm, max_dim=DMAX,
        callback=cb, seed=SEED2,
    )
    s2f = float(res.sci_state.spin_square())
    ef = res.energy + ECORE
    log(f"{arm} BEST: E {ef:.6f} (err {(ef-E_EXACT)*1e3:+8.3f} mHa) "
        f"THEIR S2 {s2f:.3f} dim {len(res.sci_state.ci_strs_a)}x"
        f"{len(res.sci_state.ci_strs_b)}")
    if not SMOKE:
        np.savez(out, hist=np.array(hist, dtype=object),
                 best_e=ef, best_s2=s2f)

print("SQDSHIP_DONE", flush=True)
