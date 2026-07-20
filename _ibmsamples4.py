"""[4Fe-4S] leg of the gap-1 closure: IBM's published hardware measurement
outcomes (April 2024, Job IDs 25532/25538/25561) through their as-shipped
pipeline, spin audited by their own diagnostic.

Inputs 100% theirs (zenodo.15324153):
  - counts:  experiments/4Fe-4S/.../Fe4S4_measurement_outcomes_...pkl
  - H:       integrals/4Fe-4S/fcidump_Fe4S4_MO.txt (NORB=36, NELEC=54, MS2=0)
  - E_ref:   their DMRG -327.2396369; their HCI(best) -326.793854099389
Stage I1 additionally MEASURES the raw in-sector fraction of their hardware
samples (the number their paper cites as vanishingly small) directly from
their own archive.

Stages (idempotent):
  J1. ibm4raw_counts.npz -- concatenated counts + raw statistics
  J2. ibm4ship_<arm>.npz -- their recovery loop, arm symm1/symm0
Env: SPB (500), NBATCH (3), DMAX (500), MAXIT (4), SEED2 (17), NT (6).
Sentinel: IBMSAMPLES4_DONE
"""
import os
import pickle
import time

import numpy as np

NT = int(os.environ.get("NT", "6"))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))

from pyscf import ao2mo  # noqa: E402
from pyscf.tools import fcidump  # noqa: E402

T0 = time.time()
BASE = "_ibm_data/sqd_data_repository-main"
PKL = (f"{BASE}/experiments/4Fe-4S/data/experiment_data/"
       "Fe4S4_measurement_outcomes_04_2024_Job_IDS_25532_25538_25561.pkl")
FCID = f"{BASE}/integrals/4Fe-4S/fcidump_Fe4S4_MO.txt"
E_REF = -327.2396369            # their DMRG reference
E_HCI_THEIRS = -326.793854099389
NC, NA, NB = 36, 27, 27
SPB = int(os.environ.get("SPB", "500"))
NBATCH = int(os.environ.get("NBATCH", "3"))
DMAX = int(os.environ.get("DMAX", "500"))
MAXIT = int(os.environ.get("MAXIT", "4"))
SEED2 = int(os.environ.get("SEED2", "17"))


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


fd = fcidump.read(FCID)
ECORE = float(fd["ECORE"])
H1 = fd["H1"]
ERI = ao2mo.restore(1, fd["H2"], NC)
log(f"THEIR integrals loaded (ecore {ECORE:.6f}, norb {fd['NORB']}, "
    f"nelec {fd['NELEC']})")

# ---- stage J1: their counts (their loader: single pickled dict) --------------
if not os.path.exists("ibm4raw_counts.npz"):
    with open(PKL, "rb") as f:
        raw = pickle.load(f)
    log(f"stage J1: loaded pkl ({type(raw).__name__})")
    # their values are normalized probabilities (every value = 1/len: each
    # recorded string appeared once); convert to integer counts
    nstr = len(raw)
    counts = {}
    for k, v in raw.items():
        kk = k.replace(" ", "")
        n = int(round(float(v) * nstr))
        if n:
            counts[kk] = counts.get(kk, 0) + n
    keys = list(counts.keys())
    nbits = len(keys[0])
    tot = sum(counts.values())
    log(f"stage J1: {len(keys)} unique strings, {tot} total shots, "
        f"{nbits} bits/string")
    ok = 0
    okshots = 0
    for k, v in counts.items():
        b, a = k[:nbits // 2], k[nbits // 2:]
        if a.count("1") == NA and b.count("1") == NB:
            ok += 1
            okshots += v
    log(f"stage J1: MEASURED in-sector fraction of THEIR hardware samples: "
        f"{ok} unique ({ok/len(keys):.3e}), {okshots} shots "
        f"({okshots/max(tot,1):.3e})")
    np.savez("ibm4raw_counts.npz",
             keys=np.array(keys), vals=np.array([counts[k] for k in keys]),
             nbits=nbits, tot=tot, insec_uniq=ok, insec_shots=okshots)

# ---- stage J2: their pipeline on their samples -------------------------------
from qiskit.primitives import BitArray  # noqa: E402
from qiskit_addon_sqd.fermion import (  # noqa: E402
    diagonalize_fermionic_hamiltonian,
)

d = np.load("ibm4raw_counts.npz")
counts = {str(k): int(v) for k, v in zip(d["keys"], d["vals"])}
nbits = int(d["nbits"])
assert nbits == 2 * NC, f"expected {2*NC}-bit strings, got {nbits}"
bit_array = BitArray.from_counts(counts, num_bits=nbits)
log(f"stage J2: BitArray built ({len(counts)} unique keys, "
    f"{int(d['tot'])} shots)")

for arm, symm in [("symm1", True), ("symm0", False)]:
    out = f"ibm4ship_{arm}.npz"
    if os.path.exists(out):
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
                         "err_mHa": (e_tot - E_REF) * 1e3, "s2": s2})
            log(f"{arm} it{it_box[0]} b{j}: dim {da}x{db}={da*db:,} "
                f"E {e_tot:.6f} (err {(e_tot-E_REF)*1e3:+8.3f} mHa) "
                f"THEIR S2 {s2:.3f}")

    res = diagonalize_fermionic_hamiltonian(
        H1, ERI, bit_array,
        samples_per_batch=SPB, norb=NC, nelec=(NA, NB),
        num_batches=NBATCH, max_iterations=MAXIT,
        symmetrize_spin=symm, max_dim=DMAX,
        callback=cb, seed=SEED2,
    )
    st = res.sci_state
    s2f = float(st.spin_square())
    ef = res.energy + ECORE
    log(f"{arm} BEST: E {ef:.6f} (err {(ef-E_REF)*1e3:+8.3f} mHa vs their "
        f"DMRG; {(ef-E_HCI_THEIRS)*1e3:+8.3f} vs their HCI-best) "
        f"THEIR S2 {s2f:.3f} dim {len(st.ci_strs_a)}x{len(st.ci_strs_b)}")
    np.savez(out, hist=np.array(hist, dtype=object), best_e=ef, best_s2=s2f,
             ci_strs_a=np.asarray(st.ci_strs_a, np.uint64),
             ci_strs_b=np.asarray(st.ci_strs_b, np.uint64),
             amplitudes=np.asarray(st.amplitudes))

print("IBMSAMPLES4_DONE", flush=True)
