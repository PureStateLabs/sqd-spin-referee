"""THE gap-1 closure: run IBM's as-shipped pipeline on IBM's OWN PUBLISHED
HARDWARE SAMPLES for [2Fe-2S] and audit the spin of what comes out.

Inputs are 100% theirs (Sci Adv 2025 data repository, zenodo.15324153):
  - counts:  experiments/2Fe-2S/.../2023-12-22T19-07-32.839270_results.npy
             (list of per-batch count dicts; concatenated per their loader)
  - H:       integrals/2Fe-2S/fcidump_Fe2S2_MO.txt (NORB=20, NELEC=30, MS2=0)
  - E_ref:   their own DMRG line in classical_methods_energies.txt
             = -116.6056091 (identical to the Li-Chan-basis exact value:
             same active space, unitary orbital rotation)
Pipeline is 100% theirs: qiskit_addon_sqd.diagonalize_fermionic_hamiltonian
with the SAME knobs as our soup-sample (_sqdship.py) and noiseless-LUCJ
(_lucjfe.py) runs -> apples-to-apples three-column table. S^2 is THEIR OWN
SCIState.spin_square() at every iteration; final states saved for the
independent S^2-Gram cross-instrument audit.

Stages (idempotent):
  I1. ibmraw_counts.npz    -- concatenated counts + raw-sample statistics
                              (shots, unique, in-sector fraction)
  I2. ibmship_<arm>.npz    -- their recovery loop, arm symm1/symm0
                              (+ final ci_strs/amplitudes for the Gram audit)
Env: SPB (500), NBATCH (3), DMAX (500), MAXIT (5), SEED2 (17), NT (6).
Sentinel: IBMSAMPLES_DONE
"""
import os
import time

import numpy as np

NT = int(os.environ.get("NT", "6"))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))

from pyscf import ao2mo  # noqa: E402
from pyscf.tools import fcidump  # noqa: E402

T0 = time.time()
BASE = "_ibm_data/sqd_data_repository-main"
NPY = (f"{BASE}/experiments/2Fe-2S/data/experiment_data/"
       "2023-12-22T19-07-32.839270_results.npy")
FCID = f"{BASE}/integrals/2Fe-2S/fcidump_Fe2S2_MO.txt"
E_REF = -116.6056091          # THEIR DMRG reference (== Li-Chan-basis exact)
E_HCI_THEIRS = -116.60540927  # their "HCI (best)" line
NC, NA, NB = 20, 15, 15
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
log(f"THEIR integrals loaded (ecore {ECORE:.6f}, norb {fd['NORB']}, "
    f"nelec {fd['NELEC']})")

# ---- stage I1: their counts, concatenated per their own loader ---------------
if not os.path.exists("ibmraw_counts.npz"):
    data = np.load(NPY, allow_pickle=True)
    log(f"stage I1: loaded {NPY.split('/')[-1]} -- {len(data)} batch dicts")
    # their values are per-batch PROBABILITIES (each batch dict sums to 1.0;
    # 8192 shots/batch, 300 batches = their documented 2.4576e6 shots);
    # convert to integer shot counts via round(v * 8192) per batch
    SPB_THEIRS = 8192
    counts = {}
    for ib, cts in enumerate(data):
        bsum = 0
        for k, v in cts.items():
            kk = k.replace(" ", "")
            n = int(round(float(v) * SPB_THEIRS))
            bsum += n
            if n:
                counts[kk] = counts.get(kk, 0) + n
        assert abs(bsum - SPB_THEIRS) <= SPB_THEIRS // 100, (
            f"batch {ib}: rounded shots {bsum} != {SPB_THEIRS}")
    keys = list(counts.keys())
    nbits = len(keys[0])
    assert all(len(k) == nbits for k in keys[:1000]), "mixed key lengths"
    tot = sum(counts.values())
    log(f"stage I1: {len(keys)} unique strings, {tot} total shots "
        f"(their protocol: 300x8192 = 2,457,600), {nbits} bits/string")
    # raw-sample statistics: in-sector fraction (right half = alpha per their
    # convention; both halves must have exactly NA/NB ones)
    ok = 0
    okshots = 0
    for k, v in counts.items():
        b, a = k[:nbits // 2], k[nbits // 2:]
        if a.count("1") == NA and b.count("1") == NB:
            ok += 1
            okshots += v
    log(f"stage I1: in-sector: {ok} unique ({ok/len(keys):.3e}), "
        f"{okshots} shots ({okshots/max(tot,1):.3e})")
    np.savez("ibmraw_counts.npz",
             keys=np.array(keys), vals=np.array([counts[k] for k in keys]),
             nbits=nbits, tot=tot, insec_uniq=ok, insec_shots=okshots)

# ---- stage I2: their pipeline on their samples -------------------------------
from qiskit.primitives import BitArray  # noqa: E402
from qiskit_addon_sqd.fermion import (  # noqa: E402
    diagonalize_fermionic_hamiltonian,
)

d = np.load("ibmraw_counts.npz")
counts = {str(k): int(v) for k, v in zip(d["keys"], d["vals"])}
nbits = int(d["nbits"])
assert nbits == 2 * NC, f"expected {2*NC}-bit strings, got {nbits}"
bit_array = BitArray.from_counts(counts, num_bits=nbits)
log(f"stage I2: BitArray built ({len(counts)} unique keys, "
    f"{int(d['tot'])} shots)")

for arm, symm in [("symm1", True), ("symm0", False)]:
    out = f"ibmship_{arm}.npz"
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

print("IBMSAMPLES_DONE", flush=True)
