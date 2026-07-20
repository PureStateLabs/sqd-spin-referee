"""The zero-of-ours leg: IBM's OWN 58,644,964 noiseless samples from IBM's
OWN variationally optimized [2Fe-2S] LUCJ circuit (published in their
numerics/2Fe-2S_optimal_circuits directory), pushed through IBM's OWN
as-shipped pipeline in IBM's OWN MO basis, spin read by IBM's OWN diagnostic.
Nothing in the loop is our construction.

Their artifacts (zenodo.15324153):
  samples:  numerics/.../samples_from_circuit/samples_58644964_partition_*.npy
            (bool matrix; row = configuration; first half alpha, last beta;
            row 0 documented as the Hartree-Fock string -- packing self-test)
  circuit:  numerics/.../circuit_parameters/optimal_params.npy
            (UCJOpSpinBalanced.from_parameters, n_reps=1, chain aa pairs,
            ab pairs (0,0),(4,4),(8,8),(12,12),(16,16),
            with_final_orbital_rotation=True)
  basis:    integrals/2Fe-2S/fcidump_Fe2S2_MO.txt (== their h1e/h2e npy)
  E_ref:    their DMRG -116.6056091

Stages (idempotent):
  O1. lucjopt_counts.npz + lucjopt_stats.npz
      -- pack+aggregate their samples; HF-row self-test; total/unique/HF
         weight/in-sector; empirical U(M) at prefix budgets
  O2. lucjopt_state.npz [env DO_STATE=1, needs ~10GB alone]
      -- rebuild their optimal state from their params: <S^2> of the state,
         closed-form coupon curve, cross-check vs empirical U(M)
  O3. lucjoptship_<arm>.npz -- their recovery loop, symm1/symm0, same knobs
      as _sqdship/_ibmsamples/_lucjfe2
Env: SPB (500), NBATCH (3), DMAX (500), MAXIT (5), SEED2 (17), NT (6),
     DO_STATE (0). Sentinel: LUCJOPT_DONE
"""
import gc
import os
import time

import numpy as np

NT = int(os.environ.get("NT", "6"))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))

from pyscf import ao2mo  # noqa: E402
from pyscf.tools import fcidump  # noqa: E402

T0 = time.time()
BASE = "_ibm_data/sqd_data_repository-main"
NUM = f"{BASE}/numerics/2Fe-2S_optimal_circuits"
FCID = f"{BASE}/integrals/2Fe-2S/fcidump_Fe2S2_MO.txt"
E_REF = -116.6056091
NC, NA, NB = 20, 15, 15
SPB = int(os.environ.get("SPB", "500"))
NBATCH = int(os.environ.get("NBATCH", "3"))
DMAX = int(os.environ.get("DMAX", "500"))
MAXIT = int(os.environ.get("MAXIT", "5"))
SEED2 = int(os.environ.get("SEED2", "17"))
DO_STATE = os.environ.get("DO_STATE", "0") == "1"


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


# ---- stage O1: their samples, packed and aggregated --------------------------
if not os.path.exists("lucjopt_counts.npz"):
    w = (np.uint64(1) << np.arange(20, dtype=np.uint64))
    keys_parts = []
    hf = np.uint64((1 << NA) - 1)
    for i in range(40):
        part = np.load(f"{NUM}/samples_from_circuit/"
                       f"samples_58644964_partition_{i}.npy")
        # within each half their columns run orbital 19 -> 0 (row 0 is the
        # documented HF string; the reversal makes it pack to 0x7fff|0x7fff)
        pa = np.asarray(part[:, :20][:, ::-1], np.uint64) @ w
        pb = np.asarray(part[:, 20:][:, ::-1], np.uint64) @ w
        if i == 0:
            assert pa[0] == hf and pb[0] == hf, (
                f"HF self-test FAILED: row0 a={int(pa[0]):020b} "
                f"b={int(pb[0]):020b}")
            log("stage O1: HF packing self-test PASSED (their row 0)")
        keys_parts.append((pb << np.uint64(20)) | pa)
        if i % 10 == 9:
            log(f"stage O1: packed partition {i+1}/40")
    keys = np.concatenate(keys_parts)
    del keys_parts
    gc.collect()
    tot = len(keys)
    log(f"stage O1: total samples {tot:,} (their filename says 58,644,964)")
    # empirical unique-vs-shots at prefix budgets (their sample order)
    emp_M, emp_U = [], []
    for M in [100_000, 1_000_000, 10_000_000, tot]:
        if M <= tot:
            emp_M.append(M)
            emp_U.append(int(len(np.unique(keys[:M]))))
    log(f"stage O1: empirical U(M): {list(zip(emp_M, emp_U))}")
    uk, cnt = np.unique(keys, return_counts=True)
    del keys
    gc.collect()
    a = uk & np.uint64((1 << 20) - 1)
    b = uk >> np.uint64(20)
    hfw = 0.0
    hfsel = (a == hf) & (b == hf)
    if hfsel.any():
        hfw = float(cnt[hfsel][0]) / tot
    pop = np.array([bin(int(x)).count("1") for x in uk], np.int32)
    insec = int(((pop == NA + NB)
                 & (np.array([bin(int(x)).count("1") for x in a]) == NA)).sum())
    log(f"stage O1: {len(uk):,} unique; HF weight {hfw:.4f}; "
        f"in-sector unique {insec:,}/{len(uk):,}")
    np.savez("lucjopt_counts.npz", a=a, b=b, n=cnt)
    np.savez("lucjopt_stats.npz", tot=tot, nuniq=len(uk), hf_weight=hfw,
             insec_uniq=insec, emp_M=np.array(emp_M), emp_U=np.array(emp_U))

# ---- stage O2: their optimal state, rebuilt from their params ----------------
if DO_STATE and not os.path.exists("lucjopt_state.npz"):
    import ffsim
    from pyscf import fci

    p_flat = np.load(f"{NUM}/circuit_parameters/optimal_params.npy")
    pairs_aa = [(p, p + 1) for p in range(NC - 1)]
    pairs_ab = [(0, 0), (4, 4), (8, 8), (12, 12), (16, 16)]
    ucj = ffsim.UCJOpSpinBalanced.from_parameters(
        params=p_flat, norb=NC, n_reps=1,
        interaction_pairs=(pairs_aa, pairs_ab),
        with_final_orbital_rotation=True)
    ref = ffsim.hartree_fock_state(NC, (NA, NB))
    vec = ffsim.apply_unitary(ref, ucj, norb=NC, nelec=(NA, NB))
    del ref, ucj
    gc.collect()
    dima = int(fci.cistring.num_strings(NC, NA))
    dimb = int(fci.cistring.num_strings(NC, NB))
    log(f"stage O2: THEIR optimal state rebuilt, dim {dima*dimb:,}")
    s2vec = np.nan
    try:
        vm = vec.reshape(dima, dimb)
        re = np.ascontiguousarray(vm.real)
        wr = float(np.vdot(re, re).real)
        s2r = float(fci.spin_op.spin_square(
            re / np.sqrt(max(wr, 1e-300)), NC, (NA, NB))[0])
        del re
        gc.collect()
        im = np.ascontiguousarray(vm.imag)
        wi = float(np.vdot(im, im).real)
        s2i = 0.0
        if wi > 1e-14:
            s2i = float(fci.spin_op.spin_square(
                im / np.sqrt(wi), NC, (NA, NB))[0])
        del im, vm
        gc.collect()
        s2vec = (wr * s2r + wi * s2i) / (wr + wi)
        log(f"stage O2: THEIR OPTIMAL ANSATZ <S^2> = {s2vec:.6f}")
    except Exception as ex:
        log(f"stage O2: S^2 failed ({type(ex).__name__}) -- continuing")
    prob = np.abs(vec)
    del vec
    gc.collect()
    np.multiply(prob, prob, out=prob)
    prob = prob.ravel()
    prob /= prob.sum()
    lp = np.log1p(-np.minimum(prob, 1.0 - 1e-16))
    st = np.load("lucjopt_stats.npz")
    mgrid = np.unique(np.round(np.logspace(3, 10, 36)).astype(np.int64))
    eu = np.empty(len(mgrid))
    ntot = len(prob)
    for j, M in enumerate(mgrid):
        acc = 0.0
        for s in range(0, ntot, 20_000_000):
            acc += float(np.exp(M * lp[s:s + 20_000_000]).sum())
        eu[j] = ntot - acc
    # cross-check their published samples against their published circuit
    dev = []
    for M, U in zip(st["emp_M"], st["emp_U"]):
        pred = float(np.interp(np.log(M), np.log(mgrid), eu))
        dev.append(abs(pred - U) / U)
        log(f"stage O2: U({M:,}) their samples {U:,} vs their circuit "
            f"closed-form {pred:,.0f} (dev {dev[-1]:.3f})")
    np.savez("lucjopt_state.npz", s2vec=s2vec, mgrid=mgrid, eu=eu,
             emp_dev=np.array(dev))
    del prob, lp
    gc.collect()

# ---- stage O3: their pipeline on their samples -------------------------------
from qiskit.primitives import BitArray  # noqa: E402
from qiskit_addon_sqd.fermion import (  # noqa: E402
    diagonalize_fermionic_hamiltonian,
)

fd = fcidump.read(FCID)
ECORE = float(fd["ECORE"])
H1 = fd["H1"]
ERI = ao2mo.restore(1, fd["H2"], NC)
log(f"THEIR integrals loaded (ecore {ECORE:.6f})")

d = np.load("lucjopt_counts.npz")
a_, b_, n_ = d["a"], d["b"], d["n"]
SUBN = int(float(os.environ.get("SUBN", "2000000")))
if len(a_) > 5_000_000:
    # a 58.6M-key python dict OOM-kills a 14GB VM; their pipeline draws
    # SPB-sized batches anyway, so a uniform SUBN-subsample of the set is
    # statistically equivalent input at these knobs
    rng = np.random.default_rng(11)
    idx = rng.choice(len(a_), size=SUBN, replace=False)  # set entries are
    a_, b_, n_ = a_[idx], b_[idx], np.ones(SUBN, np.int64)  # all unit-count
    log(f"stage O3: subsampled {SUBN:,} of {len(d['a']):,} set entries "
        f"(memory guard; seed 11)")
counts = {f"{int(bb):020b}{int(aa):020b}": int(nn)
          for aa, bb, nn in zip(a_, b_, n_)}
bit_array = BitArray.from_counts(counts, num_bits=2 * NC)
log(f"stage O3: BitArray built ({len(counts):,} unique keys, "
    f"{sum(counts.values()):,} shots)")

for arm, symm in [("symm1", True), ("symm0", False)]:
    out = f"lucjoptship_{arm}.npz"
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
    log(f"{arm} BEST: E {ef:.6f} (err {(ef-E_REF)*1e3:+8.3f} mHa) "
        f"THEIR S2 {s2f:.3f} dim {len(st.ci_strs_a)}x{len(st.ci_strs_b)}")
    np.savez(out, hist=np.array(hist, dtype=object), best_e=ef, best_s2=s2f,
             ci_strs_a=np.asarray(st.ci_strs_a, np.uint64),
             ci_strs_b=np.asarray(st.ci_strs_b, np.uint64),
             amplitudes=np.asarray(st.amplitudes))

print("LUCJOPT_DONE", flush=True)
