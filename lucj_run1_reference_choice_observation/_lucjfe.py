"""Close the 'not our samples' gap: run IBM's pipeline on samples from IBM's
OWN ANSATZ for [2Fe-2S] -- the noiseless best case for their method.

Build the LUCJ(t2-CCSD) state for the fe2s2 (30e,20o) FCIDUMP exactly the way
the SQD papers parameterize it (CCSD t2 amplitudes -> UCJOpSpinBalanced,
unrestricted J = a strict superset of the hardware-local LUCJ, i.e. MORE
expressive than what ran on the device), simulate it EXACTLY with ffsim
(2.4e8-amplitude sector vector), measure <S^2> of the ansatz state itself,
draw SHOTS multinomial samples (zero device noise), and push them through
qiskit-addon-sqd's diagonalize_fermionic_hamiltonian AS SHIPPED with the same
knobs as the soup-sample run (_sqdship.py) for apples-to-apples.

Stages (idempotent):
  L1. lucj_ccsd.npz       -- CCSD t2 in the FCIDUMP orbital basis (fake-RHF shell)
  L2. lucjship_counts.npz -- exact LUCJ statevector -> samples (+ lucj_stats.npz:
                             <S^2> of the ansatz vector, HF/top weights, unique-det
                             coupon curve E[U(M)] of the LUCJ distribution)
  L3. lucjship_<arm>.npz  -- THEIR recovery loop, arm symm1/symm0

Env: NREPS (1), LOCAL (0: unrestricted J; 1: heavy-hex-style local pairs),
     SHOTS (1e6), SPB (500), NBATCH (3), DMAX (500), MAXIT (5), SEED2 (17),
     SEED_SAMPLE (7), SKIP_S2VEC (0), NT (6). Sentinel: LUCJFE_DONE
"""
import gc
import os
import time

import numpy as np

NT = int(os.environ.get("NT", "6"))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))

from pyscf import ao2mo, cc, fci, gto, scf  # noqa: E402
from pyscf.tools import fcidump  # noqa: E402

T0 = time.time()
FCID = ("_qsci_benchmarks/fe2s2/Active-space-model-for-Iron-Sulfur-"
        "Clusters/Fe2S2_and_Fe4S4/Fe2S2/fe2s2")
E_EXACT = -116.6056091
NC, NA, NB = 20, 15, 15
NREPS = int(os.environ.get("NREPS", "1"))
LOCAL = os.environ.get("LOCAL", "0") == "1"
SHOTS = int(float(os.environ.get("SHOTS", "1000000")))
SPB = int(os.environ.get("SPB", "500"))
NBATCH = int(os.environ.get("NBATCH", "3"))
DMAX = int(os.environ.get("DMAX", "500"))
MAXIT = int(os.environ.get("MAXIT", "5"))
SEED2 = int(os.environ.get("SEED2", "17"))
SEED_SAMPLE = int(os.environ.get("SEED_SAMPLE", "7"))
SKIP_S2VEC = os.environ.get("SKIP_S2VEC", "0") == "1"


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


fd = fcidump.read(FCID)
ECORE = float(fd["ECORE"])
H1 = fd["H1"]
ERI = ao2mo.restore(1, fd["H2"], NC)
log(f"integrals loaded (ecore {ECORE:.6f})")

# ---- stage L1: CCSD t2 in the FCIDUMP basis ---------------------------------
if not os.path.exists("lucj_ccsd.npz"):
    log("stage L1: hand-built RHF shell + CCSD (fairfight recipe)")
    mol = gto.M(verbose=0)
    mol.nelectron = NA + NB
    mol.spin = 0
    mol.incore_anyway = True
    mol.max_memory = 8000
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *a: H1
    mf.get_ovlp = lambda *a: np.eye(NC)
    mf._eri = ao2mo.restore(8, np.asarray(fd["H2"]), NC)
    dm = np.zeros((NC, NC))
    for i in range(NA):
        dm[i, i] = 2.0
    vj, vk = mf.get_jk(mol, dm)
    fock = H1 + vj - 0.5 * vk
    mf.mo_coeff = np.eye(NC)
    mf.mo_energy = np.diag(fock).copy()
    mf.mo_occ = np.array([2.0] * NA + [0.0] * (NC - NA))
    mf.converged = True
    note = "ccsd"
    mycc = cc.CCSD(mf)
    mycc.max_cycle = 200
    try:
        mycc.kernel()
        t2 = mycc.t2
        if not mycc.converged:
            note = "ccsd-UNCONVERGED"
    except Exception as ex:
        note = f"ccsd-FAILED({type(ex).__name__})->mp2"
        t2 = mycc.init_amps()[2]
    np.savez("lucj_ccsd.npz", t2=t2, note=note)
    log(f"stage L1: t2 {t2.shape} saved (note={note})")

# ---- stage L2: exact LUCJ state -> S^2, coupon curve, samples ----------------
if not os.path.exists("lucjship_counts.npz"):
    import ffsim

    d1 = np.load("lucj_ccsd.npz", allow_pickle=True)
    t2, note = d1["t2"], str(d1["note"])
    pairs = None
    if LOCAL:
        pairs = ([(p, p + 1) for p in range(NC - 1)],
                 [(p, p) for p in range(NC)])
    ucj = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2, n_reps=NREPS, interaction_pairs=pairs)
    log(f"stage L2: UCJ built ({note}, n_reps={NREPS}, "
        f"pairs={'local' if LOCAL else 'unrestricted'})")
    ref = ffsim.hartree_fock_state(NC, (NA, NB))
    vec = ffsim.apply_unitary(ref, ucj, norb=NC, nelec=(NA, NB))
    del ref, ucj
    gc.collect()
    dima = int(fci.cistring.num_strings(NC, NA))
    dimb = int(fci.cistring.num_strings(NC, NB))
    log(f"stage L2: statevector done, dim {dima}x{dimb} = {dima*dimb:,}")

    s2vec = np.nan
    if not SKIP_S2VEC:
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
            log(f"stage L2: ANSATZ-STATE <S^2> = {s2vec:.6f} "
                f"(wr {wr:.6f} s2r {s2r:.4f} | wi {wi:.2e} s2i {s2i:.4f})")
        except Exception as ex:
            log(f"stage L2: S^2(vec) FAILED ({type(ex).__name__}: {ex}) "
                f"-- continuing")

    prob = np.abs(vec)
    del vec
    gc.collect()
    np.multiply(prob, prob, out=prob)
    prob = prob.ravel()
    prob /= prob.sum()

    # coupon curve of the LUCJ distribution: E[U(M)] = sum_i 1-(1-p_i)^M
    lp = np.log1p(-np.minimum(prob, 1.0 - 1e-16))
    mgrid = np.unique(np.round(np.logspace(3, 10, 36)).astype(np.int64))
    eu = np.empty(len(mgrid))
    ntot = len(prob)
    CH = 20_000_000
    for j, M in enumerate(mgrid):
        acc = 0.0
        for s in range(0, ntot, CH):
            acc += float(np.exp(M * lp[s:s + CH]).sum())
        eu[j] = ntot - acc
    del lp
    gc.collect()
    log(f"stage L2: coupon curve done (E[U] at 1e6 shots ~ "
        f"{np.interp(np.log(1e6), np.log(mgrid), eu):,.0f})")

    rng = np.random.default_rng(SEED_SAMPLE)
    hits = rng.multinomial(SHOTS, prob)
    nz = np.nonzero(hits)[0]
    topidx = np.argsort(prob)[-10:][::-1]
    topw = prob[topidx]
    del prob
    gc.collect()
    log(f"stage L2: {SHOTS} shots -> {len(nz)} unique dets "
        f"(top w {topw[0]:.4f})")

    stra, strb = ffsim.addresses_to_strings(
        nz, norb=NC, nelec=(NA, NB), concatenate=False)
    a = np.asarray(stra, np.uint64)
    b = np.asarray(strb, np.uint64)
    ta, tb = ffsim.addresses_to_strings(
        topidx.copy(), norb=NC, nelec=(NA, NB), concatenate=False)
    hfmask = np.uint64((1 << NA) - 1)
    hf_is_top = bool(np.uint64(ta[0]) == hfmask and np.uint64(tb[0]) == hfmask)
    if not hf_is_top:
        log(f"stage L2: NOTE top-weight det is NOT HF "
            f"(a {int(ta[0]):020b} b {int(tb[0]):020b}) -- multireference, ok")
    np.savez("lucjship_counts.npz", a=a, b=b, n=hits[nz])
    np.savez("lucj_stats.npz", s2vec=s2vec, note=note, nreps=NREPS,
             local=LOCAL, shots=SHOTS, nuniq=len(nz), mgrid=mgrid, eu=eu,
             top_w=topw, top_a=np.asarray(ta, np.uint64),
             top_b=np.asarray(tb, np.uint64), hf_is_top=hf_is_top)
    del hits
    gc.collect()
    log("stage L2: counts + stats saved")

# ---- stage L3: their pipeline (verbatim _sqdship.py stage 3) -----------------
from qiskit.primitives import BitArray  # noqa: E402
from qiskit_addon_sqd.fermion import (  # noqa: E402
    diagonalize_fermionic_hamiltonian,
)

d = np.load("lucjship_counts.npz")
counts = {f"{int(bb):020b}{int(aa):020b}": int(nn)
          for aa, bb, nn in zip(d["a"], d["b"], d["n"])}
bit_array = BitArray.from_counts(counts, num_bits=2 * NC)
log(f"stage L3: BitArray built ({len(counts)} unique keys, "
    f"{sum(counts.values())} shots)")

for arm, symm in [("symm1", True), ("symm0", False)]:
    out = f"lucjship_{arm}.npz"
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
                         "err_mHa": (e_tot - E_EXACT) * 1e3, "s2": s2})
            log(f"{arm} it{it_box[0]} b{j}: dim {da}x{db}={da*db:,} "
                f"E {e_tot:.6f} (err {(e_tot-E_EXACT)*1e3:+8.3f} mHa) "
                f"THEIR S2 {s2:.3f}")

    res = diagonalize_fermionic_hamiltonian(
        H1, ERI, bit_array,
        samples_per_batch=SPB, norb=NC, nelec=(NA, NB),
        num_batches=NBATCH, max_iterations=MAXIT,
        symmetrize_spin=symm, max_dim=DMAX,
        callback=cb, seed=SEED2,
    )
    s2f = float(res.sci_state.spin_square())
    ef = res.energy + ECORE
    log(f"{arm} BEST: E {ef:.6f} (err {(ef-E_EXACT)*1e3:+8.3f} mHa) "
        f"THEIR S2 {s2f:.3f} dim {len(res.sci_state.ci_strs_a)}x"
        f"{len(res.sci_state.ci_strs_b)}")
    np.savez(out, hist=np.array(hist, dtype=object), best_e=ef, best_s2=s2f)

print("LUCJFE_DONE", flush=True)
