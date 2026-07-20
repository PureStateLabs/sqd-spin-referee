"""Noiseless best case for IBM's method, THEIR construction verbatim:
LUCJ built exactly per their published circuit_generation_template.py
(their MO-basis FCIDUMP -> pyscf to_scf -> CCSD t1+t2 -> UCJOpSpinBalanced,
n_reps=1, chain alpha-alpha pairs, every-4th alpha-beta pairs), simulated
EXACTLY with ffsim, sampled with zero device noise, pushed through
qiskit-addon-sqd as shipped.

This supersedes _lucjfe.py (which used the localized Li-Chan-basis FCIDUMP
where a naive aufbau reference is not an RHF solution; kept as a
reference-choice observation, not IBM's recipe).

Stages (idempotent):
  M1. lucj2_ccsd.npz        -- CCSD t1,t2 on THEIR fcidump (their loader)
  M2. lucj2ship_counts.npz  -- exact LUCJ statevector -> samples
                               (+ lucj2_stats.npz: <S^2> of ansatz state,
                               top weights, coupon curve E[U(M)])
  M3. lucj2ship_<arm>.npz   -- their recovery loop, arm symm1/symm0,
                               same knobs as _sqdship/_ibmsamples
Env: SHOTS (1e6), SPB (500), NBATCH (3), DMAX (500), MAXIT (5), SEED2 (17),
     SEED_SAMPLE (7), SKIP_S2VEC (0), NT (6). Sentinel: LUCJFE2_DONE
"""
import gc
import os
import time

import numpy as np

NT = int(os.environ.get("NT", "6"))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))

from pyscf import ao2mo, cc, fci, tools  # noqa: E402
from pyscf.tools import fcidump  # noqa: E402

T0 = time.time()
BASE = "_ibm_data/sqd_data_repository-main"
FCID = f"{BASE}/integrals/2Fe-2S/fcidump_Fe2S2_MO.txt"
E_REF = -116.6056091          # their own DMRG reference
NC, NA, NB = 20, 15, 15
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
log(f"THEIR integrals loaded (ecore {ECORE:.6f})")

# ---- stage M1: CCSD exactly per their template -------------------------------
if not os.path.exists("lucj2_ccsd.npz"):
    log("stage M1: their template loader (to_scf), reference = identity-MO "
        "aufbau det (their dump is already in canonical RHF orbitals; "
        "re-running SCF finds a different, WORSE solution)")
    mf_as = tools.fcidump.to_scf(FCID)
    dm = np.zeros((NC, NC))
    for i in range(NA):
        dm[i, i] = 2.0
    vj, vk = mf_as.get_jk(mf_as.mol, dm)
    fock = H1 + vj - 0.5 * vk
    e_ref = float(np.einsum("ii->", dm @ H1)
                  + 0.5 * np.einsum("ij,ji->", dm, vj - 0.5 * vk)) + ECORE
    mf_as.mo_coeff = np.eye(NC)
    mf_as.mo_energy = np.diag(fock).copy()
    mf_as.mo_occ = np.array([2.0] * NA + [0.0] * (NC - NA))
    mf_as.converged = True
    offd = float(np.abs(fock - np.diag(np.diag(fock))).max())
    log(f"stage M1: reference det E = {e_ref:.6f} "
        f"(their table RHF -116.205816); max offdiag Fock {offd:.3e}")
    mycc = cc.CCSD(mf_as)
    mycc.max_cycle = 300
    mycc.diis_space = 12
    mycc.level_shift = 0.05
    mycc.kernel()
    log(f"stage M1: CCSD E = {mycc.e_tot:.6f} converged={mycc.converged} "
        f"(their table CCSD -116.405662)")
    np.savez("lucj2_ccsd.npz", t1=mycc.t1, t2=mycc.t2,
             e_ccsd=mycc.e_tot, e_rhf=e_ref,
             converged=bool(mycc.converged))

# ---- stage M2: exact LUCJ state -> S^2, coupon curve, samples ----------------
if not os.path.exists("lucj2ship_counts.npz"):
    import ffsim

    d1 = np.load("lucj2_ccsd.npz")
    t1, t2 = d1["t1"], d1["t2"]
    # their template, verbatim
    n_reps = 1
    alpha_alpha_indices = [(p, p + 1) for p in range(NC - 1)]
    alpha_beta_indices = [(p, p) for p in range(0, NC, 4)]
    ucj = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=t2, t1=t1, n_reps=n_reps,
        interaction_pairs=(alpha_alpha_indices, alpha_beta_indices))
    log("stage M2: UCJ built (their template: t1+t2, n_reps=1, "
        "chain aa + every-4th ab pairs)")
    ref = ffsim.hartree_fock_state(NC, (NA, NB))
    vec = ffsim.apply_unitary(ref, ucj, norb=NC, nelec=(NA, NB))
    del ref, ucj
    gc.collect()
    dima = int(fci.cistring.num_strings(NC, NA))
    dimb = int(fci.cistring.num_strings(NC, NB))
    log(f"stage M2: statevector done, dim {dima}x{dimb} = {dima*dimb:,}")

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
            log(f"stage M2: ANSATZ-STATE <S^2> = {s2vec:.6f} "
                f"(wr {wr:.6f} s2r {s2r:.4f} | wi {wi:.2e} s2i {s2i:.4f})")
        except Exception as ex:
            log(f"stage M2: S^2(vec) FAILED ({type(ex).__name__}: {ex}) "
                f"-- continuing")

    prob = np.abs(vec)
    del vec
    gc.collect()
    np.multiply(prob, prob, out=prob)
    prob = prob.ravel()
    prob /= prob.sum()

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
    log(f"stage M2: coupon curve done (E[U] at 1e6 shots ~ "
        f"{np.interp(np.log(1e6), np.log(mgrid), eu):,.0f})")

    rng = np.random.default_rng(SEED_SAMPLE)
    hits = rng.multinomial(SHOTS, prob)
    nz = np.nonzero(hits)[0]
    topidx = np.argsort(prob)[-10:][::-1]
    topw = prob[topidx]
    del prob
    gc.collect()
    log(f"stage M2: {SHOTS} shots -> {len(nz)} unique dets "
        f"(top w {topw[0]:.4f})")

    stra, strb = ffsim.addresses_to_strings(
        nz, norb=NC, nelec=(NA, NB), concatenate=False)
    a = np.asarray(stra, np.uint64)
    b = np.asarray(strb, np.uint64)
    ta, tb = ffsim.addresses_to_strings(
        topidx.copy(), norb=NC, nelec=(NA, NB), concatenate=False)
    hfmask = np.uint64((1 << NA) - 1)
    hf_is_top = bool(np.uint64(ta[0]) == hfmask and np.uint64(tb[0]) == hfmask)
    log(f"stage M2: top det is {'HF' if hf_is_top else 'NOT HF'}")
    np.savez("lucj2ship_counts.npz", a=a, b=b, n=hits[nz])
    np.savez("lucj2_stats.npz", s2vec=s2vec, nreps=n_reps, shots=SHOTS,
             nuniq=len(nz), mgrid=mgrid, eu=eu, top_w=topw,
             top_a=np.asarray(ta, np.uint64), top_b=np.asarray(tb, np.uint64),
             hf_is_top=hf_is_top)
    del hits
    gc.collect()
    log("stage M2: counts + stats saved")

# ---- stage M3: their pipeline (same knobs as _sqdship/_ibmsamples) -----------
from qiskit.primitives import BitArray  # noqa: E402
from qiskit_addon_sqd.fermion import (  # noqa: E402
    diagonalize_fermionic_hamiltonian,
)

d = np.load("lucj2ship_counts.npz")
counts = {f"{int(bb):020b}{int(aa):020b}": int(nn)
          for aa, bb, nn in zip(d["a"], d["b"], d["n"])}
bit_array = BitArray.from_counts(counts, num_bits=2 * NC)
log(f"stage M3: BitArray built ({len(counts)} unique keys, "
    f"{sum(counts.values())} shots)")

for arm, symm in [("symm1", True), ("symm0", False)]:
    out = f"lucj2ship_{arm}.npz"
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

print("LUCJFE2_DONE", flush=True)
