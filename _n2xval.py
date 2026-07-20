"""N2 (10e,22o) cross-validation against Reinholdt et al. arXiv:2501.07231,
using THEIR public protocol (github.com/peter-reinholdt/qsci-benchmarks) and
THEIR HCI engine (PyCI) wherever possible.

Published anchors to reproduce (paper Fig. 2, R=1.09 A):
  HCI  eps=1e-3      -> ~15k dets  -> ~1e-2 Ha error vs CASCI
  QSCI 1e6 samples   -> ~15k dets  -> ~1e-2 Ha
  QSCI 1e8 samples   -> ~182k dets -> ~1 mHa
On-box substitutions (300 GB CASCI reference infeasible here):
  reference = deepest-HCI E_var + PT2; QSCI samples the deepest-HCI |c|^2
  (captured-mass logged) instead of the exact CASCI vector -- both proxies
  bias in QSCI's FAVOR (its distribution is more compact than exact).

Stages (idempotent -- each skipped if its .npz exists):
  1 n2_ints.npz  their integral protocol verbatim (N2_cas.py minus the CASCI)
  2 n2_hci.npz   THEIR HCI: pyci add_hci eps-ladder 1e-1 -> EPS_END (0.794x),
                 checkpoint (eps, ndets, E); det lists kept at <=SAVE_DETS
  3 n2_ref.npz   reference: deepest HCI + PT2 (pyci ENPT2 if present, else
                 our Engine EN-PT2 on the top-K dets)
  4 n2_qsci.npz  THEIR QSCI arm: sample |c|^2 (+ their spin-partner freebie),
                 N in 1e5..1e7 (QBIG=1 adds 1e8), diagonalize in pyci
  5 n2_ours.npz  our Engine hci_hf growth to 15k on the same integrals +
                 Engine re-solve of pyci's eps~1e-3 det set (H cross-check)
Env: EPS_END (2.5e-4), DET_CAP (400000), QBIG (0), CHUNK (30000), NT (4),
     SKIPOURS (0).  Sentinel: N2XVAL_DONE
"""
import os
import time

import numpy as np

NT = int(os.environ.get("NT", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
from pyscf import lib  # noqa: E402

lib.num_threads(NT)
import pyci  # noqa: E402
import pyscf  # noqa: E402
import pyscf.mcscf  # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


EPS_END = float(os.environ.get("EPS_END", "2.5e-4"))
DET_CAP = int(os.environ.get("DET_CAP", "400000"))
QBIG = os.environ.get("QBIG", "0") == "1"
SAVE_DETS = 40000
NCAS, NELEC = 22, (5, 5)
RNG = np.random.default_rng(7)

# geometry + working-dir parameterization (RGEOM in Angstrom; artifacts land
# in WORKDIR so multiple geometries don't collide)
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RGEOM = float(os.environ.get("RGEOM", "1.09"))
WORKDIR = os.environ.get("WORKDIR", "")
if WORKDIR:
    os.makedirs(WORKDIR, exist_ok=True)
    os.chdir(WORKDIR)

# ---------------- stage 1: their integrals (N2_cas.py verbatim) -------------
if not os.path.exists("n2_ints.npz"):
    m = pyscf.M(atom=f"N 0 0 {-RGEOM/2}; N 0 0 {RGEOM/2}", basis="cc-pVDZ")
    mf = pyscf.scf.RHF(m).run()
    cas = pyscf.mcscf.CASCI(mf, NCAS, 10)
    h1, ecore = cas.h1e_for_cas()
    eri = pyscf.ao2mo.full(m, mf.mo_coeff[:, 2:2 + NCAS],
                           aosym="1").reshape(NCAS, NCAS, NCAS, NCAS)
    np.savez("n2_ints.npz", h1=h1, eri=eri, ecore=float(ecore),
             e_hf=float(mf.e_tot))
    log(f"stage1: ints saved; E_HF {mf.e_tot:.6f}")
d = np.load("n2_ints.npz")
h1, eri, ecore = d["h1"], d["eri"], float(d["ecore"])
ham = pyci.hamiltonian(ecore, h1, eri.transpose(0, 2, 1, 3))
log(f"pyci hamiltonian ready (nbasis {ham.nbasis}); E_HF {float(d['e_hf']):.6f}")

# ---------------- stage 2: THEIR HCI ladder (run_hci_from_npy.py) ------------
if not os.path.exists("n2_hci.npz"):
    wfn = pyci.fullci_wfn(ham.nbasis, *NELEC)
    wfn.add_hartreefock_det()
    op = pyci.sparse_op(ham, wfn)
    e_vals, e_vecs = op.solve(n=1, tol=1e-9)
    eps, rows, ckpt = 1e-1, [], {}
    while eps >= EPS_END and len(wfn) <= DET_CAP:
        added = True
        while added and len(wfn) <= DET_CAP:
            added = pyci.add_hci(ham, wfn, e_vecs[0], eps=eps)
            op.update(ham, wfn)
            e_vals, e_vecs = op.solve(n=1)
        rows.append((eps, len(wfn), float(e_vals[0])))
        log(f"stage2: eps={eps:.3e} ndets={len(wfn):>7} E={e_vals[0]:.6f}")
        if len(wfn) <= SAVE_DETS:
            ckpt = {"eps": eps,
                    "dets": wfn.to_det_array().reshape(len(wfn), 2).copy(),
                    "coef": e_vecs[0].copy(), "e": float(e_vals[0])}
        eps *= 0.7943282347242815
    np.savez("n2_hci.npz", rows=np.array(rows),
             deep_dets=wfn.to_det_array().reshape(len(wfn), 2),
             deep_coef=e_vecs[0], deep_e=float(e_vals[0]),
             ck_eps=ckpt["eps"], ck_dets=ckpt["dets"],
             ck_coef=ckpt["coef"], ck_e=ckpt["e"])
    log(f"stage2: done; deepest ndets={len(wfn)} E={e_vals[0]:.6f}")
hci = np.load("n2_hci.npz")
deep_dets, deep_coef = hci["deep_dets"], hci["deep_coef"]
deep_e = float(hci["deep_e"])
log(f"stage2 loaded: {len(deep_coef)} dets, E {deep_e:.6f}; ladder:")
for eps, nd, e in hci["rows"]:
    log(f"    eps={eps:.3e} ndets={int(nd):>7} E={e:.6f}")

# ---------------- stage 3: reference (deepest HCI + PT2) ---------------------
if not os.path.exists("n2_ref.npz"):
    ept2, how = 0.0, "none"
    if hasattr(pyci, "compute_enpt2"):
        wfn = pyci.fullci_wfn(ham.nbasis, *NELEC)
        for dd in deep_dets:
            wfn.add_det(dd)
        raw = float(pyci.compute_enpt2(ham, wfn, deep_coef, deep_e, eps=0.0))
        # returns total (E+PT2) or the bare correction depending on version
        ept2 = raw - deep_e if abs(raw - deep_e) < 0.5 else raw
        how = f"pyci_enpt2 (raw {raw:.6f})"
    else:
        import _fairfight as ff
        K = min(30000, len(deep_coef))
        top = np.argsort(np.abs(deep_coef))[::-1][:K]
        dets64 = (deep_dets[top, 0].astype(np.int64)
                  | (deep_dets[top, 1].astype(np.int64) << NCAS))
        eng = ff.Engine(h1, eri, ecore, NCAS, *NELEC)
        e0, c0, dd = eng.solve(dets64)
        ept2 = eng.pt2(dd, c0, e0)
        how = f"engine_pt2_top{K} (E_var {e0:.6f} vs pyci {deep_e:.6f})"
        deep_e = e0
    np.savez("n2_ref.npz", e_ref=deep_e + ept2, ept2=ept2, how=how)
    log(f"stage3: reference E {deep_e+ept2:.6f} (PT2 {ept2*1e3:+.3f} mHa, {how})")
ref = np.load("n2_ref.npz")
E_REF = float(ref["e_ref"])
log(f"reference: {E_REF:.6f} ({str(ref['how'])})")

# ---------------- stage 4: THEIR QSCI arm (run_qsci_from_npy.py) -------------
if not os.path.exists("n2_qsci.npz"):
    p = np.abs(deep_coef) ** 2
    mass = float(p.sum())
    p = p / p.sum()
    log(f"stage4: sampling the {len(p)}-det deep-HCI |c|^2 as the exact-dist "
        f"proxy (unique-det harvest is capped by this support)")
    Ns = [10**5, 10**6, 10**7] + ([10**8] if QBIG else [])
    rows = []
    for N in Ns:
        counts = RNG.multinomial(N, p)
        hit = counts > 0
        sd = deep_dets[hit]
        # their spin-partner augmentation ("from ci-vector symmetry")
        aug = np.concatenate([sd, sd[:, ::-1]])
        aug = np.unique(aug, axis=0)
        wfn = pyci.fullci_wfn(ham.nbasis, *NELEC)
        for dd in aug:
            wfn.add_det(dd)
        op = pyci.sparse_op(ham, wfn)
        es, vs = op.solve(n=1, tol=1e-9)
        rows.append((N, int(hit.sum()), len(wfn), float(es[0])))
        log(f"stage4: N={N:.0e} uniq={hit.sum():>7} ndets={len(wfn):>7} "
            f"E={es[0]:.6f}  err_vs_ref {(es[0]-E_REF)*1e3:+9.3f} mHa "
            f"(var-only vs deepHCI {(es[0]-deep_e)*1e3:+9.3f})")
    np.savez("n2_qsci.npz", rows=np.array(rows), mass=mass)
    log("stage4: done")
else:
    for N, uq, nd, e in np.load("n2_qsci.npz")["rows"]:
        log(f"stage4 loaded: N={N:.0e} uniq={int(uq)} ndets={int(nd)} "
            f"E={e:.6f} err {(e-E_REF)*1e3:+9.3f} mHa")

# ---------------- stage 5: our Engine arm + H cross-check --------------------
if os.environ.get("SKIPOURS", "0") != "1" and not os.path.exists("n2_ours.npz"):
    import _fairfight as ff
    eng = ff.Engine(h1, eri, ecore, NCAS, *NELEC)
    # (a) H cross-check: Engine re-solve of pyci's ~eps=1e-3 checkpoint space
    ck_dets, ck_e = hci["ck_dets"], float(hci["ck_e"])
    dets64 = np.unique(ck_dets[:, 0].astype(np.int64)
                       | (ck_dets[:, 1].astype(np.int64) << NCAS))
    e_x, _, _ = eng.solve(dets64)
    log(f"stage5a: Engine on pyci eps={float(hci['ck_eps']):.2e} space "
        f"({len(dets64)} dets): {e_x:.6f} vs pyci {ck_e:.6f} "
        f"(delta {(e_x-ck_e)*1e6:+.1f} uHa)")
    # (b) our heat-bath growth to their 15k anchor
    rows = []
    dets = np.array([eng.hf], np.int64)
    e0, c, dets = eng.solve(dets)
    for Dt in (1000, 4000, 15000):
        while len(dets) < Dt:
            step = min(Dt, max(len(dets) + 32, int(len(dets) * 1.5)))
            grown = eng.grow_to(dets, c, e0, step, "hci")
            if len(grown) == len(dets):
                break
            dets = grown
            e0, c, dets = eng.solve(dets)
        rows.append((len(dets), e0))
        log(f"stage5b: [ours hci_hf] D={len(dets):>6} E={e0:.6f} "
            f"err_vs_ref {(e0-E_REF)*1e3:+9.3f} mHa")
    np.savez("n2_ours.npz", rows=np.array(rows),
             xcheck=np.array([e_x, ck_e, len(dets64)]))
print("N2XVAL_DONE", flush=True)
