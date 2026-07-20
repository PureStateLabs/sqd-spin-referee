"""Diazene (HN=NH) trans<->cis isomerization targets — the Google 2020 Science
companion molecule.  Sycamore treated it at HARTREE-FOCK (10q one-spin basis
rotation, 9 path points, 40 mHa TS-gap resolution *within the HF model*).  Here
the SAME molecule/question at FULL correlation: STO-3G, frozen 2xN-1s core ->
CAS(12,10) = 20 JW qubits; in-CAS FCI reference computed from the DUMPED
integrals (round-trip policy, same as _hchain_targets.py).

Two competing mechanisms, both scanned trans -> cis with exact shared endpoints:
  inversion (in-plane): one HNN angle theta1 106.85 -> 253.15 deg through linear
                        (theta1=180 = TS), dihedral fixed 180, planar throughout
  rotation (out-of-plane): rigid dihedral twist phi 180 -> 0 (phi=90 ~ TS)
9 points per path (matching Google's 9); endpoints duplicated across paths as a
free cross-check.  Output: dz_i{theta}.npz / dz_r{phi}.npz with
(h1, h2, ecore, ncas, na, nb, e_fci, e_hf).  Sentinel DIAZENE_TARGETS_DONE.
"""
import time

import numpy as np
from pyscf import ao2mo, fci, gto, mcscf, scf

T0 = time.time()
RNN, RNH, ANG = 1.252, 1.028, 106.85       # literature trans-diazene parameters
NCAS, NA, NB = 10, 6, 6                    # CAS(12,10), frozen 2x N-1s


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def geom(theta1, phi, theta2=ANG):
    """N1 at origin, N2 on +z.  H1 bonded to N1 at HNN angle theta1, azimuth 0
    (xz-plane); H2 bonded to N2 at angle theta2 from N2->N1, azimuth phi = the
    H1-N1-N2-H2 dihedral.  phi=180 & theta=ANG -> trans; theta1 sweep through
    180 at phi=180 is the planar inversion path ending at the exact cis."""
    t1, t2, ph = np.radians(theta1), np.radians(theta2), np.radians(phi)
    n1 = np.zeros(3)
    n2 = np.array([0.0, 0.0, RNN])
    h1 = n1 + RNH * np.array([np.sin(t1), 0.0, np.cos(t1)])
    h2 = n2 + RNH * np.array([np.sin(t2) * np.cos(ph),
                              np.sin(t2) * np.sin(ph),
                              -np.cos(t2)])
    return [("N", tuple(n1)), ("N", tuple(n2)),
            ("H", tuple(h1)), ("H", tuple(h2))]


def solve(atoms, dm0=None):
    """best-of {continuation, fresh} RHF: dm continuation alone can drag the
    rotation path into a HIGHER local SCF solution past 90 deg (caught by the
    cis-endpoint cross-check: E_FCI agreed to 0.018 mHa, E_HF differed 378)."""
    mol = gto.M(atom=atoms, basis="sto-3g", spin=0, verbose=0)
    cands = []
    for guess in ((dm0, None) if dm0 is not None else (None,)):
        m = scf.RHF(mol)
        m.max_cycle = 200
        m.kernel(dm0=guess)
        if m.converged:
            cands.append(m)
    if not cands:                              # twisted diradicaloid geometries
        m = scf.RHF(mol).newton()
        m.kernel()
        cands = [m]
    mf = min(cands, key=lambda m: m.e_tot)
    mc = mcscf.CASCI(mf, NCAS, NA + NB)        # ncore=2 auto (N 1s pair)
    h1, ecore = mc.get_h1eff()
    h2 = ao2mo.restore(1, mc.get_h2eff(), NCAS)
    e_fci, _ = fci.direct_spin1.kernel(h1, h2, NCAS, (NA, NB), ecore=ecore,
                                       nroots=1, max_cycle=400, conv_tol=1e-12)
    return mf, h1, h2, float(ecore), float(e_fci)


def emit(tag, theta1, phi, dm0):
    mf, h1, h2, ecore, e_fci = solve(geom(theta1, phi), dm0)
    np.savez(f"dz_{tag}.npz", h1=h1, h2=h2, ecore=ecore, ncas=NCAS,
             na=NA, nb=NB, e_fci=e_fci, e_hf=float(mf.e_tot),
             theta1=theta1, phi=phi)
    log(f"dz_{tag}: E_HF={mf.e_tot:.6f}  E_FCI={e_fci:.6f}  "
        f"corr={(mf.e_tot-e_fci)*1e3:7.2f} mHa  conv={mf.converged}")
    return mf.make_rdm1()


# inversion path: planar, theta1 through linear; 253.15 = exact cis endpoint
dm = None
for th in (106.85, 125.0, 143.0, 161.0, 180.0, 199.0, 217.0, 235.0, 253.15):
    dm = emit(f"i{th:06.2f}", th, 180.0, dm)

# rotation path: rigid dihedral twist; phi=0 = exact cis (same as i253.15)
dm = None
for ph in (180.0, 157.5, 135.0, 112.5, 90.0, 67.5, 45.0, 22.5, 0.0):
    dm = emit(f"r{ph:06.2f}", ANG, ph, dm)

log("diazene targets done.")
print("DIAZENE_TARGETS_DONE", flush=True)
