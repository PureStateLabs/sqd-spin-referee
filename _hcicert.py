"""Cross-implementation certification of the FAIRFIGHT classical baseline.

pyscf.fci.selected_ci is an independent heat-bath selected-CI implementation
(|H*c| selection, Holmes/Umrigar criterion — same family as Reinholdt's HCI).
Run it on the SAME diazene (12e,10o) systems as fairfight round 3, sweeping
select_cutoff -> (Ndet, E) points, with:
  - spin labels per root (dense-embed the SCI vector, spin_square),
  - our Engine re-diagonalizing pyscf's selected product space (three-way
    Hamiltonian agreement: pyscf-SCI vs our Slater-Condon on identical dets),
  - the stored round-3 hci_hf / cipsi_hf curves printed alongside.
If our arm tracks pyscf-SCI at matched det count, the fight's classical
baseline is certified at full heat-bath strength by a third-party code.
Sentinel: HCICERT_DONE
"""
import os
import time

import numpy as np
from pyscf import fci
from pyscf.fci import cistring, selected_ci, spin_op

import _fairfight as ff

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


CUTS = [float(x) for x in os.environ.get(
    "CUTS", "3e-3,1.5e-3,8e-4,4e-4,2e-4,1e-4").split(",")]
NR = int(os.environ.get("NR", "2"))


def sci_to_dense(civec, ncas, na, nb):
    sa, sb = civec._strs
    A = np.zeros((cistring.num_strings(ncas, na),
                  cistring.num_strings(ncas, nb)))
    ia = cistring.strs2addr(ncas, na, np.asarray(sa))
    ib = cistring.strs2addr(ncas, nb, np.asarray(sb))
    A[np.ix_(ia, ib)] = np.asarray(civec).reshape(len(sa), len(sb))
    return A


try:
    stored = np.load("fairfight_dzsinglet.npz", allow_pickle=True)["results"]
    stored = {r["file"]: r["results"] for r in stored}
except Exception as e:  # noqa: BLE001 - comparison table is optional
    log(f"no stored round-3 curves ({e}); running pyscf side only")
    stored = {}

for fname in os.environ.get("GLOB",
                            "dz_r090.00.npz dz_i180.00.npz").split():
    d = np.load(fname)
    ncas, na, nb = int(d["ncas"]), int(d["na"]), int(d["nb"])
    h1, h2, ecore = d["h1"], d["h2"], float(d["ecore"])
    roots, all_dets = ff.exact_reference(d)
    ti = next(i for i, r in enumerate(roots) if abs(r[1] - 1.0) < 0.2)
    e_s = roots[ti][0]
    log(f"{fname}: singlet target root{ti} E {e_s:.6f}; "
        + "  ".join(f"r{i}:E{r[0]:.6f}/m{r[1]:.1f}"
                    for i, r in enumerate(roots[:3])))
    eng = ff.Engine(h1, h2, ecore, ncas, na, nb)

    for cut in CUTS:
        try:
            es, civs = selected_ci.kernel(
                h1, h2, ncas, (na, nb), ecore=ecore, nroots=NR,
                select_cutoff=cut, ci_coeff_cutoff=cut)
        except Exception as e:  # noqa: BLE001 - degrade to single root
            log(f"  cut={cut:.1e} nroots={NR} failed ({e}); retry nroots=1")
            es, civs = selected_ci.kernel(
                h1, h2, ncas, (na, nb), ecore=ecore, nroots=1,
                select_cutoff=cut, ci_coeff_cutoff=cut)
        if NR == 1 or np.ndim(es) == 0:
            es, civs = [float(es)], [civs]
        sa, sb = civs[0]._strs
        ndet = len(sa) * len(sb)
        mults = []
        for v in civs:
            _, m = spin_op.spin_square(sci_to_dense(v, ncas, na, nb),
                                       ncas, (na, nb))
            mults.append(float(m))
        # our Engine on pyscf's product space (identical dets)
        pdets = (np.asarray(sa, np.int64)[:, None]
                 | (np.asarray(sb, np.int64)[None, :] << ncas)).ravel()
        ws, V, _ = eng.solve_k(np.unique(pdets), k=min(2, len(pdets)))
        dlow = (ws[0] - es[0]) * 1e3
        rr = " | ".join(f"r{i} {(e-e_s)*1e3:+8.3f} mHa m{m:.1f}"
                        for i, (e, m) in enumerate(zip(es, mults)))
        log(f"  cut={cut:.1e} |Sa|x|Sb|={len(sa)}x{len(sb)}={ndet:>6}  {rr}"
            f"  engine-vs-pyscf {dlow:+.4f} mHa")

    if fname in stored:
        for arm in ("hci_hf", "cipsi_hf"):
            rows = stored[fname][arm]
            pts = "  ".join(f"D={int(r[0])}:{r[1]:+.3f}" for r in rows)
            log(f"  [round3 {arm}] E_var-singlet (mHa): {pts}")

print("HCICERT_DONE", flush=True)
