"""Flagship-dimension S2 point via THEIR OWN matrix-free solver.

Diagonalizes outer-product subspaces of IBM's released [2Fe-2S] configuration
set (7658 alpha x 7658 beta unique half-strings = 58,644,964 determinants,
the space that contains every published desktop-checkable subspace) with the
as-shipped `qiskit_addon_sqd.fermion.solve_fermion` (pyscf-backed direct CI,
matrix-free: RAM ~ Davidson vectors, not the 2.16 TB explicit operator).
Rungs = top-N strings per spin ranked by the exact-state marginals of THEIR
optimized circuit (`lucjopt_marginals.npz`); N=7658 = the verbatim full set.
Their solver returns THEIR spin^2 for the same state: energy + identity on
one state at their flagship dimension.

Cross-solver gates: pyci anchors from the frontier-D recon ladder
(n=300/500/600/1000/2000 on the same construction + integrals).

Env: N (strings/spin, required; 7658=full), NT (8), SAVEVEC (1: store civec
     in the npz for warm starts/xchecks; ~8B*N^2), SMOKE (0)
Out: flagsolve_n{N}.npz   Sentinel: FLAGSOLVE_DONE
"""
import os
import time
import resource
import socket

import numpy as np

NT = int(os.environ.get("NT", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
os.environ.setdefault("PYSCF_MAX_MEMORY", os.environ.get("MAXMEM", "12000"))

from pyscf import ao2mo  # noqa: E402
from pyscf.tools import fcidump  # noqa: E402
from qiskit_addon_sqd.fermion import solve_fermion  # noqa: E402

T0 = time.time()
FCID = ("_ibm_data/sqd_data_repository-main/integrals/2Fe-2S/"
        "fcidump_Fe2S2_MO.txt")
E_EXACT = -116.6056091
NC = 20

N = int(os.environ["N"])
SAVEVEC = os.environ.get("SAVEVEC", "1") == "1"
SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT = f"flagsolve_n{N}.npz"

# pyci anchors (frontier-D recon ladder, same ranking + THEIR integrals)
ANCHOR = {300: (234.8, 3.808), 500: (186.8, 3.790), 600: (173.2, 3.797),
          1000: (129.8, 3.786), 2000: (80.0, 3.5192)}
ANCHOR_E600 = -116.432419617  # bit-identical cross-silicon recon value


def log(m):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.time()-T0:8.1f}s rss{rss:6.2f}G] {m}", flush=True)


if os.path.exists(OUT) and not SMOKE:
    log(f"{OUT} exists -- skip")
    print("FLAGSOLVE_DONE", flush=True)
    raise SystemExit

z = np.load("lucjopt_marginals.npz")
ra, rb, wra, wrb = z["ra"], z["rb"], z["wra"], z["wrb"]
strs = z["strs"]
assert wra[0] >= wra[-1] and np.all(np.diff(wra) <= 1e-18), "ra not ranked"
assert wrb[0] >= wrb[-1] and np.all(np.diff(wrb) <= 1e-18), "rb not ranked"
# smallest 15-of-20 string is 0x7FFF=32767; any value below that is an index
sa_all = strs[ra] if int(ra.max()) < 32767 else ra
sb_all = strs[rb] if int(rb.max()) < 32767 else rb
pc = np.vectorize(lambda x: bin(int(x)).count("1"))
assert np.all(pc(sa_all[:64]) == 15) and np.all(pc(sb_all[:64]) == 15)
assert N <= len(sa_all), f"N={N} > released {len(sa_all)}"
sa = np.sort(sa_all[:N].astype(np.int64))
sb = np.sort(sb_all[:N].astype(np.int64))
log(f"strings: top-{N}/{len(sa_all)} per spin (ranked by their exact-state "
    f"marginals), dim {N}x{N} = {N*N:,}")

fd = fcidump.read(FCID)
H1 = fd["H1"]
ERI = ao2mo.restore(1, fd["H2"], NC)
log(f"THEIR MO integrals loaded (ecore {float(fd['ECORE']):.6f})")

# optional pyscf kernel_fixed_space kwargs (unset => as-shipped defaults);
# MAXCYCLE guards against silent under-convergence at flagship D (pyci NMV
# reached 133 there vs pyscf default max_cycle 50), VERBOSE=5 logs davidson
kw_solver = {}
if os.environ.get("MAXCYCLE"):
    kw_solver["max_cycle"] = int(os.environ["MAXCYCLE"])
if os.environ.get("VERBOSE"):
    kw_solver["verbose"] = int(os.environ["VERBOSE"])
log(f"solve_fermion (as shipped, spin_sq unset, kwargs={kw_solver or 'none'})"
    f" NT={NT} ...")
t0 = time.time()
e, sci_state, occ, s2 = solve_fermion((sa, sb), H1, ERI, **kw_solver)
wall = time.time() - t0
err = (e - E_EXACT) * 1e3
log(f"E {e:.9f}  err {err:+9.3f} mHa  THEIR S2 {s2:.4f}  "
    f"[solve {wall:,.0f}s]  amps {np.asarray(sci_state.amplitudes).shape} "
    f"strs {len(np.asarray(sci_state.ci_strs_a))}x"
    f"{len(np.asarray(sci_state.ci_strs_b))}")

if N in ANCHOR:
    ae, as2 = ANCHOR[N]
    de = abs(err - ae)
    ds = abs(s2 - as2)
    tag = "PASS" if (de < 0.15 and ds < 5e-3) else "FAIL"
    log(f"GATE vs pyci anchor n={N}: dE {de:.3f} mHa dS2 {ds:.4f} [{tag}]")
    if N == 600:
        log(f"GATE vs exact recon E600: dE {abs(e-ANCHOR_E600)*1e6:.2f} uHa "
            f"[{'PASS' if abs(e-ANCHOR_E600) < 5e-6 else 'FAIL'}]")

kw = dict(n=N, e=e, err_mHa=err, s2=float(s2), wall_s=wall, nt=NT,
          host=socket.gethostname(), ndets=N * N,
          sa_first8=sa[:8], sb_first8=sb[:8])
if SAVEVEC and not SMOKE:
    kw["amps"] = np.asarray(sci_state.amplitudes)
    kw["ci_strs_a"] = np.asarray(sci_state.ci_strs_a)
    kw["ci_strs_b"] = np.asarray(sci_state.ci_strs_b)
if not SMOKE:
    np.savez_compressed(OUT, **kw)
    log(f"saved {OUT}")
print("FLAGSOLVE_DONE", flush=True)
