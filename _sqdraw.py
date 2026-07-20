"""Run IBM's shipped qiskit-addon-sqd configuration-recovery pipeline on THEIR
raw [2Fe-2S] hardware data with THEIR MO integrals, at their published
subspace construction, and spin-label the states it produces.

Data: 2023-12-22T19-07-32.839270_results.npy = 300 mitigated per-twirl
distributions (each mass 1.0), 2,457,132 unique 40-bit strings, near-flat
weights (max/min 3.4), in-sector(15,15) mass 0.44%. Merged exactly as their
load_data_dict.py does, then deterministically scaled x1e5 to integer counts
(verified: 100.00% of keys and mass preserved) -> BitArray.

Pipeline: diagonalize_fermionic_hamiltonian as shipped (0.12.1; manuscript
used 0.1.0 per their README, same published workflow). Subspace construction
matches their published dims: max_dim=N strings/spin -> dim N^2 under
symmetrize_spin. Per iteration x batch we record E_tot, dims, and THEIR OWN
SCIState.spin_square(), plus wall/RSS (solver-scaling data for flagship
feasibility). E_EXACT is basis-invariant across their MO and Li-Chan
integrals: -116.6056091.

Env: N (strings/spin, required), ARM (symm1|symm0), SPB (default 3N),
     NBATCH (5), MAXIT (5), SEED2 (17), NT (6), SMOKE (0)
Out: sqdraw_n{N}_{ARM}_s{SEED2}.npz ; BitArray cache sqdraw_bits.npz
Sentinel: SQDRAW_DONE
"""
import os
import time
import resource

import numpy as np

NT = int(os.environ.get("NT", "6"))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))

from pyscf import ao2mo  # noqa: E402
from pyscf.tools import fcidump  # noqa: E402

T0 = time.time()
RAW = ("_ibm_data/sqd_data_repository-main/experiments/2Fe-2S/data/"
       "experiment_data/2023-12-22T19-07-32.839270_results.npy")
FCID = ("_ibm_data/sqd_data_repository-main/integrals/2Fe-2S/"
        "fcidump_Fe2S2_MO.txt")
E_EXACT = -116.6056091
NC = 20
SCALE = 1e5

N = int(os.environ["N"])
ARM = os.environ.get("ARM", "symm1")
SYMM = ARM == "symm1"
SPB = int(os.environ.get("SPB", str(3 * N)))
NBATCH = int(os.environ.get("NBATCH", "5"))
MAXIT = int(os.environ.get("MAXIT", "5"))
SEED2 = int(os.environ.get("SEED2", "17"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT = f"sqdraw_n{N}_{ARM}_s{SEED2}.npz"


def log(m):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.time()-T0:7.1f}s rss{rss:5.1f}G] {m}", flush=True)


if os.path.exists(OUT) and not SMOKE:
    log(f"{OUT} exists -- skip")
    print("SQDRAW_DONE", flush=True)
    raise SystemExit

fd = fcidump.read(FCID)
ECORE = float(fd["ECORE"])
H1 = fd["H1"]
ERI = ao2mo.restore(1, fd["H2"], NC)
log(f"THEIR MO integrals loaded (ecore {ECORE:.6f})")

from qiskit.primitives import BitArray  # noqa: E402
from qiskit_addon_sqd.fermion import (  # noqa: E402
    diagonalize_fermionic_hamiltonian,
)

if os.path.exists("sqdraw_bits.npz"):
    z = np.load("sqdraw_bits.npz")
    bit_array = BitArray(z["arr"], num_bits=int(z["nbits"]))
    log(f"BitArray cache hit: {bit_array.num_shots:,} rows")
else:
    d = np.load(RAW, allow_pickle=True)
    merged = {}
    for cts in d:
        for k, v in cts.items():
            merged[k] = merged.get(k, 0.0) + v
    ints = {k: int(round(v * SCALE)) for k, v in merged.items()}
    ints = {k: c for k, c in ints.items() if c > 0}
    log(f"their loader: {len(d)} dicts -> {len(merged):,} unique keys; "
        f"x{SCALE:.0e} -> {sum(ints.values()):,} rows ({len(ints):,} kept)")
    bit_array = BitArray.from_counts(ints, num_bits=2 * NC)
    np.savez_compressed("sqdraw_bits.npz", arr=bit_array.array, nbits=2 * NC)
    log(f"BitArray built + cached ({bit_array.num_shots:,} rows)")

hist = []
it_box = [0]
t_box = [time.time()]


def cb(results):
    it_box[0] += 1
    dt = time.time() - t_box[0]
    t_box[0] = time.time()
    for j, r in enumerate(results):
        s2 = float(r.sci_state.spin_square())  # THEIR diagnostic
        da = len(r.sci_state.ci_strs_a)
        db = len(r.sci_state.ci_strs_b)
        e_tot = r.energy + ECORE
        hist.append({"iter": it_box[0], "batch": j, "dim_a": da, "dim_b": db,
                     "e_tot": e_tot, "err_mHa": (e_tot - E_EXACT) * 1e3,
                     "s2": s2, "iter_wall_s": dt})
        log(f"it{it_box[0]} b{j}: dim {da}x{db}={da*db:,} E {e_tot:.6f} "
            f"(err {(e_tot-E_EXACT)*1e3:+8.3f} mHa) THEIR S2 {s2:.3f} "
            f"[iter {dt:.0f}s]")


log(f"=== N={N} arm {ARM} SPB={SPB} NBATCH={NBATCH} MAXIT={MAXIT} "
    f"seed {SEED2} ===")
res = diagonalize_fermionic_hamiltonian(
    H1, ERI, bit_array,
    samples_per_batch=SPB, norb=NC, nelec=(15, 15),
    num_batches=NBATCH, max_iterations=MAXIT,
    symmetrize_spin=SYMM, max_dim=N,
    callback=cb, seed=SEED2,
)
s2f = float(res.sci_state.spin_square())
ef = res.energy + ECORE
log(f"BEST: E {ef:.6f} (err {(ef-E_EXACT)*1e3:+8.3f} mHa) THEIR S2 {s2f:.3f} "
    f"dim {len(res.sci_state.ci_strs_a)}x{len(res.sci_state.ci_strs_b)}")
if not SMOKE:
    np.savez(OUT, hist=np.array(hist, dtype=object), best_e=ef, best_s2=s2f,
             n=N, arm=ARM, spb=SPB, nbatch=NBATCH, maxit=MAXIT, seed=SEED2)
print("SQDRAW_DONE", flush=True)
