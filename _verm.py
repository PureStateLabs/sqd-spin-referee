# -*- coding: utf-8 -*-
"""Version matrix: qiskit-addon-sqd earliest-public (0.3.0) vs current
(0.12.1) on identical physical inputs (review round 2, Tier 3b).

Finding that motivates this script: the flagship manuscript states package
version 0.1.0 (their data-repo README also claims all versions are on PyPI
and GitHub), but 0.1.0 is retrievable from NEITHER channel -- PyPI's earliest
release is 0.3.0 and no 0.1.0 git tag exists (checked 2026-07-15). The
matrix therefore runs at the earliest PUBLICLY AUDITABLE release.

Three legs, identical inputs under both versions, each version using its own
construction + solver + spin diagnostic:
  V1  reproducer core: top-60 exact-marginal strings (alpha set == beta set,
      spin-inversion closure structural) fed as diagonal-det samples
  V2  their raw hardware shots: postselect to the (15,15) sector, seed-17
      subsample of 500 shots, closure ON
  V3  same 500 shots, closure OFF (open_shell=True)
Plus the semantics record: solve_fermion signature (spin_sq default), whether
spin_sq wires to PySCF fix_spin_, and the construction function's docstring.

Output: verm_<version>.npz + printed table. Sentinel: VERM_DONE
Env: MAXCYC (default 400)
"""
import inspect
import time

import numpy as np
from pyscf import ao2mo
from pyscf.tools import fcidump

import qiskit_addon_sqd
from qiskit_addon_sqd import fermion

try:
    from importlib.metadata import version as _pkgver
    VER = _pkgver("qiskit-addon-sqd")
except Exception:
    VER = getattr(qiskit_addon_sqd, "__version__", "unknown")
TAG = VER.replace(".", "_")
T0 = time.time()
MAXCYC = 400

FCID = ("_ibm_data/sqd_data_repository-main/integrals/2Fe-2S/"
        "fcidump_Fe2S2_MO.txt")
NORB, NELEC = 20, (15, 15)
E2 = -116.6056091


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


log(f"qiskit-addon-sqd {VER}")

# ---- semantics record --------------------------------------------------
ssig = inspect.signature(fermion.solve_fermion)
spin_default = ssig.parameters["spin_sq"].default
src = inspect.getsource(fermion)
wires_fix_spin = "fix_spin" in src
if hasattr(fermion, "bitstring_matrix_to_sorted_addresses"):
    build_name = "bitstring_matrix_to_sorted_addresses"
elif hasattr(fermion, "bitstring_matrix_to_ci_strs"):
    build_name = "bitstring_matrix_to_ci_strs"
else:
    raise RuntimeError("no known construction function in fermion module")
build = getattr(fermion, build_name)
bsig = inspect.signature(build)
# iteration-cap kwarg: 0.3.0 names max_davidson explicitly; later versions
# forward **kwargs to pyscf kernel_fixed_space, whose kwarg is max_cycle
if "max_davidson" in ssig.parameters:
    mc_kw = "max_davidson"
elif "max_cycle" in ssig.parameters:
    mc_kw = "max_cycle"
else:
    mc_kw = "max_cycle"  # reaches pyscf via the signature's **kwargs
log(f"construction fn: {build_name}{bsig}")
log(f"solve_fermion: spin_sq default = {spin_default!r}; "
    f"iteration kwarg = {mc_kw}; source wires fix_spin: {wires_fix_spin}")

fd = fcidump.read(FCID)
H1 = fd["H1"]
ERI = ao2mo.restore(1, fd["H2"], NORB)
ECORE = float(fd.get("ECORE", 0.0))
log(f"THEIR MO integrals loaded (ecore {ECORE:.6f})")


def solve(mat, open_shell):
    """One version-native construction + solve + diagnostic pass."""
    strs = build(mat, open_shell=open_shell)
    na, nb = len(strs[0]), len(strs[1])
    res = fermion.solve_fermion(strs, H1, ERI, **{mc_kw: MAXCYC})
    if hasattr(res, "energy"):                       # dataclass style
        e = float(res.energy)
        ss = float(getattr(res, "spin_sq", np.nan))
        how = "attrs"
    else:                                            # tuple style
        e = float(res[0])
        ss = float(res[-1])
        how = f"tuple[{len(res)}]"
    return e + ECORE, ss, na, nb, how


# ---- V1: reproducer core (N=60 marginal strings, closure structural) ---
z = np.load("lucjopt_marginals.npz")
ra, rb = z["ra"], z["rb"]
strs = z["strs"]
sa_all = strs[ra] if int(ra.max()) < 32767 else ra
sb_all = strs[rb] if int(rb.max()) < 32767 else rb
N1 = 60
S = np.sort(sa_all[:N1].astype(np.int64))
assert set(S) == set(np.sort(sb_all[:N1].astype(np.int64))), \
    "alpha/beta top-60 sets differ?!"
# diagonal-det samples: row i = [beta bits of S_i | alpha bits of S_i],
# columns run orbital NORB-1 -> 0 within each half (their layout)
mat1 = np.zeros((N1, 2 * NORB), dtype=bool)
for i, s in enumerate(S):
    bits = [(int(s) >> p) & 1 for p in range(NORB - 1, -1, -1)]
    mat1[i, :NORB] = bits          # beta half
    mat1[i, NORB:] = bits          # alpha half
e1, ss1, na1, nb1, how1 = solve(mat1, open_shell=False)
log(f"V1 reproducer N=60: dim {na1}x{nb1}, E {e1:.6f} "
    f"(err {(e1-E2)*1e3:+.1f} mHa), their <S^2> {ss1:.4f}  [{how1}]")

# ---- V2/V3: their raw hardware shots, one seed-17 batch of 500 ---------
d = np.load("ibmraw_counts.npz")
keys, vals = d["keys"], d["vals"].astype(np.int64)
nbits = len(str(keys[0]))
assert nbits == 2 * NORB, f"unexpected bit width {nbits}"
km = np.frombuffer("".join(str(k) for k in keys).encode(),
                   dtype="S1").reshape(len(keys), nbits) == b"1"
pop_b = km[:, :NORB].sum(1)
pop_a = km[:, NORB:].sum(1)
sector = (pop_a == 15) & (pop_b == 15)
log(f"V2 prep: {vals.sum():,} shots, {sector.sum():,} unique in-sector keys "
    f"({vals[sector].sum():,} shots = {vals[sector].sum()/vals.sum():.3%})")
rows = np.repeat(np.flatnonzero(sector), vals[sector])
rng = np.random.default_rng(17)
pick = rng.choice(rows, size=500, replace=False)
mat2 = km[pick]

e2, ss2, na2, nb2, how2 = solve(mat2, open_shell=False)
log(f"V2 hardware batch closure ON : dim {na2}x{nb2}, E {e2:.6f} "
    f"(err {(e2-E2)*1e3:+.1f} mHa), their <S^2> {ss2:.4f}")
e3, ss3, na3, nb3, how3 = solve(mat2, open_shell=True)
log(f"V3 hardware batch closure OFF: dim {na3}x{nb3}, E {e3:.6f} "
    f"(err {(e3-E2)*1e3:+.1f} mHa), their <S^2> {ss3:.4f}")

np.savez(f"verm_{TAG}.npz",
         version=VER, build_name=build_name, mc_kw=mc_kw,
         spin_default=repr(spin_default), wires_fix_spin=wires_fix_spin,
         build_sig=str(bsig), solve_sig=str(ssig),
         v1=np.array([e1, ss1, na1, nb1]),
         v2=np.array([e2, ss2, na2, nb2]),
         v3=np.array([e3, ss3, na3, nb3]),
         n_sector_keys=int(sector.sum()),
         n_sector_shots=int(vals[sector].sum()),
         n_shots=int(vals.sum()))
log(f"saved verm_{TAG}.npz")
print("VERM_DONE", flush=True)
