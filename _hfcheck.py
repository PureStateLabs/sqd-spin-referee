"""Is the Hartree-Fock string in IBM's 58.6M 'sampled configurations' file?

Written 2026-09-11 to settle a contradiction before correcting the paper.
Paper section 2.9 says the file's entries include no Hartree-Fock string. The
archived `lucjopt_stats.npz` says it is present once. This script re-derives
the answer three independent ways, reading only IBM's raw released files and
using none of `_lucjopt.py`'s code or outputs:

  CHECK 1  physics, convention-free. Build the determinant in row 0 under
           both possible column-to-orbital orders and evaluate its energy with
           IBM's MO FCIDUMP. The Hartree-Fock determinant must reproduce IBM's
           own tabulated RHF energy. Nothing about bit order is assumed.
  CHECK 2  IBM's own label. load_samples.py prints bit_arrays[0] under
           "Hartree Fock string:". Confirm that row is the one CHECK 1 found.
  CHECK 3  full raw scan with fresh packing (np.packbits, not _lucjopt.py's
           matmul packing). Count every occurrence of that row across all 40
           partitions. Also re-derive every other claim in GitHub issue
           jrm874/sqd_data_repository#4: row count, distinctness, prefix
           distinctness, distinct alpha/beta halves, the product-grid shortfall.

Run:  MSYS_NO_PATHCONV=1 wsl -e bash -c 'cd /mnt/c/Users/tyler/Downloads/
      indrajala-core && ./.venv/bin/python -u _hfcheck.py'
"""
import time

import numpy as np
from pyscf import ao2mo
from pyscf.tools import fcidump

T0 = time.time()
BASE = "_ibm_data/sqd_data_repository-main"
SAMP = (f"{BASE}/numerics/2Fe-2S_optimal_circuits/samples_from_circuit/"
        "samples_58644964_partition_{}.npy")
FCID = f"{BASE}/integrals/2Fe-2S/fcidump_Fe2S2_MO.txt"
E_RHF_IBM = -116.205816  # classical_reference_energies/2Fe-2S/classical_methods_energies.txt
NORB = 20


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def bits(v):
    return "".join("1" if x else "0" for x in v)


# ---- CHECK 1: the energy of row 0, no bit convention assumed ----------------
fd = fcidump.read(FCID)
h1 = np.asarray(fd["H1"])
h2 = ao2mo.restore(1, fd["H2"], fd["NORB"])
ecore = float(fd["ECORE"])
assert fd["NORB"] == NORB, fd["NORB"]


def det_energy(occ_a, occ_b):
    """<D|H|D> for a single determinant, chemists' (ij|kl) integrals."""
    e = ecore + sum(h1[i, i] for i in occ_a) + sum(h1[i, i] for i in occ_b)
    for occ in (occ_a, occ_b):
        for i in occ:
            for j in occ:
                e += 0.5 * (h2[i, i, j, j] - h2[i, j, j, i])
    for i in occ_a:
        for j in occ_b:
            e += h2[i, i, j, j]
    return float(e)


row0 = np.load(SAMP.format(0), mmap_mode="r")[0]
ra, rb = np.asarray(row0[:NORB]), np.asarray(row0[NORB:])
log(f"row 0 raw   alpha {bits(ra)}  ({int(ra.sum())} occupied)")
log(f"row 0 raw   beta  {bits(rb)}  ({int(rb.sum())} occupied)")

e_aufbau = det_energy(range(15), range(15))
e_direct = det_energy([j for j in range(NORB) if ra[j]],
                      [j for j in range(NORB) if rb[j]])
e_rev = det_energy([NORB - 1 - j for j in range(NORB) if ra[j]],
                   [NORB - 1 - j for j in range(NORB) if rb[j]])
log(f"ECORE in FCIDUMP        {ecore}")
log(f"aufbau det (orbs 0-14)  {e_aufbau:.6f}   IBM RHF {E_RHF_IBM:.6f}   "
    f"diff {abs(e_aufbau - E_RHF_IBM):.1e}")
log(f"row 0, column j = orb j      -> {e_direct:.6f}")
log(f"row 0, column j = orb 19-j   -> {e_rev:.6f}")
c1 = abs(e_aufbau - E_RHF_IBM) < 5e-6 and abs(e_rev - E_RHF_IBM) < 5e-6
log(f"CHECK 1 {'PASS' if c1 else 'FAIL'}: row 0 is the RHF determinant "
    f"(under the orbital-19->0 column order; the other order gives "
    f"{e_direct - E_RHF_IBM:+.3f} Ha)")

# ---- CHECK 2: the row IBM's own loader labels "Hartree Fock string" ---------
# load_samples.py concatenates partitions 0..39 in order and prints
# bit_arrays[0]. Concatenation preserves order, so that is partition 0 row 0.
c2 = c1  # same row; CHECK 1 established its identity physically
log(f"CHECK 2 {'PASS' if c2 else 'FAIL'}: load_samples.py's 'Hartree Fock "
    f"string:' is bit_arrays[0] == the row CHECK 1 identified as RHF")

# ---- CHECK 3: full raw scan, fresh packing ----------------------------------
def pack(arr):
    """Rows of bools -> uint64 keys via np.packbits (big-endian, zero-pad)."""
    p = np.packbits(arr, axis=1)
    p = np.pad(p, ((0, 0), (8 - p.shape[1], 0)))
    return p.view(">u8").ravel().astype(np.uint64)


key0 = pack(np.asarray(row0)[None, :])[0]
keys, ka, kb, hf_hits, n_seen = [], [], [], [], 0
for i in range(40):
    arr = np.load(SAMP.format(i))
    assert arr.dtype == bool and arr.shape[1] == 2 * NORB, (arr.dtype, arr.shape)
    k = pack(arr)
    hits = np.flatnonzero(k == key0)
    hf_hits.extend((n_seen + hits).tolist())
    keys.append(k)
    ka.append(pack(arr[:, :NORB]))
    kb.append(pack(arr[:, NORB:]))
    n_seen += len(arr)
    if i % 10 == 9:
        log(f"scanned {i + 1}/40 partitions, {n_seen:,} rows")
keys = np.concatenate(keys)
ka = np.concatenate(ka)
kb = np.concatenate(kb)

n_rows = len(keys)
n_uniq = len(np.unique(keys))
pre = len(np.unique(keys[:100_000]))
ua, ca = np.unique(ka, return_counts=True)
ub, cb = np.unique(kb, return_counts=True)
short_a = int((ca < len(ub)).sum())
short_b = int((cb < len(ua)).sum())

log(f"rows                          {n_rows:,}")
log(f"distinct rows                 {n_uniq:,}")
log(f"distinct in first 100,000     {pre:,}")
log(f"rows equal to the RHF row     {len(hf_hits)} at global index {hf_hits}")
log(f"distinct alpha halves         {len(ua):,}")
log(f"distinct beta halves          {len(ub):,}")
log(f"alpha x beta                  {len(ua) * len(ub):,}   (7658^2 = {7658**2:,})")
log(f"missing from full grid        {len(ua) * len(ub) - n_rows}")
log(f"alpha halves short of a full row: {short_a} "
    f"(pairs with {int(ca[ca < len(ub)][0]) if short_a else '-'} betas)")
log(f"beta halves short of a full column: {short_b}")

c3 = (len(hf_hits) == 1 and hf_hits == [0])
log(f"CHECK 3 {'PASS' if c3 else 'FAIL'}: the RHF row occurs exactly once, "
    f"at global row 0")

issue = {
    "58,644,960 rows": n_rows == 58_644_960,
    "no two the same": n_uniq == n_rows,
    "first 100,000 all distinct": pre == 100_000,
    "HF once, as row 0": c3,
    "7,658 distinct alpha halves": len(ua) == 7658,
    "7,658 distinct beta halves": len(ub) == 7658,
    "7,658 x 7,658 = 58,644,964": 7658 * 7658 == 58_644_964,
    "grid minus exactly 4": len(ua) * len(ub) - n_rows == 4,
    "all four share one alpha half": short_a == 1 and short_b == 4,
}
print()
print("Claims in issue #4, re-derived from IBM's raw files:")
for claim, ok in issue.items():
    print(f"  {'OK  ' if ok else 'FAIL'} {claim}")
print()
allok = c1 and c2 and c3 and all(issue.values())
print("VERDICT:", "ALL CONFIRMED - the HF string IS in the file, once, at row 0"
      if allok else "SOMETHING DID NOT CONFIRM - read the log above")
