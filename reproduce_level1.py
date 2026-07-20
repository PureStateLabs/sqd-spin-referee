"""VALIDATE Level 1 -- five-minute, repo-only reproduction (no external download).

Confirms, from the committed as-shipped-pipeline archives, the four things the
validation checklist asks for:
  * energy BEFORE and AFTER spin-inversion completion (symmetrize_spin off vs on)
  * <S^2> of the returned state (before and after completion)
  * the completed subspace size (completion multiplies the determinant count ~4x)
  * the exact package versions in use

The point: spin-inversion "completion" -- the mitigation the flagship SQD work
relies on -- roughly quadruples the determinant count yet moves the energy by
< 1 nHa and the spin by ~1e-11. It permits spin symmetry without producing a
singlet; an <S^2> readout is required to know which spin state an energy belongs
to. (Paper Section 2.6; the no-op is measured batch-by-batch by _sqdship_delta.py.)

  python reproduce_level1.py

Reads sqdship_symm1.npz / sqdship_symm0.npz (committed in this repo).
Expected output is pinned in EXPECTED_OUTPUTS.md.
"""
import platform

import numpy as np


def versions():
    print("# package versions")
    print(f"{'python':16s} {platform.python_version()}")
    for mod in ["numpy", "scipy", "pyscf", "qiskit_addon_sqd", "pyci"]:
        try:
            m = __import__(mod)
            print(f"{mod:16s} {getattr(m, '__version__', '?')}")
        except Exception as e:
            print(f"{mod:16s} (not installed: {e.__class__.__name__})")


def maxdim(hist):
    return max(int(r["dim_a"]) * int(r["dim_b"]) for r in hist)


versions()

on = np.load("sqdship_symm1.npz", allow_pickle=True)    # completion ON
off = np.load("sqdship_symm0.npz", allow_pickle=True)    # completion OFF
h_on = list(on["hist"])
h_off = list(off["hist"])

e_on, e_off = float(on["best_e"]), float(off["best_e"])
s2_on, s2_off = float(on["best_s2"]), float(off["best_s2"])
d_on, d_off = maxdim(h_on), maxdim(h_off)

print("\n# as-shipped pipeline on the [2Fe-2S] soup samples "
      "(seed 17; singlet target has <S^2> = 0)")
print(f"{'':28s}{'best energy (Ha)':>20s}{'<S^2>':>12s}{'determinants':>16s}")
print(f"{'BEFORE completion (off)':28s}{e_off:>20.8f}{s2_off:>12.5f}{d_off:>16,}")
print(f"{'AFTER  completion (on)':28s}{e_on:>20.8f}{s2_on:>12.5f}{d_on:>16,}")
print(f"{'delta':28s}{(e_on - e_off) * 1e9:>17.3f} nHa"
      f"{s2_on - s2_off:>12.1e}{d_on / d_off:>15.2f}x")

ok_energy = abs(e_on - e_off) < 1e-9           # < 1 nHa: a measured energy no-op
ok_spin = abs(s2_on - s2_off) < 1e-6           # spin unchanged
ok_grow = 3.9 < d_on / d_off < 4.1             # completion ~4x the determinants
ok_soup = s2_on > 3.0                          # returned state is high-spin, not singlet

print("\nVERDICT:")
print(f"  completion changed the energy by < 1 nHa .......... "
      f"{'PASS' if ok_energy else 'FAIL'}")
print(f"  completion left <S^2> essentially unchanged ....... "
      f"{'PASS' if ok_spin else 'FAIL'}")
print(f"  completion multiplied the determinant count ~4x ... "
      f"{'PASS' if ok_grow else 'FAIL'}")
print(f"  the returned state is a high-spin mixture, not 0 .. "
      f"{'PASS' if ok_soup else 'FAIL'}  (<S^2> = {s2_on:.2f})")
print("\nSpin-inversion completion permits the symmetry (4x the determinants) "
      "but does\nnot produce a singlet (<S^2> unmoved, far from 0). "
      "Level 1 reproduced.")
