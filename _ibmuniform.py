"""Quantify IBM's own uniform-distribution null control for [2Fe-2S] and
[4Fe-4S] from their published energetics files: if SQD on uniform random
bitstrings matches SQD on hardware samples at the same subspace dimension,
the quantum samples carry no usable structure beyond their particle-number
statistics.

Reads their sqd_hardware_energetics / sqd_on_uniform_distribution txt files
verbatim; reports per-eigenstate means/mins and the hardware-minus-uniform
delta. Output: ibmuniform.npz + printed table. Sentinel: IBMUNIFORM_DONE
"""
import numpy as np

BASE = "_ibm_data/sqd_data_repository-main/experiments"
E2 = -116.6056091   # their DMRG, 2Fe-2S
E4 = -327.2396369   # their DMRG, 4Fe-4S

out = {}


def load(path):
    rows = np.loadtxt(path, skiprows=2)
    return rows.reshape(-1, rows.shape[-1])


print("=== [2Fe-2S], their files, subspace dim column reported by them ===")
for tag in ["A", "B", "C"]:
    hw = load(f"{BASE}/2Fe-2S/sqd_hardware_energetics/"
              f"energy-variance_data_SQD_eigenstate_{tag}.txt")
    un = load(f"{BASE}/2Fe-2S/sqd_on_uniform_distribution/"
              f"energy-variance_data_SQD_eigenstate_{tag}_uniform.txt")
    for name, r in [("hw", hw), ("uniform", un)]:
        e = r[:, 1]
        d = r[:, 2]
        print(f"  {tag}/{name:8s}: n={len(e):3d} dims "
              f"{d.min():.2e}..{d.max():.2e}  E mean {e.mean():.5f} "
              f"best {e.min():.5f} (err vs their DMRG "
              f"{(e.min()-E2)*1e3:+8.2f} mHa)")
        out[f"fe2s2_{tag}_{name}_best"] = e.min()
        out[f"fe2s2_{tag}_{name}_mean"] = e.mean()
    dlt = (hw[:, 1].min() - un[:, 1].min()) * 1e3
    print(f"  {tag}: hardware-best minus uniform-best = {dlt:+.2f} mHa "
          f"({'hardware ahead' if dlt < 0 else 'UNIFORM ahead or tie'})")
    out[f"fe2s2_{tag}_delta_best"] = dlt

print("=== [4Fe-4S], their files ===")
hw = load(f"{BASE}/4Fe-4S/sqd_hardware_energetics/energy-variance_data_SQD.txt")
un = load(f"{BASE}/4Fe-4S/sqd_on_uniform_distribution/"
          f"energy-variance_data_SQD_uniform.txt")
for name, r in [("hw", hw), ("uniform", un)]:
    e = r[:, 1]
    print(f"  {name:8s}: n={len(e):3d} dims {r[:, 2].min():.2e}.."
          f"{r[:, 2].max():.2e}  E mean {e.mean():.5f} best {e.min():.5f} "
          f"(err vs their DMRG {(e.min()-E4)*1e3:+8.2f} mHa)")
    out[f"fe4s4_{name}_best"] = e.min()
    out[f"fe4s4_{name}_mean"] = e.mean()
dlt = (hw[:, 1].min() - un[:, 1].min()) * 1e3
print(f"  fe4s4: hardware-best minus uniform-best = {dlt:+.2f} mHa "
      f"({'hardware ahead' if dlt < 0 else 'UNIFORM ahead or tie'})")
out["fe4s4_delta_best"] = dlt

np.savez("ibmuniform.npz", **out)
print("IBMUNIFORM_DONE", flush=True)
