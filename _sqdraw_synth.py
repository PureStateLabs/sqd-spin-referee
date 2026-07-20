"""Full raw-counts-ladder campaign synthesis: all venues, all configs.

Ladder trio (AWS, THEIR pipeline+counts+integrals, symm1 seed17): N=500/1000/2000.
Ablations (desktop): n1000 symm0 (symmetrize OFF), n1000 seed43 (seed robustness).
Cross-box: n500 ran on BOTH venues (RNG regime gate).
"""
import numpy as np

E_EXACT = -116.6056091
RUNS = [
    ("AWS ", "sqdraw_results_aws/sqdraw_n500_symm1_s17.npz"),
    ("AWS ", "sqdraw_results_aws/sqdraw_n1000_symm1_s17.npz"),
    ("AWS ", "sqdraw_results_aws/sqdraw_n2000_symm1_s17.npz"),
    ("DESK", "sqdraw_n500_symm1_s17.npz"),
    ("DESK", "sqdraw_n1000_symm0_s17.npz"),
    ("DESK", "sqdraw_n1000_symm1_s43.npz"),
]

for venue, path in RUNS:
    z = np.load(path, allow_pickle=True)
    tag = path.split("/")[-1].replace("sqdraw_", "").replace(".npz", "")
    err = (float(z["best_e"]) - E_EXACT) * 1e3
    print("%s %-18s best E %.6f  err %+8.3f mHa  S2 %.4f"
          % (venue, tag, z["best_e"], err, z["best_s2"]))
    h = z["hist"]
    for it in range(1, 6):
        rows = [r for r in h if r["iter"] == it]
        if not rows:
            continue
        es = [r["err_mHa"] for r in rows]
        s2 = [r["s2"] for r in rows]
        d = rows[0].get("dim", -1)
        print("   it%d: E %+8.2f..%+8.2f  S2 %.3f..%.3f  wall %6ds  dim %d"
              % (it, min(es), max(es), min(s2), max(s2),
                 rows[0]["iter_wall_s"], d))
    print()

print("exact E = %.7f ; IBM published best-of-10 raw-SQD @ 4M dets: +216.7 mHa" % E_EXACT)
