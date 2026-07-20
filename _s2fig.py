"""Consolidate the S2-audit matrix into the master table + paper figures.

Inputs: s2audit_fe2s2.npz (aufbau 68k, valid first run), s2audit_fe2s2bs.npz,
s2audit_fe2s2deep.npz, s2audit_fe4s4.npz.
Outputs: s2audit_master.csv, fig_s2_vs_d.png, fig_err_vs_d.png.
"""
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REF = {"fe2s2": -116.6056091, "fe4s4": -327.2396369}  # Li-Chan DMRG (fe4s4 approx)
ARMS = [
    ("fe2s2bs",   "fe2s2", "[2Fe-2S] BS seed",     "tab:red"),
    # first valid run: same aufbau protocol, small-D checkpoints (wandering
    # arc 3.0 -> 2.1 -> 4.7 lives only here); deep run continues the curve
    ("fe2s2",     "fe2s2", "_nolegend_",           "tab:blue"),
    ("fe2s2deep", "fe2s2", "[2Fe-2S] aufbau",      "tab:blue"),
    ("fe4s4",     "fe4s4", "[4Fe-4S] aufbau",      "tab:green"),
]

rows_out = []
fig1, ax1 = plt.subplots(figsize=(7.0, 4.6))
fig2, ax2 = plt.subplots(figsize=(7.0, 4.6))

for tag, sysk, label, color in ARMS:
    rs = list(np.load(f"s2audit_{tag}.npz", allow_pickle=True)["results"])
    nd = np.array([r["ndets"] for r in rs], float)
    e0 = np.array([r["e_pyci"][0] for r in rs])
    s2 = np.array([r["s2_pyci"][0] for r in rs])
    err = (e0 - REF[sysk]) * 1e3
    ax1.plot(nd, s2, "o-", color=color, label=label)
    ax2.plot(nd, err, "o-", color=color, label=label)
    for r in rs:
        span = ""
        if "g_symm" in r:
            G = np.asarray(r["g_symm"])
            es = np.asarray(r["e_symm"])
            if len(es) > 1 and es[1] - es[0] < 2e-6:
                ev = np.linalg.eigvalsh(G[:2, :2])
                span = f"[{ev[0]:.3f},{ev[1]:.3f}]"
        rows_out.append({
            "arm": tag, "ndets": r["ndets"],
            "E0": f"{r['e_pyci'][0]:.6f}",
            "err_mHa": f"{(r['e_pyci'][0]-REF[sysk])*1e3:+.1f}",
            "S2_r0": f"{r['s2_pyci'][0]:.3f}",
            "S2_r1": f"{r['s2_pyci'][1]:.3f}" if len(r["s2_pyci"]) > 1 else "",
            "ndets_symm": r.get("ndets_symm", ""),
            "dE0_symm_uHa": (f"{(r['e_symm'][0]-r['e_pyci'][0])*1e6:+.1f}"
                             if "e_symm" in r else ""),
            "symm_pair_S2_span": span,
        })
    # mitigation spans as open markers on fig1 (identical to r0 -> visual proof)
    ns = [r["ndets_symm"] for r in rs if "s2_symm" in r]
    ss = [r["s2_symm"][0] for r in rs if "s2_symm" in r]
    if ns:
        ax1.plot(ns, ss, "s", mfc="none", color=color, ms=9,
                 label=f"{label} + spin-completion")

ax1.axhline(0, color="k", lw=0.8)
for y, name in [(0, "S=0 (target)"), (2, "S=1"), (6, "S=2"), (12, "S=3")]:
    ax1.axhline(y, color="gray", lw=0.5, ls=":")
    ax1.text(1.02, y, name, transform=ax1.get_yaxis_transform(),
             fontsize=7, va="center", color="gray")
ax1.set_xscale("log")
ax1.set_xlabel("determinants in selected space")
ax1.set_ylabel(r"$\langle S^2\rangle$ of followed sector root")
ax1.set_title("Audited Fe-S selected-CI trajectories do not follow the singlet\n"
              "(their protocols, their Hamiltonians, first $S^2$ audit)")
ax1.legend(fontsize=8, loc="lower right", framealpha=0.95)
fig1.tight_layout()
fig1.savefig("fig_s2_vs_d.png", dpi=200)

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.axhspan(15, 30, color="orange", alpha=0.25,
            label="[2Fe-2S] full Heisenberg ladder span (~15-30 mHa)")
ax2.axhline(1.6, color="k", lw=0.8, ls="--", label="chemical accuracy")
ax2.set_xlabel("determinants in selected space")
ax2.set_ylabel("E0 - E(exact singlet)  [mHa]")
ax2.set_title("Energy error vs the exact singlet reference")
ax2.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig("fig_err_vs_d.png", dpi=200)

with open("s2audit_master.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)

print(f"rows: {len(rows_out)}")
for r in rows_out:
    print(" | ".join(str(r[k]) for k in r))
print("S2FIG_DONE")
