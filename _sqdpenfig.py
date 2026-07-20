"""Penalty-ON frontier figure (fig_penalty_frontier.png): what a spin penalty
buys on the shipped pipeline's own [2Fe-2S] soup subspaces.

Panel A: the spin-energy frontier. <S^2> of the ground of H + lam*S^2 vs its
true energy error, for the captured it1 (245^2) and it4 (441^2, the +45.5 mHa
operating point) subspaces, markers at each lam; spin-ladder rails at
S(S+1) = 0,2,6; the full penalized recovery protocol (S5) overlaid as stars.
The jaws -- benchmark-grade energy at <S^2> ~ 4.8, a genuine singlet at
+2.9 Ha -- are ~3 Hartree apart, with nothing competitive between.
Panel B: the shipped penalty path cannot reach the frontier. For each
kernel_fixed_space + fix_spin_(ss=0) arm, the penalized Rayleigh quotient
their code returns (converged=False) sits 0.27-0.33 Ha above the certified
penalized minimum (our eigsh solver, pyci-anchored).
Data: sqdpen_frontier.npz, sqdpen_theirs.npz (all values re-derived)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RED, BLUE, GRAY = "#c1272d", "#1f5fa8", "0.55"
fr = np.load("sqdpen_frontier.npz", allow_pickle=True)["frontier"].item()

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))

# ---------- Panel A: the frontier ----------
a = ax[0]
for s2v, lab in ((0, "S=0 (singlet)"), (2, "S=1 (triplet)")):
    a.axhline(s2v, color="0.82", lw=1, ls="--", zorder=0)
    a.text(33, s2v + 0.08, lab, color="0.45", fontsize=8, ha="left")
for tag, col, mk, lbl in (("it1", BLUE, "s", "iter-1 space (245$^2$)"),
                          ("it4", RED, "o",
                           "iter-4 space (441$^2$, +45.5 mHa row)")):
    pts = fr[tag]["pts"]
    err = np.array([p["err_mHa"] for p in pts])
    s2 = np.array([p["s2"] for p in pts])
    lam = np.array([p["lam"] for p in pts])
    a.plot(err, s2, "-", color=col, lw=1.3, zorder=2)
    a.plot(err, s2, mk, color=col, ms=6, zorder=3, label=lbl)
    for p in pts:
        if p["lam"] in (0.0, 0.1, 0.2, 0.5):
            dy = 6 if tag == "it1" else -12
            a.annotate(f"$\\lambda${p['lam']:g}",
                       (p["err_mHa"], p["s2"]),
                       textcoords="offset points", xytext=(4, dy),
                       fontsize=7, color=col)
# single combined singlet-floor label (both spaces land at <S^2>=0)
a.annotate("best true singlet in the\nsubspace: +2.9 Ha",
           (fr["it4"]["pts"][-1]["err_mHa"], 0.0),
           textcoords="offset points", xytext=(-8, 30),
           fontsize=7.8, color="0.2", ha="right",
           arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
# S5 full-protocol points overlaid (their loop, penalty on, converged)
s5 = {0.2: (110.447, 3.9337), 0.1: (72.550, 4.1762)}
for lam, (er, s2v) in s5.items():
    a.plot(er, s2v, "*", color="black", ms=13, zorder=4,
           label="full protocol (S5)" if lam == 0.2 else None)
a.set_xscale("log")
a.set_xlim(30, 5000)
a.set_ylim(-0.3, 5.2)
a.set_xlabel("energy error vs exact singlet (mHa, log)")
a.set_ylabel(r"$\langle S^2\rangle$ of the penalized ground")
a.set_title("A. Penalty on: the spin-energy frontier", fontsize=10)
a.legend(fontsize=7.3, loc="center right", framealpha=0.9)

# ---------- Panel B: the shipped kernel can't reach it ----------
b = ax[1]
rows = list(np.load("sqdpen_theirs.npz", allow_pickle=True)["rows"])
# certified penalized minimum per (tag,shift) from the frontier
cert = {}
for tag in ("it1", "it4"):
    for p in fr[tag]["pts"]:
        cert[(tag, p["lam"])] = p["e_h"] + p["lam"] * p["s2"]
labels, gaps, cols = [], [], []
for r in rows:
    key = (r["tag"], r["shift"])
    if key not in cert:
        continue
    gap = (r["pen_rq"] - cert[key]) * 1e3  # mHa above certified pen-min
    mc = "def" if r["max_cycle"] == "default" else str(r["max_cycle"])
    labels.append(f"{r['tag']}\n$\\lambda${r['shift']:g}, mc={mc}")
    gaps.append(gap)
    cols.append(RED if r["shift"] == 0.2 else BLUE)
x = np.arange(len(labels))
b.bar(x, gaps, color=cols, width=0.62, zorder=3)
b.axhline(0, color="black", lw=1.1)
b.text(len(x) - 0.5, 6, "certified penalized minimum", fontsize=8,
       ha="right", va="bottom", color="0.3")
for xi, g in zip(x, gaps):
    b.text(xi, g + 4, f"+{g:.0f}", ha="center", fontsize=7.5)
b.text(0.02, 0.96, "all arms: converged = False", transform=b.transAxes,
       fontsize=8.5, va="top", color=RED, style="italic")
b.set_xticks(x)
b.set_xticklabels(labels, fontsize=7)
b.set_ylabel("penalized Rayleigh quotient\nabove certified min (mHa)")
b.set_title("B. The shipped penalty path never converges", fontsize=10)
b.set_ylim(0, max(gaps) * 1.22)

fig.tight_layout()
fig.savefig("fig_penalty_frontier.png", dpi=150, bbox_inches="tight")
print("wrote fig_penalty_frontier.png")
print(f"panel B gaps (mHa): {[round(g,1) for g in gaps]}")
