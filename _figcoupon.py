"""fig_coupon.png: the sampling wall as a validated law.
E[U(M)] = sum_i 1-(1-p_i)^M for every measured distribution in the study,
with the actually-sampled unique counts overlaid as markers (the validation).
Log-log; gamma = local slope annotated at the 1e6-shot operating point.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

c = np.load("couponlaw.npz", allow_pickle=True)

fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=160)

curves = []
curves.append(("soup", c["soup_mgrid"], c["soup_eu"],
               "[2Fe-2S] converged HCI vector (170,448 dets)", "#1f77b4",
               [(1e6, len(np.load("sqdship_counts.npz")["a"]))]))
try:
    s = np.load("lucj2_stats.npz")
    curves.append(("lucj2", s["mgrid"], s["eu"],
                   "[2Fe-2S] noiseless LUCJ (their template)", "#d62728",
                   [(float(s["shots"]), int(s["nuniq"]))]))
except FileNotFoundError:
    pass
for tag, label, color in [("n2eq", "N$_2$ eq. HCI vector (85,388 dets)",
                           "#2ca02c"),
                          ("n2r200", "N$_2$ R=2.0 HCI vector", "#9467bd")]:
    if f"{tag}_mgrid" in c.files:
        pts = list(zip(c[f"{tag}_shots"], c[f"{tag}_uniq"]))
        curves.append((tag, c[f"{tag}_mgrid"], c[f"{tag}_eu"], label,
                       color, pts))

gtexts = []
for tag, mg, eu, label, color, pts in curves:
    m = np.asarray(mg, float)
    e = np.asarray(eu, float)
    keep = e > 0.5
    ax.plot(m[keep], e[keep], color=color, lw=2, label=label)
    if pts:
        px = [p[0] for p in pts]
        py = [p[1] for p in pts]
        ax.plot(px, py, "o", color=color, ms=7, mec="white", mew=1.2,
                zorder=5)
    lm, lu = np.log(m), np.log(np.maximum(e, 1e-12))
    j = min(max(np.searchsorted(m, 1e6), 1), len(m) - 2)
    g = (lu[j + 1] - lu[j - 1]) / (lm[j + 1] - lm[j - 1])
    gtexts.append((label.split("(")[0].strip(), g, color))

# gamma legend block (staggered, no overlap)
for i, (lbl, g, color) in enumerate(gtexts):
    ax.text(0.985, 0.05 + 0.055 * i,
            f"$\\gamma_{{10^6}}$ = {g:.2f}  {lbl}", color=color, fontsize=8,
            ha="right", transform=ax.transAxes)

ax.axvline(1e6, color="gray", ls=":", lw=1)
ax.text(0.48, 0.965, "10$^6$ shots (operating point)", fontsize=8,
        color="gray", ha="right", va="top", transform=ax.transAxes,
        rotation=90)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("shots $M$")
ax.set_ylabel("unique determinants  E[U(M)]")
ax.set_title("The sampling wall: closed form (lines) vs actual sampling "
             "(markers)\n$E[U(M)] = \\sum_i 1-(1-p_i)^M$, "
             "$\\gamma = d\\log U / d\\log M$")
ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig("fig_coupon.png")
print("fig_coupon.png written")
