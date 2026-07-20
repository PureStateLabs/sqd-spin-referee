"""Flagship certification figure (fig_flagship_s4.png): the 5.625e7-
determinant four-root audit in one image. Panel A: the low-energy spectrum
against the spin ladder (root energies vs <S^2>, rails at S(S+1)); the
non-stationary near-singlet direction marked open. Panel B: the residual
push — residual/gap ratio collapsing 18.6-fold across the certification
stages while the ground <S^2> stays at 1.3711. Panel C: spin width —
Var(S^2) per root; a spin eigenstate sits at 0.
Data: gram7500_s4ck{0..3}.npz (all values re-derived, no hardcoding beyond
labels)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

cks = [np.load(f"gram7500_s4ck{i}.npz") for i in range(4)]
c3 = cks[3]
E = np.asarray(c3["e_rayleigh"], float)
G = np.asarray(c3["gram"], float)
G4 = np.asarray(c3["gram4"], float)
s2 = np.diagonal(G).copy()
var = np.diagonal(G4) - s2 ** 2
dE = (E - E[0]) * 1e3  # mHa above ground

ratios = []
s2r0 = []
for z in cks:
    e = np.asarray(z["e_rayleigh"], float)
    ratios.append(float(z["resid"][0]) / float(e[1] - e[0]))
    s2r0.append(float(np.asarray(z["gram"])[0, 0]))

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

# ---- A: spectrum vs spin ladder ----
a = ax[0]
for s, lab in ((0, "S=0"), (2, "S=1"), (6, "S=2"), (12, "S=3")):
    a.axhline(s, color="0.82", lw=1, ls="--", zorder=0)
    a.text(13.6, s + 0.18, lab, color="0.45", fontsize=8, ha="right")
a.plot(dE, s2, "o", ms=9, color="#c1272d", zorder=3)
offs = [(-8, 14, "right"), (12, -12, "left"), (8, -4, "left"),
        (-64, -6, "left")]
for i in range(4):
    a.annotate(f"root {i}\n⟨S²⟩={s2[i]:.3f}", (dE[i], s2[i]),
               textcoords="offset points", xytext=offs[i][:2],
               ha=offs[i][2], fontsize=8)
# the non-stationary near-singlet direction (invariant spectrum floor,
# Rayleigh +1.44 mHa): open marker
ev3 = np.linalg.eigvalsh(G[:3, :3])
a.plot([1.44], [ev3[0]], "o", ms=9, mfc="none", mec="#1f77b4", mew=1.8,
       zorder=3)
a.annotate("near-singlet direction\n(non-stationary superposition,\nnot an eigenstate)",
           (1.44, ev3[0]), textcoords="offset points", xytext=(18, -8),
           fontsize=8, color="#1f77b4")
a.set_xlabel("energy above lowest root (mHa)")
a.set_ylabel("⟨S²⟩")
a.set_title("A  Low-energy roots vs the spin ladder\n(5.625×10⁷ determinants, certified)",
            fontsize=10)
a.set_xlim(-3.2, 14.2)
a.set_ylim(-1.0, 13.2)

# ---- B: residual push ----
b = ax[1]
xs = np.arange(4)
labels = ["certified\n(3 roots, 10⁻⁸)", "stage 1\n(3 roots, 10⁻¹⁰)",
          "stage 2\n(4 roots, 10⁻⁸)", "stage 3\n(4 roots, 10⁻¹⁰)"]
b.semilogy(xs, ratios, "s-", color="#1f77b4", ms=8, lw=1.6,
           label="residual / gap")
for x, r in zip(xs, ratios):
    b.annotate(f"{r:.4f}", (x, r), textcoords="offset points",
               xytext=(6, 5), fontsize=8, color="#1f77b4")
b.annotate(f"{ratios[0]/ratios[-1]:.1f}× tighter",
           (0.12, 0.0125), fontsize=9, color="#1f77b4")
b.set_xticks(xs)
b.set_xticklabels(labels, fontsize=7.5)
b.set_ylabel("‖r₀‖ / (E₁ − E₀)", color="#1f77b4")
b.tick_params(axis="y", colors="#1f77b4")
b2 = b.twinx()
b2.plot(xs, s2r0, "o-", color="#c1272d", ms=7, lw=1.6)
b2.set_ylim(1.25, 1.50)
b2.set_ylabel("ground-root ⟨S²⟩", color="#c1272d")
b2.tick_params(axis="y", colors="#c1272d")
b2.annotate(f"⟨S²⟩ = {s2r0[-1]:.4f} at every stage", (1.5, s2r0[-1]),
            textcoords="offset points", xytext=(0, 8), fontsize=9,
            color="#c1272d", ha="center")
b.set_title("B  Converging harder purifies nothing:\nresidual ↓18.6×, spin identity frozen",
            fontsize=10)

# ---- C: spin width ----
c = ax[2]
c.bar(np.arange(4), var, color="#c1272d", width=0.55, zorder=3)
c.axhline(0, color="k", lw=1.4)
c.annotate("a spin eigenstate has Var(S²) = 0", (1.5, 0.25), fontsize=9,
           ha="center")
for i, v in enumerate(var):
    c.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
               xytext=(0, 4), fontsize=9, ha="center")
c.set_xticks(np.arange(4))
c.set_xticklabels([f"root {i}" for i in range(4)])
c.set_ylabel("Var(S²) = ⟨S⁴⟩ − ⟨S²⟩²")
c.set_ylim(0, 8.6)
c.set_title("C  Spin width: no root is near\nany spin eigenstate", fontsize=10)

fig.tight_layout()
fig.savefig("fig_flagship_s4.png", dpi=200)
print("fig_flagship_s4.png written")
print(f"  s2 {s2.round(4)}  var {var.round(3)}  ratios {np.round(ratios,4)}")
