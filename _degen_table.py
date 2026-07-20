"""Render the completion-mechanism table (paper Section 2.6) from degen.npz
+ degen_dense.npz, in md and tex forms, plus the aggregate bounds used in
prose. The fe4s4 1106->2097 row is dense-adjudicated (iterative Davidson
returned one member per degenerate pair there); its iterative artifact values
are quoted in the note. Sentinel: TABLE_DONE
"""
import numpy as np

recs = [dict(r) for r in np.load("degen.npz", allow_pickle=True)["recs"]]
dd = np.load("degen_dense.npz", allow_pickle=True)

ARM = {"2Fe2S/bs": "[2Fe-2S] BS", "2Fe2S/deep": "[2Fe-2S] aufbau",
       "4Fe4S": "[4Fe-4S]"}


def sci(x, d=1):
    if x == 0:
        return "0"
    m, e = f"{abs(x):.{d}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def sci_md(x, d=1):
    if x == 0:
        return "0"
    m, e = f"{abs(x):.{d}e}".split("e")
    sup = str(int(e)).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m}×10{sup}"


rows = []
for r in recs:
    tag = str(r["system"])
    n0, n1 = int(r["n_orig"]), int(r["n_comp"])
    art = (n0 == 1106)
    if art:
        dE, split = float(dd["dE"]), float(dd["split"])
        p = np.asarray(dd["p_spec"], float)
        coup = float(dd["coup"])
        blk = np.asarray(dd["s2blk"], float)
        off = float(dd["s2off"])
    else:
        dE, split = float(r["dE"]), float(r["split"])
        p = np.array([float(r["p_hi"]), float(r["p_lo"])])
        coup = float(r["coup"])
        blk = np.sort(np.asarray(r["s2_block"], float))
        off = float(r["g_off"])
    rows.append(dict(sys=tag, n0=n0, n1=n1, dE=abs(dE), split=split,
                     p1=p[0], p2=p[1] if len(p) > 1 else float("nan"),
                     coup=coup, blk=blk, off=off, art=art))

print("=== aggregates (excluding none; artifact row = dense values) ===")
print(f"max |dE0|      : {max(r['dE'] for r in rows):.2e}")
print(f"max split      : {max(r['split'] for r in rows):.2e}")
print(f"max p2         : {max(r['p2'] for r in rows):.5f}")
print(f"max (1-p1)     : {max(1-r['p1'] for r in rows):.2e}")
print(f"max coup       : {max(r['coup'] for r in rows):.2e}")
print(f"max s2 offdiag : {max(r['off'] for r in rows):.2e}")
print(f"max |blk spread|: {max(abs(r['blk'][1]-r['blk'][0]) for r in rows):.2e}")
print(f"iter artifact  : split {float(dd['iter_split']):.3e} "
      f"(dense {float(dd['split']):.1e}), e_iter {np.asarray(dd['e_iter'])[:3]}")

print("\n=== markdown ===")
print("| space | D → D′ | \\|ΔE₀\\| (Ha) | pair split (Ha) | p₁, p₂ | ‖P₊Hv‖ (Ha) | S² block |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    mark = "†" if r["art"] else ""
    print(f"| {ARM[r['sys']]} | {r['n0']:,} → {r['n1']:,}{mark} | "
          f"{sci_md(r['dE'])} | {sci_md(r['split'])} | "
          f"{r['p1']:.4f}, {r['p2']:.4f} | {sci_md(r['coup'])} | "
          f"[{r['blk'][0]:.4f}, {r['blk'][1]:.4f}] |")

print("\n=== tex body ===")
for r in rows:
    mark = "$^\\dagger$" if r["art"] else ""
    print(f"{ARM[r['sys']]} & {r['n0']:,} $\\to$ {r['n1']:,}{mark} & "
          f"${sci(r['dE'])}$ & ${sci(r['split'])}$ & "
          f"{r['p1']:.4f}, {r['p2']:.4f} & ${sci(r['coup'])}$ & "
          f"[{r['blk'][0]:.4f}, {r['blk'][1]:.4f}] \\\\")
print("TABLE_DONE")
