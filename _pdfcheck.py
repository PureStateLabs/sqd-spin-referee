import pypdf

r = pypdf.PdfReader("paper_sqd_spin_audit.pdf")
print("pages:", len(r.pages))
t0 = r.pages[0].extract_text()[:160].replace("\n", " ")
print("p1:", t0)
full = "\n".join(p.extract_text() for p in r.pages)
for probe in ["vise", "58,644,960", "own data archive", "coupon",
              "0.000329", "does not depend on pricing the samples",
              "Package version",
              "falsifiable", "spin-pure triplet", "0.445",
              "Principal findings", "References", "138.740", "12.047",
              "second moment", "scalar mean does not identify",
              "numerically certified", "72.30", "raw-counts ladder",
              "kernel closed out", "127,975,471,122",
              "1.371065, 2.362748, 4.737714",
              "The penalty, switched on", "penalty-matched rerun",
              "overlap 1.000000",
              "mechanism, measured per checkpoint", "Completion-mechanism",
              "Dense-adjudicated",
              "spurious 93 mHa",
              "moment-constrained sector bounds",
              "not of tight tolerances"]:
    print(f"{'OK ' if probe in full else 'MISS'} {probe!r}")
