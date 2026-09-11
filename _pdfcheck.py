import pypdf
import re
import sys

# Defaults to the frozen preprint. Pass one or more PDFs to probe a different
# build; the JCTC submission needs the pair, because content that lives in the
# main text of the preprint now lives in the Supporting Information.
#     python _pdfcheck.py paper_jctc_main.pdf paper_jctc_si.pdf
PDFS = sys.argv[1:] or ["paper_sqd_spin_audit.pdf"]

full = ""
for n, path in enumerate(PDFS):
    r = pypdf.PdfReader(path)
    print(f"{path}: {len(r.pages)} pages")
    if n == 0:
        print("p1:", r.pages[0].extract_text()[:160].replace("\n", " "))
    full += "\n" + "\n".join(p.extract_text() for p in r.pages)
# Strip ALL whitespace for matching: robust both to reflow line/page breaks and to
# fonts whose PDF text extraction drops inter-word spaces (e.g. newtx/Times in justified text).
full_ns = re.sub(r"\s+", "", full)
for probe in ["trade-off", "58,644,960", "own data archive", "coupon",
              "0.000329", "does not depend on pricing the samples",
              "Package version",
              "falsifiable", "spin-pure triplet", "0.445",
              "Principal findings", "References", "138.740", "12.047",
              "second moment", "scalar mean does not identify",
              "numerically certified", "72.30", "raw-counts ladder",
              "Independent implementation at the flagship dimension", "127,975,471,122",
              "1.371065, 2.362748, 4.737714",
              "run with the penalty on", "penalty-matched rerun",
              "overlap 1.000000",
              "completion mechanism at each checkpoint", "Completion-mechanism",
              "Dense-adjudicated",
              "spurious 93 mHa",
              "moment-constrained sector bounds",
              "not of tight tolerances"]:
    hit = probe in full or re.sub(r"\s+", "", probe) in full_ns
    print(f"{'OK ' if hit else 'MISS'} {probe!r}")
