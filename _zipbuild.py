"""Rebuild launch/psl_sqd_supplementary.zip (forward-slash arcnames).
Explicit manifest; hard-fails on any missing file."""
import os
import zipfile

CORE = [
    "paper_sqd_spin_audit.md", "paper_sqd_spin_audit.tex",
    "paper_sqd_spin_audit.pdf",
    "launch/AUDIT_TRAIL.md:AUDIT_TRAIL.md",
    "launch/REPRO_MAP.md:REPRO_MAP.md",
    "s2audit_master.csv", "fig_s2_vs_d.png", "fig_err_vs_d.png",
    "fig_coupon.png", "fig_flagship_s4.png", "fig_penalty_frontier.png",
    "_s2audit.py", "_s2fig.py", "_figflagship.py", "_s2matrix.sh",
    "_s2matrix.run.log",
    "_paperaudit.py", "_paperaudit.out", "_pdfcheck.py",
    "_gramxcheck.py", "_gramxcheck.out",
    "_sqdship.py", "_sqdship_delta.py", "_sqdship.run.log",
    "_fairfight.py", "_n2xval.py", "_diazene_targets.py",
    "_hcicert.py", "_probe_pyci.py",
    "_couponlaw.py", "_figcoupon.py", "_costmodel.py", "_ibmuniform.py",
    "_ibmuniform2.py", "spin_inversion_is_not_singlet.py",
    "_ibmsamples.py", "_ibmsamples.run.log",
    "_ibmsamples4.py", "_ibmsamples4.run.log",
    "_lucjfe2.py", "_lucjfe2.run.log", "_lucjfe2b.run.log",
    "_lucjopt.py", "_lucjopt.run.log",
    "_sqdraw.py", "_sqdraw_synth.py",
    "_sqdpen.py", "_sqdpen_diagprobe.py", "_sqdpen_s3b.py", "_sqdpenfig.py",
    "_sqdpen_sq.py", "_sqdpen_true.py",
    "_sqdpen.run.log", "_sqdpen_sq.run.log", "_sqdpen_true.run.log",
    "_degen.py", "_degen_dense.py", "_degen_flip.py", "_degen_table.py",
    "_degen.run.log", "_s6stress.py", "_thrscan.py",
    "_flagsolve.py", "_flagladder.sh",
    "flagsolve_box1_attempt1_live.log", "flagsolve_box1_attempt2_live.log",
    "flagsolve_box1_attempt3_live.log",
    "_gram7500.py", "gram7500.run.log",
    "_gram7500s4.py", "_s4gate.py", "gram7500s4_live.log",
    "_s4states.py", "_s6moments.py",
    "_verm.py",
    "_tb7500box.py", "tb7500_live.log", "tb7500_bootstrap.log",
    "tb7500_build2_live.log", "tb7500_build2_bootstrap.log",
]
NPZ = [
    "s2audit_fe2s2.npz", "s2audit_fe2s2bs.npz", "s2audit_fe2s2deep.npz",
    "s2audit_fe4s4.npz",
    "sqdship_vec.npz", "sqdship_counts.npz",
    "sqdship_symm1.npz", "sqdship_symm0.npz",
    "sqdpen_frontier.npz", "sqdpen_theirs.npz", "sqdpen_spaces.npz",
    "sqdpen_loop_l10.npz", "sqdpen_loop_l20.npz", "sqdpen_s3b.npz",
    "sqdpen_sq_it4.npz", "sqdpen_true_it4.npz", "sqdpen_true_it1.npz",
    "degen.npz", "degen_dense.npz", "s6stress.npz", "thrscan.npz",
    "fairfight_dz.npz", "fairfight_dzsinglet.npz",
    "fairfight_dzs11.npz", "fairfight_dzs23.npz",
    "n2_hci.npz", "n2_qsci.npz", "n2_ours.npz", "n2_ref.npz",
    "n2_ints.npz",
    "dz_r090.00.npz", "dz_i180.00.npz",
    "n2_r200/n2_hci.npz", "n2_r200/n2_qsci.npz", "n2_r200/n2_ours.npz",
    "n2_r200/n2_ref.npz", "n2_r200/n2_ints.npz",
    "couponlaw.npz", "costmodel.npz", "ibmuniform.npz", "ibmuniform_stats.npz",
    "ibmship_symm1.npz", "ibmship_symm0.npz",
    "ibm4ship_symm1.npz", "ibm4ship_symm0.npz",
    "lucj2_stats.npz", "lucj2_ccsd.npz", "lucj2ship_counts.npz",
    "lucj2ship_symm1.npz", "lucj2ship_symm0.npz",
    "lucjopt_stats.npz", "lucjopt_state.npz",
    "lucjoptship_symm1.npz", "lucjoptship_symm0.npz",
    "sqdraw_n500_symm1_s17.npz", "sqdraw_n1000_symm0_s17.npz",
    "sqdraw_n1000_symm1_s43.npz",
    "sqdraw_results_aws/sqdraw_n500_symm1_s17.npz",
    "sqdraw_results_aws/sqdraw_n1000_symm1_s17.npz",
    "sqdraw_results_aws/sqdraw_n2000_symm1_s17.npz",
    "lucjopt_marginals.npz",
    "flagsolve_n300.npz", "flagsolve_n500.npz", "flagsolve_n600.npz",
    "flagsolve_n1000.npz", "flagsolve_n2000.npz", "flagsolve_n2000_box1.npz",
    "gram7500.npz", "gram7500_ck1.npz", "gram7500_ck2.npz",
    "gram7500_ck3.npz",
    "gram7500_s4ck0.npz", "gram7500_s4ck1.npz", "gram7500_s4ck2.npz",
    "gram7500_s4ck3.npz", "gram7500_s4.npz", "s4states.npz", "s6moments.npz",
    "verm_0_3_0.npz", "verm_0_12_1.npz",
    "tb7500_strings.npz", "tb7500_result_n7500.npz",
    # flagsolve_n7500_box2.npz (413 MB), gram7500_restart.npz (1.3 GB),
    # gram7500_s4_restart.npz (1.8 GB) and tb7500_ck.npz (1.35 GB) ship as
    # standalone companion files in the Zenodo record, not inside this zip.
]
RUN1 = [  # self-audit event #5: reference-choice cautionary observation
    "_lucjfe.py", "_lucjfe.run.log", "lucj_stats.npz", "lucj_ccsd.npz",
    "lucjship_counts.npz", "lucjship_symm1.npz", "lucjship_symm0.npz",
]
KIT = [  # reproduction kit: newcomer entry docs + friendly reproducers, so the
         # zip is a complete, runnable repository (added 2026-07-19)
    "START_HERE.md", "VALIDATE.md", "FALSIFY.md", "EXPECTED_OUTPUTS.md",
    "setup.sh", "reproduce_level1.py", "higher_moment_example.py",
    "sampling_wall_calculator.py",
    # repo-meta pulled from the mirror (these live only in repo_sqd_spin_audit/)
    "repo_sqd_spin_audit/README.md:README.md",
    "repo_sqd_spin_audit/requirements.txt:requirements.txt",
    "repo_sqd_spin_audit/LICENSE:LICENSE",
    "repo_sqd_spin_audit/CITATION.cff:CITATION.cff",
]

OUT = "launch/psl_sqd_supplementary.zip"
missing = []
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for spec in CORE + NPZ + KIT:
        if ":" in spec:
            src, arc = spec.split(":")
        else:
            src = arc = spec
        if not os.path.exists(src):
            missing.append(spec)
            continue
        z.write(src, arc.replace("\\", "/"))
    for f in RUN1:
        if not os.path.exists(f):
            missing.append(f)
            continue
        z.write(f, "lucj_run1_reference_choice_observation/" + f)

assert not missing, f"MISSING: {missing}"
sz = os.path.getsize(OUT)
with zipfile.ZipFile(OUT) as z:
    n = len(z.namelist())
print(f"{OUT}: {n} entries, {sz/1e6:.1f} MB")
