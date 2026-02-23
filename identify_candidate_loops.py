#!/usr/bin/env python
r"""
Select candidate H1 loops per subject×ROI and ALSO export the full H1 feature list.

Outputs (in RDMs_Final/PH_VR_ripser/):
  - candidate_loops_H1.csv
  - candidate_loops_summary_byROI.csv
  - candidate_loops_summary_bySubjROI.csv
  - all_H1_features_bySubjROI.csv   <-- NEW (for --all sanity validation)

Definition (defensible):
  Candidate loop = among the strongest H1 features within each subject×ROI:
    - finite death only
    - persistence = death - birth
    - persistence >= percentile_cut (e.g., 95th percentile) within that subj×ROI
    - keep at most top_k loops
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Emily\NSD_cogsci_sandbox")
PH_ROOT = ROOT / "COGS401" / "COSYNE" / "RDMs_Final" / "PH_VR_ripser"

SUBJECTS = [1, 2, 3, 4, 5]  # you excluded 6,8
PERCENTILE_CUT = 95         # within each subj×ROI
TOP_K = 5                   # cap

def load_dgm_h1(subj: int, roi: str) -> np.ndarray:
    f = PH_ROOT / f"subj{subj:02d}" / f"ROI-{roi}" / "dgm_H1.npy"
    if not f.exists():
        raise FileNotFoundError(f)
    return np.load(f)

def main():
    rows = []       # candidates
    all_rows = []   # NEW: all finite H1 features (unfiltered, per subj×ROI)

    for subj in SUBJECTS:
        subj_dir = PH_ROOT / f"subj{subj:02d}"
        if not subj_dir.exists():
            continue

        roi_dirs = sorted([p for p in subj_dir.glob("ROI-*") if p.is_dir()])

        for roi_dir in roi_dirs:
            roi = roi_dir.name.replace("ROI-", "")
            dgm_file = roi_dir / "dgm_H1.npy"
            if not dgm_file.exists():
                continue

            dgm = np.load(dgm_file)
            if dgm.size == 0:
                continue

            birth = dgm[:, 0]
            death = dgm[:, 1]

            # keep only finite deaths
            finite = np.isfinite(death)
            birth_f = birth[finite]
            death_f = death[finite]
            if birth_f.size == 0:
                continue

            pers_f = death_f - birth_f

            # ----------------------------
            # NEW: write ALL finite H1 features
            # ----------------------------
            for i in range(birth_f.size):
                all_rows.append({
                    "subject": subj,
                    "roi": roi,
                    "birth": float(birth_f[i]),
                    "death": float(death_f[i]),
                    "persistence": float(pers_f[i]),
                    "source_file": str(dgm_file),
                })

            # ----------------------------
            # Candidate selection (UNCHANGED logic)
            # ----------------------------
            cut = np.percentile(pers_f, PERCENTILE_CUT)

            keep_idx = np.where(pers_f >= cut)[0]
            if keep_idx.size == 0:
                continue

            keep_idx = keep_idx[np.argsort(pers_f[keep_idx])[::-1]]
            keep_idx = keep_idx[:TOP_K]

            for rank, idx in enumerate(keep_idx, start=1):
                rows.append({
                    "subject": subj,
                    "roi": roi,
                    "rank_within_roi": rank,
                    "birth": float(birth_f[idx]),
                    "death": float(death_f[idx]),
                    "persistence": float(pers_f[idx]),
                    "percentile_cut_used": float(cut),
                    "percentile_cut": PERCENTILE_CUT,
                    "top_k_cap": TOP_K,
                })

    # --- Save candidates ---
    out_df = pd.DataFrame(rows)
    if len(out_df) > 0:
        out_df = out_df.sort_values(["subject", "roi", "rank_within_roi"])

    out_csv = PH_ROOT / "candidate_loops_H1.csv"
    out_df.to_csv(out_csv, index=False)

    # --- Save ALL finite features (NEW) ---
    all_df = pd.DataFrame(all_rows)
    all_csv = PH_ROOT / "all_H1_features_bySubjROI.csv"
    all_df.to_csv(all_csv, index=False)

    # --- Summaries (unchanged) ---
    if len(out_df) > 0:
        summary = out_df.groupby(["roi"]).size().reset_index(name="n_candidates_total")
        summary2 = out_df.groupby(["subject", "roi"]).size().reset_index(name="n_candidates_subj_roi")
    else:
        summary = pd.DataFrame(columns=["roi", "n_candidates_total"])
        summary2 = pd.DataFrame(columns=["subject", "roi", "n_candidates_subj_roi"])

    summary_byroi_csv = PH_ROOT / "candidate_loops_summary_byROI.csv"
    summary_bysubjroi_csv = PH_ROOT / "candidate_loops_summary_bySubjROI.csv"
    summary.to_csv(summary_byroi_csv, index=False)
    summary2.to_csv(summary_bysubjroi_csv, index=False)

    print("Wrote:")
    print(" ", out_csv)
    print(" ", all_csv)
    print(" ", summary_byroi_csv)
    print(" ", summary_bysubjroi_csv)

if __name__ == "__main__":
    main()
