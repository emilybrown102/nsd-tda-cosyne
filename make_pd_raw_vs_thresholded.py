#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- CONFIG ----------------
BASE = Path(r"D:\Emily\NSD_cogsci_sandbox\COGS401\COSYNE\RDMs_Final\PH_VR_ripser")

# raw H1 file per subj/roi:
# BASE/subj0x/ROI-(ROI)/dgm_H1.npy
SUBJECTS = [1,2,3,4,5]  # edit
ROIS = ["V1","V2","V3","V4","LO2","MT","PH","STSva","PIT"]  # edit

# robust features file from bootstrap (YOU must point this to the exact CSV)
ROBUST_CSV = Path(
    r"D:\Emily\NSD_cogsci_sandbox\COGS401\COSYNE\RDMs_Final\PH_VR_ripser"
    r"\bootstrap200_subsample_consensus\robust_clusters_H1.csv"
)

OUT_DIR = BASE / "poster_panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- HELPERS ----------------
def roi_dir(sid: int, roi: str) -> Path:
    return BASE / f"subj{sid:02d}" / f"ROI-{roi}"

def load_dgm(path: Path):
    if not path.exists():
        return np.empty((0, 2), dtype=float)
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim == 1 and arr.shape[0] == 2:
        arr = arr.reshape(1, 2)
    return arr

def plot_pd(ax, dgm: np.ndarray, title: str, alpha: float = 0.35):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    if dgm is None or len(dgm) == 0:
        ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
        return
    dgm = np.asarray(dgm, float)
    b = dgm[:,0]; d = dgm[:,1]
    finite = np.isfinite(d)
    b = b[finite]; d = d[finite]
    ax.scatter(b, d, s=10, alpha=alpha)
    mn = float(min(np.min(b), np.min(d)))
    mx = float(max(np.max(b), np.max(d)))
    ax.plot([mn, mx], [mn, mx], linewidth=1)

# ---------------- MAIN ----------------
def main():
    if not ROBUST_CSV.exists():
        raise RuntimeError(f"ROBUST_CSV not found: {ROBUST_CSV}\n"
                           f"Please set ROBUST_CSV to the exact file name in robust_cycycles_H1.")

    rob = pd.read_csv(ROBUST_CSV)
    rob.columns = [c.strip() for c in rob.columns]

    # Expect these columns; if yours differ, tell me the header row and I'll map them.
    required = {"subject","roi","cluster_id","hit_rate","birth_median","death_median"}
    missing = required - set(rob.columns)
    if missing:
        raise RuntimeError(f"Robust CSV missing columns {missing}. "
                           f"Paste the CSV header row and I will adapt the script.")

    for sid in SUBJECTS:
        for roi in ROIS:
            d1p = roi_dir(sid, roi) / "dgm_H1.npy"
            if not d1p.exists():
                continue
            d1 = load_dgm(d1p)

            sub = rob[(rob.subject==sid) & (rob.roi==roi)].copy()

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle(f"subj{sid:02d} {roi} — H1 raw vs bootstrap-robust features", fontsize=14)

            # Left: raw
            plot_pd(axes[0], d1, "Raw H1 persistence diagram", alpha=0.35)

            # Right: raw + robust overlays
            plot_pd(axes[1], d1, "Raw H1 + robust clusters (bootstrap)", alpha=0.15)

            if len(sub) == 0:
                axes[1].text(0.5, 0.5, "No robust H1 clusters", ha="center", va="center",
                             transform=axes[1].transAxes)
            else:
                # plot robust cluster medians, size by hit_rate
                sizes = 200 * (0.5 + sub["hit_rate"].values)  # scaled for visibility
                axes[1].scatter(
                    sub["birth_median"].values,
                    sub["death_median"].values,
                    s=sizes,
                    marker="*",
                    zorder=5
                )
                for _, r in sub.iterrows():
                    axes[1].text(
                        float(r["birth_median"]),
                        float(r["death_median"]),
                        f"c{int(r['cluster_id'])}\nh={float(r['hit_rate']):.2f}",
                        fontsize=8
                    )

            out = OUT_DIR / f"H1_PD_raw_vs_bootstrap_subj{sid:02d}_{roi}.png"
            plt.tight_layout()
            plt.savefig(out, dpi=300)
            plt.close(fig)

    print("Saved H1 PD panels to:", OUT_DIR)

if __name__ == "__main__":
    main()