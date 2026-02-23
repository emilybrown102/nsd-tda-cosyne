#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- CONFIG ----------------
ROBUST_CSV = Path(r"D:\Emily\NSD_cogsci_sandbox\COGS401\COSYNE\RDMs_Final\PH_VR_ripser\bootstrap200_subsample_consensus\robust_clusters_H1.csv")

OUT_DIR = Path(r"D:\Emily\NSD_cogsci_sandbox\COGS401\COSYNE\RDMs_Final\PH_VR_ripser\poster_panels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- LOAD ----------------
if not ROBUST_CSV.exists():
    raise FileNotFoundError(f"Could not find: {ROBUST_CSV}")

rob = pd.read_csv(ROBUST_CSV)
rob.columns = [c.strip() for c in rob.columns]

# Basic required columns
required = {"subject", "roi", "cluster_id", "hit_rate"}
missing = required - set(rob.columns)
if missing:
    raise RuntimeError(
        f"robust_clusters_H1.csv is missing columns: {missing}\n"
        f"Found columns: {list(rob.columns)}"
    )

# ---------------- PLOT 1: # robust clusters per ROI (all subjects) ----------------
cnt_roi = rob.groupby("roi")["cluster_id"].nunique().sort_values(ascending=False)

plt.figure(figsize=(10, 4))
cnt_roi.plot(kind="bar")
plt.ylabel("# robust H1 clusters")
plt.title("Robust H1 features per ROI (all subjects)")
plt.tight_layout()
plt.savefig(OUT_DIR / "robust_count_per_roi.png", dpi=300)
plt.close()

# ---------------- PLOT 2: max hit_rate per ROI ----------------
mx_hit = rob.groupby("roi")["hit_rate"].max().sort_values(ascending=False)

plt.figure(figsize=(10, 4))
mx_hit.plot(kind="bar")
plt.ylim(0, 1.0)
plt.ylabel("max hit_rate")
plt.title("Max robustness (hit_rate) per ROI")
plt.tight_layout()
plt.savefig(OUT_DIR / "robust_max_hitrate_per_roi.png", dpi=300)
plt.close()

# ---------------- PLOT 3: per-subject counts (heatmap-style table) ----------------
# rows=subject, cols=ROI, values=#robust clusters
tab = (rob.groupby(["subject", "roi"])["cluster_id"]
         .nunique()
         .unstack(fill_value=0)
         .sort_index())

plt.figure(figsize=(12, 4))
plt.imshow(tab.values, aspect="auto")
plt.xticks(range(tab.shape[1]), tab.columns, rotation=45, ha="right")
plt.yticks(range(tab.shape[0]), tab.index)
plt.colorbar(label="# robust clusters")
plt.title("Robust H1 clusters per subject × ROI")
plt.tight_layout()
plt.savefig(OUT_DIR / "robust_counts_subject_by_roi_heatmap.png", dpi=300)
plt.close()

# ---------------- PLOT 4 (optional): hit_rate distribution ----------------
plt.figure(figsize=(6, 4))
plt.hist(rob["hit_rate"].values, bins=15)
plt.xlabel("hit_rate")
plt.ylabel("count")
plt.title("Distribution of robust-cluster hit rates")
plt.tight_layout()
plt.savefig(OUT_DIR / "robust_hit_rate_hist.png", dpi=300)
plt.close()

print("Saved robustness summary outputs to:", OUT_DIR)
print("Total robust clusters:", rob["cluster_id"].nunique(), "(unique cluster_id values across full file may repeat across ROI/subject)")