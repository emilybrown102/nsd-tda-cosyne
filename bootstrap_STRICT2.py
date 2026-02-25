#!/usr/bin/env python
"""
============================================================
Goal:
  1) Bootstrap subsample RDMs to find robust H1 features (loops)
  2) Save robust cluster summaries (NO VR graph visualization)

Inputs:
  - STRICT2/STRICT3 raw RDMs (npy) in:
      RDMs_Final/subjects/subjXX/ROI-<ROI>_crossnobis_STRICT*_sessionSplit_raw.npy

Outputs:
  - robust_clusters_H1.csv
  - per subj×ROI:
      bootstrap_topL_features_H1.csv
      consensus_clusters_H1.csv

Requires:
  pip install ripser pandas numpy

Notes:
  - Persistent homology uses ripser on distance matrices.
  - Distances can be slightly negative (crossnobis). We shift by a constant
    so min off-diagonal becomes > 0 (topology preserved up to a constant shift).
============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from ripser import ripser


# =========================
# CONFIG (EDIT THESE)
# =========================

ROOT = Path(r"D:\Emily\NSD_cogsci_sandbox")
RDM_ROOT = ROOT / "COGS401" / "COSYNE" / "RDMs_Final" / "subjects"

STRICT_TAG = "STRICT2"   # "STRICT2" or "STRICT3"
SUBJECTS = [1, 2, 3, 4, 5]
ROIS = ["V1", "V2", "V3", "V4", "LO2", "MT", "PH", "STSva", "PIT"]

# Bootstrap settings
B = 200
SUBSAMPLE_FRAC = 0.80
RANDOM_SEED = 123

# Persistent homology settings
MAXDIM = 1  # only need H1 for loops
SHIFT_EPS = 1e-9
PERSIST_FLOOR_FRAC = 0.05   # persistence >= 5% of max distance (per bootstrap replicate)
TOP_L = 10                  # keep top L H1 features per bootstrap replicate

# Consensus clustering
CLUSTER_TOL_FRAC = 0.02     # tolerance in birth/death space as fraction of maxdist_full
MIN_HITS = max(5, int(0.05 * B))

# Output folder
OUT_ROOT = ROOT / "COGS401" / "COSYNE" / "RDMs_Final" / "PH_VR_ripser" / f"bootstrap{B}_subsample_consensus_{STRICT_TAG}"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# =========================
# Helpers
# =========================

def load_rdm(subj: int, roi: str) -> np.ndarray:
    f = RDM_ROOT / f"subj{subj:02d}" / f"ROI-{roi}_crossnobis_{STRICT_TAG}_sessionSplit_raw.npy"
    if not f.exists():
        raise FileNotFoundError(f"Missing RDM: {f}")
    D = np.load(f).astype(np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"RDM not square: {f} shape={D.shape}")
    if not np.isfinite(D).all():
        raise ValueError(f"RDM has non-finite values: {f}")
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def shift_to_nonnegative(D: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, float]:
    n = D.shape[0]
    off = D[~np.eye(n, dtype=bool)]
    m = float(np.min(off))
    if m >= 0:
        return D.copy(), 0.0
    shift = (-m) + eps
    D2 = D + shift
    np.fill_diagonal(D2, 0.0)
    return D2, shift


def max_offdiag(D: np.ndarray) -> float:
    n = D.shape[0]
    off = D[~np.eye(n, dtype=bool)]
    return float(np.max(off))


def subsample_indices(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    k = int(round(frac * n))
    k = max(5, min(n, k))
    return rng.choice(n, size=k, replace=False)


def ripser_H1(D: np.ndarray):
    return ripser(D, distance_matrix=True, maxdim=MAXDIM, do_cocycles=False)


def diagram_H1(res) -> np.ndarray:
    dgms = res["dgms"]
    return dgms[1] if len(dgms) > 1 else np.zeros((0, 2), dtype=float)


def finite_persistence(dgm: np.ndarray) -> np.ndarray:
    if dgm.size == 0:
        return np.zeros((0,), dtype=float)
    b = dgm[:, 0]
    d = dgm[:, 1]
    finite = np.isfinite(d)
    pers = np.zeros_like(b)
    pers[finite] = d[finite] - b[finite]
    pers[~finite] = np.nan
    return pers


@dataclass
class Feature:
    birth: float
    death: float
    persistence: float
    maxdist: float
    boot_i: int


def extract_top_features(
    dgmH1: np.ndarray, maxdist_here: float, top_L: int, floor_frac: float, boot_i: int
) -> List[Feature]:
    if dgmH1.size == 0:
        return []
    pers = finite_persistence(dgmH1)
    finite = np.isfinite(pers)
    if not finite.any():
        return []

    births = dgmH1[finite, 0]
    deaths = dgmH1[finite, 1]
    pers_f = pers[finite]

    floor = floor_frac * maxdist_here
    keep = pers_f >= floor
    births, deaths, pers_f = births[keep], deaths[keep], pers_f[keep]
    if births.size == 0:
        return []

    order = np.argsort(-pers_f)[: min(top_L, len(pers_f))]

    out: List[Feature] = []
    for j in order:
        out.append(
            Feature(
                birth=float(births[j]),
                death=float(deaths[j]),
                persistence=float(pers_f[j]),
                maxdist=float(maxdist_here),
                boot_i=int(boot_i),
            )
        )
    return out


def cluster_features(features: List[Feature], tol: float) -> List[List[Feature]]:
    clusters: List[List[Feature]] = []
    for f in features:
        placed = False
        for c in clusters:
            cb = np.mean([x.birth for x in c])
            cd = np.mean([x.death for x in c])
            if max(abs(f.birth - cb), abs(f.death - cd)) <= tol:
                c.append(f)
                placed = True
                break
        if not placed:
            clusters.append([f])
    return clusters


def centroid(cluster: List[Feature]) -> Tuple[float, float, float]:
    b = float(np.mean([x.birth for x in cluster]))
    d = float(np.mean([x.death for x in cluster]))
    p = float(np.mean([x.persistence for x in cluster]))
    return b, d, p


def run_bootstrap_for_subject_roi(
    D_full: np.ndarray, rng: np.random.Generator
) -> Tuple[pd.DataFrame, List[List[Feature]]]:
    D_shift, _ = shift_to_nonnegative(D_full, eps=SHIFT_EPS)
    n = D_shift.shape[0]

    all_feats: List[Feature] = []

    for b in range(B):
        idx = subsample_indices(n, SUBSAMPLE_FRAC, rng)
        Ds = D_shift[np.ix_(idx, idx)]
        md = max_offdiag(Ds)

        res = ripser_H1(Ds)
        dgm = diagram_H1(res)

        feats = extract_top_features(dgm, md, TOP_L, PERSIST_FLOOR_FRAC, boot_i=b)
        all_feats.extend(feats)

    if len(all_feats) == 0:
        return pd.DataFrame(), []

    md_full = max_offdiag(D_shift)
    tol = CLUSTER_TOL_FRAC * md_full
    clusters = cluster_features(all_feats, tol=tol)

    feats_df = pd.DataFrame(
        [{
            "boot": f.boot_i,
            "birth": f.birth,
            "death": f.death,
            "persistence": f.persistence,
            "maxdist_boot": f.maxdist,
        } for f in all_feats]
    )
    return feats_df, clusters


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    robust_rows = []

    for subj in SUBJECTS:
        for roi in ROIS:
            print(f"\n=== subj{subj:02d} {roi} ===")
            D = load_rdm(subj, roi)
            D_shift, shift_added = shift_to_nonnegative(D, eps=SHIFT_EPS)
            md_full = max_offdiag(D_shift)

            feats_df, clusters = run_bootstrap_for_subject_roi(D, rng)
            if feats_df.empty or len(clusters) == 0:
                print("  No H1 features survived floor/top-L in bootstraps.")
                continue

            # summarize clusters
            summaries = []
            for ci, cl in enumerate(clusters):
                hits = len(set([f.boot_i for f in cl]))
                hit_rate = hits / B
                b0, d0, p0 = centroid(cl)
                summaries.append({
                    "cluster_id": ci,
                    "hits": hits,
                    "hit_rate": hit_rate,
                    "birth": b0,
                    "death": d0,
                    "persistence": p0,
                })

            summ_df = pd.DataFrame(summaries).sort_values(["hit_rate", "persistence"], ascending=[False, False])
            summ_df["robust"] = summ_df["hits"] >= MIN_HITS

            # store robust rows to CSV
            for _, r in summ_df.iterrows():
                robust_rows.append({
                    "subject": subj,
                    "roi": roi,
                    "cluster_id": int(r["cluster_id"]),
                    "hits": int(r["hits"]),
                    "hit_rate": float(r["hit_rate"]),
                    "birth": float(r["birth"]),
                    "death": float(r["death"]),
                    "persistence": float(r["persistence"]),
                    "maxdist_full": float(md_full),
                    "shift_added": float(shift_added),
                    "robust": bool(r["robust"]),
                    "B": int(B),
                    "subsample_frac": float(SUBSAMPLE_FRAC),
                    "persist_floor_frac": float(PERSIST_FLOOR_FRAC),
                    "top_L": int(TOP_L),
                    "cluster_tol_frac": float(CLUSTER_TOL_FRAC),
                    "min_hits": int(MIN_HITS),
                    "strict_tag": STRICT_TAG,
                })

            # Save transparency tables
            out_dir = OUT_ROOT / f"subj{subj:02d}" / f"ROI-{roi}"
            out_dir.mkdir(parents=True, exist_ok=True)
            feats_df.to_csv(out_dir / "bootstrap_topL_features_H1.csv", index=False)
            summ_df.to_csv(out_dir / "consensus_clusters_H1.csv", index=False)

    # master robust CSV
    out_csv = OUT_ROOT / "robust_clusters_H1.csv"
    pd.DataFrame(robust_rows).to_csv(out_csv, index=False)
    print(f"\n✅ Wrote robust clusters: {out_csv}")


if __name__ == "__main__":
    main()
