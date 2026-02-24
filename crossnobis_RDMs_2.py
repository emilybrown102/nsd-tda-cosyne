#!/usr/bin/env python
r"""
===========================================================
CROSSNOBIS RDM BUILDER — STRICT 2 REPEATS, BALANCED 1v1 CV
(NSD LH ROIs) + poster-ready visuals

MODELED OFF your STRICT3 script (only minimal changes):
- Reads STRICT2 betas from unaveraged_responses_2reps
- Uses 1-vs-1 folds: repeat_index 0 vs 1 (balanced)
- No extra strict filtering inside this script (STRICT2 extraction already enforces it)
- Crossnobis math unchanged (including /P scaling)
- Across-ROI figure plots CORRELATION (SIMILARITY) with diverging cmap centered at 0
- Group means computed excluding subjects 6 and 8 (subjects 1–5 only),
  while still computing subject-level outputs for all 1–8.

===========================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------- CONFIG ----------------

ROOT = Path(r"D:\Emily\NSD_cogsci_sandbox")

# STRICT2 inputs (produced by your STRICT2 extraction script)
BETAS_DIR = ROOT / "COGS401" / "COSYNE" / "betas" / "crossnobis" / "unaveraged_responses_2reps"
IMG_LIST_FILE = BETAS_DIR / "images_commonAvailableAll8_crossnobis2reps.txt"

OUT_DIR = ROOT / "COGS401" / "COSYNE" / "RDMs_Final"
SUBJ_DIR = OUT_DIR / "subjects"
GROUP_DIR = OUT_DIR / "group"

# Per-subject outputs for all subjects:
SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]

# Group means excluding 6 and 8 (as requested: "just 1–5"):
GROUP_SUBJECTS = [1, 2, 3, 4, 5]

ROIS = ["V1", "V2", "V3", "V4", "PIT", "PH", "STSva", "MT", "LO2"]

REPEATS_REQUIRED = 2
FOLD_A_REPEAT = 0
FOLD_B_REPEAT = 1

# Diagonal noise variance shrinkage (toward global mean)
SHRINK_ALPHA = 0.10
EPS_VAR = 1e-6

# Visual-only shifting for prettier within-ROI RDM heatmaps (never use shifted for stats)
EPS_SHIFT = 1e-6

# Visuals
RDM_CMAP = "viridis"
ACROSS_ROI_CMAP = "bwr"  # diverging for correlation similarity
FIG_DPI = 300

# Font sizes
TITLE_FONTSIZE = 16
TICK_FONTSIZE = 10
VALUE_FONTSIZE = 10


# ---------------- IO ----------------

def load_image_order(img_list_file: Path) -> list[str]:
    if not img_list_file.exists():
        raise RuntimeError(f"Missing image list file: {img_list_file}")
    ids = [line.strip() for line in img_list_file.read_text().splitlines() if line.strip()]
    if not ids:
        raise RuntimeError(f"Image list file is empty: {img_list_file}")
    return ids


def load_subject_parquet(subj: int) -> pd.DataFrame:
    f = BETAS_DIR / f"subj{subj:02d}_betas_commonAvailableAll8_{REPEATS_REQUIRED}reps_ROIs_LH_crossnobis.parquet"
    if not f.exists():
        raise RuntimeError(f"Missing subject parquet: {f}")
    df = pd.read_parquet(f)

    required = {"subject", "ROI", "nsdId", "beta", "session", "repeat_index"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Parquet missing columns {missing} in {f}")

    df["nsdId"] = df["nsdId"].astype(str)
    df["ROI"] = df["ROI"].astype(str)
    df["session"] = df["session"].astype(int)
    df["repeat_index"] = df["repeat_index"].astype(int)
    return df


# ---------------- plotting ----------------

def _nice_heatmap_defaults():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titleweight": "bold",
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
    })


def _wrap_title(title: str, max_chars: int = 44) -> str:
    """Simple word-wrap so titles never clip."""
    words = title.split()
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        add_len = len(w) + (1 if cur else 0)
        if cur_len + add_len > max_chars:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add_len
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def plot_rdm_clean(
    mat: np.ndarray,
    title: str,
    out_png: Path,
    cmap: str,
    tick_labels: list[str] | None = None,
    show_values: bool = False,
    value_fmt: str = "{:.2f}",
    colorbar_label: str = "Dissimilarity",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Clean heatmap:
      - No axis titles
      - Wrapped title to prevent clipping
      - constrained_layout to fit colorbar and title reliably
    """
    _nice_heatmap_defaults()
    n = mat.shape[0]

    figsize = (10.2, 9.0) if tick_labels is not None else (
        (8.5, 7.5) if n > 200 else ((8.0, 7.0) if n > 100 else (7.5, 6.5))
    )

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(mat, interpolation="nearest", cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)

    ax.set_title(_wrap_title(title), fontsize=TITLE_FONTSIZE, pad=10)

    ax.set_xlabel("")
    ax.set_ylabel("")

    cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.02)
    cbar.ax.set_ylabel(colorbar_label, rotation=90)

    if tick_labels is not None and n <= 60:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticklabels(tick_labels)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if show_values:
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, value_fmt.format(mat[i, j]),
                    ha="center", va="center",
                    fontsize=VALUE_FONTSIZE, color="black"
                )

    fig.savefig(out_png, dpi=FIG_DPI)
    plt.close(fig)


# ---------------- math helpers ----------------

def shift_to_nonnegative(rdm: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = float(np.min(rdm))
    return rdm.copy() if m >= 0 else (rdm + (-m + eps))


def save_matrix_csv(mat: np.ndarray, labels: list[str], out_csv: Path) -> None:
    pd.DataFrame(mat, index=labels, columns=labels).to_csv(out_csv)


# ---------------- FOLD BUILDING (STRICT2) ----------------

def build_repeat_split_folds_strict2(df_subj_roi: pd.DataFrame, ids: list[str]) -> np.ndarray:
    """
    Build K=2 folds: repeat_index 0 vs 1 (balanced 1-vs-1).
    Returns X_folds: (2, N, P)
    """
    first_beta = df_subj_roi.iloc[0]["beta"]
    P = len(first_beta)
    grouped = df_subj_roi.groupby(["nsdId", "repeat_index"])

    X = np.zeros((2, len(ids), P), dtype=np.float32)

    missing = 0
    for i, img in enumerate(ids):
        try:
            rowA = grouped.get_group((img, FOLD_A_REPEAT)).iloc[0]
            rowB = grouped.get_group((img, FOLD_B_REPEAT)).iloc[0]
        except KeyError:
            missing += 1
            continue

        a = np.asarray(rowA["beta"], dtype=np.float32)
        b = np.asarray(rowB["beta"], dtype=np.float32)

        if a.shape[0] != P or b.shape[0] != P:
            raise RuntimeError(f"Beta length mismatch for nsdId={img}: got {a.shape[0]} and {b.shape[0]} vs P={P}")

        X[0, i, :] = a
        X[1, i, :] = b

    if missing > 0:
        raise RuntimeError(f"{missing} nsdIds missing repeat 0/1 in ROI fold building; check STRICT2 extraction outputs.")

    return X


# ---------------- NOISE (DIAGONAL) FROM REPEAT RESIDUALS ----------------

def diag_noise_from_repeat_residuals(df_subj_roi: pd.DataFrame, ids: list[str],
                                    shrink_alpha: float = 0.10, eps: float = 1e-6) -> np.ndarray:
    """
    Estimate diagonal noise variance per voxel from repeat residuals:
      resid = beta_repeat - mean_beta_over_repeats_for_same_image

    For STRICT2: repeats = 2.
    """
    grouped = df_subj_roi.groupby("nsdId")

    residuals = []
    for img in ids:
        rows = grouped.get_group(img)

        # We expect exactly 2 rows for STRICT2
        if len(rows) != REPEATS_REQUIRED:
            raise RuntimeError(f"nsdId={img} has {len(rows)} repeats in ROI; expected {REPEATS_REQUIRED}")

        B = np.stack(rows["beta"].to_list()).astype(np.float32)  # (2, P)
        mu = B.mean(axis=0, keepdims=True)
        R = B - mu
        residuals.append(R)

    Rall = np.concatenate(residuals, axis=0)  # (2*N, P)
    var_p = Rall.var(axis=0, ddof=1)

    vbar = float(var_p.mean())
    var_shrunk = (1.0 - shrink_alpha) * var_p + shrink_alpha * vbar

    w = 1.0 / np.sqrt(var_shrunk + eps)
    return w.astype(np.float32)


# ---------------- CROSSNOBIS ----------------

def crossnobis_rdm_fast_diagwhiten(X_folds: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Crossnobis with diagonal noise normalization.
    X_folds: (2, N, P)
    """
    X = X_folds.astype(np.float32, copy=False)
    K, N, _ = X.shape
    if K < 2:
        raise RuntimeError("Need at least 2 folds for crossnobis.")

    Y = X * w.reshape(1, 1, -1)

    Ya = Y[0]
    Yb = Y[1]

    # ✅ critical scaling by number of features (voxels/verts)
    P = Ya.shape[1]
    G = ((Ya @ Yb.T) / P).astype(np.float64)

    diagG = np.diag(G)
    D = diagG[:, None] + diagG[None, :] - G - G.T

    np.fill_diagonal(D, 0.0)
    return D


# ---------------- SECOND-ORDER ACROSS ROI ----------------

def second_order_across_roi_corrdist(rdms_by_roi_raw: dict[str, np.ndarray], roi_order: list[str]) -> np.ndarray:
    """
    Returns distance = 1 - corr (in [0,2]) between vectorized upper triangles.
    """
    vecs = []
    for roi in roi_order:
        rdm = rdms_by_roi_raw[roi]
        iu = np.triu_indices(rdm.shape[0], k=1)
        vecs.append(rdm[iu].astype(np.float64))
    V = np.vstack(vecs)
    C = np.corrcoef(V)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    return D


# ---------------- MAIN ----------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBJ_DIR.mkdir(parents=True, exist_ok=True)
    GROUP_DIR.mkdir(parents=True, exist_ok=True)

    nsdids_order = load_image_order(IMG_LIST_FILE)

    subj_valid_ids: dict[int, list[str]] = {}
    subj_roi_rdms_raw: dict[int, dict[str, np.ndarray]] = {}
    subj_across_roi_dist: dict[int, np.ndarray] = {}

    for subj in SUBJECTS:
        print(f"\n=== Subject {subj:02d} ===")
        df = load_subject_parquet(subj)

        subj_out = SUBJ_DIR / f"subj{subj:02d}"
        subj_fig = subj_out / "figures"
        subj_out.mkdir(parents=True, exist_ok=True)
        subj_fig.mkdir(parents=True, exist_ok=True)

        # STRICT2 extraction already guarantees these ids are usable
        ids_strict = nsdids_order.copy()

        if len(ids_strict) < 10:
            raise RuntimeError(f"Subject {subj:02d}: too few images in STRICT2 list: {len(ids_strict)}")

        subj_valid_ids[subj] = ids_strict
        (subj_out / "nsdId_order_STRICT2_sessionSplit_repeat01.txt").write_text("\n".join(ids_strict))
        print(f"STRICT2 images kept (balanced 1v1 repeats from 2 sessions): {len(ids_strict)}")

        roi_raw_map: dict[str, np.ndarray] = {}

        for roi in ROIS:
            print(f"  -> ROI {roi}: noise (repeat residuals) + folds (repeat0/repeat1) + crossnobis")
            df_roi = df[df["ROI"] == roi].copy()

            w = diag_noise_from_repeat_residuals(df_roi, ids_strict, shrink_alpha=SHRINK_ALPHA, eps=EPS_VAR)
            X_folds = build_repeat_split_folds_strict2(df_roi, ids_strict)

            rdm_raw = crossnobis_rdm_fast_diagwhiten(X_folds, w=w)
            rdm_vis = shift_to_nonnegative(rdm_raw, eps=EPS_SHIFT)

            roi_raw_map[roi] = rdm_raw

            np.save(subj_out / f"ROI-{roi}_crossnobis_STRICT2_sessionSplit_raw.npy", rdm_raw.astype(np.float64))
            np.save(subj_out / f"ROI-{roi}_crossnobis_STRICT2_sessionSplit_nonneg_VISUAL.npy", rdm_vis.astype(np.float64))

            save_matrix_csv(rdm_raw, ids_strict, subj_out / f"ROI-{roi}_crossnobis_STRICT2_sessionSplit_raw.csv")
            save_matrix_csv(rdm_vis, ids_strict, subj_out / f"ROI-{roi}_crossnobis_STRICT2_sessionSplit_nonneg_VISUAL.csv")

            plot_rdm_clean(
                rdm_vis,
                title=f"Subject {subj:02d} • {roi} • Crossnobis RDM (repeat0/repeat1; STRICT 2×2; visual shift)",
                out_png=subj_fig / f"ROI-{roi}_CrossnobisRDM_STRICT2_sessionSplit.png",
                cmap=RDM_CMAP,
                show_values=False,
                colorbar_label="Crossnobis dissimilarity (shifted for display)"
            )

        # Across-ROI second-order distance (1 - corr) computed from RAW RDMs
        across_dist = second_order_across_roi_corrdist(roi_raw_map, ROIS)
        subj_across_roi_dist[subj] = across_dist
        subj_roi_rdms_raw[subj] = roi_raw_map

        pd.DataFrame(across_dist, index=ROIS, columns=ROIS).to_csv(
            subj_out / "acrossROI_secondorder_STRICT2_sessionSplit_corrdist.csv"
        )

        across_sim = 1.0 - across_dist  # corr in [-1,1]
        pd.DataFrame(across_sim, index=ROIS, columns=ROIS).to_csv(
            subj_out / "acrossROI_secondorder_STRICT2_sessionSplit_corrSIM.csv"
        )

        plot_rdm_clean(
            across_sim,
            title=f"Subject {subj:02d} • RSA Similarity Between Selected ROIs (Crossnobis)",
            out_png=subj_fig / "acrossROI_RSA_similarity_CORR_VALUES_STRICT2.png",
            cmap=ACROSS_ROI_CMAP,
            tick_labels=ROIS,
            show_values=True,
            value_fmt="{:.2f}",
            colorbar_label="Correlation (similarity)",
            vmin=-1.0,
            vmax=1.0
        )

    # ---------------- GROUP MEANS (EXCLUDING subj06 & subj08 => 1–5 only) ----------------
    print(f"\n=== GROUP MEANS (STRICT2; subjects {GROUP_SUBJECTS}) ===")
    group_fig = GROUP_DIR / "figures_excluding_subj06_subj08_STRICT2"
    group_fig.mkdir(parents=True, exist_ok=True)

    # All subjects share the same STRICT2 image list, so group_ids is just that list
    group_ids = nsdids_order.copy()

    (GROUP_DIR / "nsdId_order_GROUP_COMMON_STRICT2_sessionSplit_subjs_1to5.txt").write_text("\n".join(group_ids))
    print(f"Group common STRICT2 images (subjs {GROUP_SUBJECTS}): {len(group_ids)}")

    group_raw_sum = {roi: None for roi in ROIS}
    group_vis_sum = {roi: None for roi in ROIS}

    for subj in GROUP_SUBJECTS:
        ids_subj = subj_valid_ids[subj]

        # indices match exactly because ids_subj == group_ids
        keep_idx = list(range(len(group_ids)))

        for roi in ROIS:
            r = subj_roi_rdms_raw[subj][roi]
            r = r[np.ix_(keep_idx, keep_idx)]
            rv = shift_to_nonnegative(r, eps=EPS_SHIFT)

            if group_raw_sum[roi] is None:
                group_raw_sum[roi] = r.copy()
                group_vis_sum[roi] = rv.copy()
            else:
                group_raw_sum[roi] += r
                group_vis_sum[roi] += rv

    nS = len(GROUP_SUBJECTS)
    for roi in ROIS:
        raw_mean = group_raw_sum[roi] / nS
        vis_mean = group_vis_sum[roi] / nS

        np.save(GROUP_DIR / f"groupmean_subjs_1to5_ROI-{roi}_crossnobis_STRICT2_sessionSplit_raw.npy", raw_mean.astype(np.float64))
        np.save(GROUP_DIR / f"groupmean_subjs_1to5_ROI-{roi}_crossnobis_STRICT2_sessionSplit_nonneg_VISUAL.npy", vis_mean.astype(np.float64))

        save_matrix_csv(raw_mean, group_ids, GROUP_DIR / f"groupmean_subjs_1to5_ROI-{roi}_crossnobis_STRICT2_sessionSplit_raw.csv")
        save_matrix_csv(vis_mean, group_ids, GROUP_DIR / f"groupmean_subjs_1to5_ROI-{roi}_crossnobis_STRICT2_sessionSplit_nonneg_VISUAL.csv")

        plot_rdm_clean(
            vis_mean,
            title=f"Group mean (subjs 1–5) • {roi} • Crossnobis RDM (visual shift)",
            out_png=group_fig / f"groupmean_subjs1to5_ROI-{roi}_CrossnobisRDM_visual_STRICT2.png",
            cmap=RDM_CMAP,
            show_values=False,
            colorbar_label="Crossnobis dissimilarity (shifted for display)"
        )

    # Across-ROI group mean (subjects 1–5)
    across_sum = None
    for subj in GROUP_SUBJECTS:
        A = subj_across_roi_dist[subj]
        across_sum = A.copy() if across_sum is None else across_sum + A
    across_mean_dist = across_sum / nS

    pd.DataFrame(across_mean_dist, index=ROIS, columns=ROIS).to_csv(
        GROUP_DIR / "groupmean_subjs_1to5_acrossROI_secondorder_STRICT2_sessionSplit_corrdist.csv"
    )

    across_mean_sim = 1.0 - across_mean_dist
    pd.DataFrame(across_mean_sim, index=ROIS, columns=ROIS).to_csv(
        GROUP_DIR / "groupmean_subjs_1to5_acrossROI_secondorder_STRICT2_sessionSplit_corrSIM.csv"
    )

    plot_rdm_clean(
        across_mean_sim,
        title="Group mean (subjs 1–5) • RSA Similarity Between Selected ROIs (Crossnobis)",
        out_png=group_fig / "groupmean_subjs1to5_acrossROI_RSA_similarity_CORR_VALUES_STRICT2.png",
        cmap=ACROSS_ROI_CMAP,
        tick_labels=ROIS,
        show_values=True,
        value_fmt="{:.2f}",
        colorbar_label="Correlation (similarity)",
        vmin=-1.0,
        vmax=1.0
    )

    print("\n✅ DONE (STRICT2).")
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()