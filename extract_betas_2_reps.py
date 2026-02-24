#!/usr/bin/env python
"""
------------------------------------------

Extract left-hemisphere ROI betas for NSD subjects 01–08 (CROSSNOBIS-READY) — STRICT2

Goal (Crossnobis pipeline, STRICT2 parallel):
1) Start with images common to ALL 8 subjects in the TRIAL file (design-level common).
2) Further restrict to images ACTUALLY AVAILABLE in the beta files for ALL 8 subjects,
   AND with EXACTLY 2 selected repeats per subject per image,
   where the 2 repeats come from 2 DISTINCT SESSIONS (prefer odd/even if possible).
3) For each subject + ROI + nsdId + repeat (session/trial), extract ROI beta vector WITHOUT averaging.
4) Save subject-level parquet files with identical nsdId sets across all 8 subjects.

Outputs:
- subjXX_betas_commonAvailableAll8_2reps_ROIs_LH_crossnobis.parquet
  (one row per subject x ROI x nsdId x repeat; includes metadata + beta vector)

- repeat_counts_designCommonAll8_crossnobis2reps.csv
  (diagnostics: how many valid repeats existed BEFORE picking 2, per subject/nsdId)

- images_commonAvailableAll8_crossnobis2reps.txt
  (final nsdId list used)

------------------------------------------
"""

import numpy as np
import pandas as pd
import nibabel as nib

from pathlib import Path
from collections import defaultdict

# ---------------- CONFIG ----------------

ROOT = Path(r"D:\Emily\NSD_cogsci_sandbox")

BETAS_DIR = ROOT / "all_betas_fithrf"
MASKS_DIR = ROOT / "atlases" / "roi_masks"
TRIAL_FILE = ROOT / "full_trial_info.csv"

# STRICT2 outputs (parallel to your STRICT3 folder)
OUT_DIR = ROOT / "COGS401" / "COSYNE" / "betas" / "crossnobis" / "unaveraged_responses_2reps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]
ROIS = ["V1", "V2", "V3", "V4", "PIT", "PH", "STSva", "MT", "LO2"]

# STRICT2: we will WRITE exactly 2 repeats per subject per image
REPEATS_REQUIRED = 2

# store beta as float32 to save space
BETA_DTYPE = np.float32

# ---------------- HELPERS ----------------

def _normalize_subject_col(trial_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize trial_df['subject'] into int 1..8, accepting strings like 'subj01'/'subj001'/'subj0X'."""
    if "subject" not in trial_df.columns:
        raise RuntimeError("TRIAL_FILE must contain a 'subject' column.")

    s = trial_df["subject"].astype(str).str.strip()
    s = s.str.replace("subj", "", regex=False).str.lstrip("0")
    s = s.replace("", np.nan)
    trial_df["subject"] = s.astype(float).astype("Int64")  # nullable int

    if trial_df["subject"].isna().any():
        bad = trial_df.loc[trial_df["subject"].isna()].head(10)
        raise RuntimeError(f"Could not parse some 'subject' values. Example rows:\n{bad}")

    trial_df["subject"] = trial_df["subject"].astype(int)
    return trial_df


def _load_roi_masks(mask_dir: Path, rois: list[str]) -> dict[str, np.ndarray]:
    """
    Load LH ROI masks from npy files named like lh_<ROI>_*.npy or lh_<ROI>.npy.
    Matches ROI token as the second underscore-separated field (lh_<ROI>_...).
    """
    masks = {}
    npy_files = list(mask_dir.glob("lh_*.npy"))
    if not npy_files:
        raise RuntimeError(f"No ROI mask files found in: {mask_dir}")

    for f in npy_files:
        stem = f.stem
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        candidate = parts[1]
        if candidate in rois and candidate not in masks:
            mask = np.load(f)
            if mask.ndim != 1:
                raise RuntimeError(f"Mask {f} is not 1D. Got shape {mask.shape}")
            masks[candidate] = mask

    if len(masks) == 0:
        raise RuntimeError(f"No matching ROI masks loaded from {mask_dir} for ROIs={rois}")

    empty = [k for k, v in masks.items() if (v.sum() if v.dtype == bool else len(v)) == 0]
    if empty:
        raise RuntimeError(f"Some ROI masks are empty (0 vox/verts): {empty}")

    return masks


def _get_mgh_ntrials(mgh_file: Path) -> int:
    """Return the number of trials in an .mgh file without forcing full data load."""
    img = nib.load(str(mgh_file))
    shape = img.shape
    if len(shape) == 4:
        return int(shape[-1])
    if len(shape) == 2:
        return int(shape[-1])
    raise RuntimeError(f"Unexpected .mgh shape {shape} for file: {mgh_file}")


def _load_mgh_as_2d(mgh_file: Path) -> np.ndarray:
    """Load .mgh data as a 2D array: vertices x trials."""
    img = nib.load(str(mgh_file))
    data = np.asanyarray(img.dataobj)
    data = np.squeeze(data)
    if data.ndim != 2:
        raise RuntimeError(f"Loaded data is not 2D after squeeze. Got shape {data.shape} for {mgh_file}")
    return data


def _mask_to_indices(mask: np.ndarray) -> np.ndarray:
    """Convert boolean mask or index array to a 1D int index array."""
    if mask.dtype == bool:
        return np.where(mask)[0].astype(int)
    mask = np.asarray(mask)
    if mask.ndim != 1:
        raise RuntimeError(f"Mask indices must be 1D; got shape {mask.shape}")
    return mask.astype(int)


def pick_two_repeats_two_sessions(rows_df: pd.DataFrame, meta_cols: list[str]) -> list[dict]:
    """
    Pick exactly 2 repeats from 2 distinct sessions (deterministic).
    Preference:
      1) one odd-session + one even-session (earliest by session, then trial)
      2) else earliest two distinct sessions
    Within a session, pick smallest trial.

    Returns: list of 2 dict rows (metadata fields only), or [] if not possible.
    """
    rows_df = rows_df.sort_values(["session", "trial"], ascending=[True, True]).copy()
    sessions = rows_df["session"].to_numpy()
    uniq_sessions = np.unique(sessions)
    if len(uniq_sessions) < 2:
        return []

    odd = rows_df[rows_df["session"] % 2 == 1]
    even = rows_df[rows_df["session"] % 2 == 0]

    if len(odd) > 0 and len(even) > 0:
        r1 = odd.iloc[0]
        r2 = even.iloc[0]
    else:
        s1, s2 = uniq_sessions[0], uniq_sessions[1]
        r1 = rows_df[rows_df["session"] == s1].iloc[0]
        r2 = rows_df[rows_df["session"] == s2].iloc[0]

    return pd.DataFrame([r1, r2])[meta_cols].to_dict(orient="records")


# ---------------- LOAD ROI MASKS ----------------

roi_masks = _load_roi_masks(MASKS_DIR, ROIS)
roi_indices = {roi: _mask_to_indices(mask) for roi, mask in roi_masks.items()}
print(f"Loaded {len(roi_masks)} ROIs:", list(roi_masks.keys()))

# ---------------- LOAD FULL TRIAL INFO ----------------

trial_df = pd.read_csv(TRIAL_FILE)
trial_df = _normalize_subject_col(trial_df)
trial_df = trial_df[trial_df["subject"].isin(SUBJECTS)].copy()

for col in ["nsdId", "session", "trial"]:
    if col not in trial_df.columns:
        raise RuntimeError(f"TRIAL_FILE must contain '{col}' column.")

trial_df["nsdId"] = trial_df["nsdId"].astype(str).str.strip()
trial_df["session"] = trial_df["session"].astype(int)
trial_df["trial"] = trial_df["trial"].astype(int)

# Keep all metadata columns (for later category mapping)
META_COLS = [c for c in trial_df.columns if c not in ["subject"]]

# ---------------- DESIGN-LEVEL COMMON IMAGES (ALL 8) ----------------

images_per_subj = [set(trial_df.loc[trial_df["subject"] == s, "nsdId"]) for s in SUBJECTS]
design_common_images = sorted(list(set.intersection(*images_per_subj)))

print(f"Design-level common images across all {len(SUBJECTS)} subjects: {len(design_common_images)}")
if len(design_common_images) == 0:
    raise RuntimeError("No common nsdId found across all 8 subjects in TRIAL_FILE.")

# ---------------- PASS 1: FIND VALID REPEATS PER SUBJECT (NO BETA EXTRACTION) ----------------
#
# For each subject, identify which design-common nsdIds are extractable:
# - session .mgh exists
# - trial index within mgh bounds
# Then choose exactly 2 repeats from 2 distinct sessions per image (deterministic).
#

repeat_summary_rows = []
valid_trials_by_subj_img = defaultdict(lambda: defaultdict(list))  # subj -> nsdId -> list[dict(meta rows)] (picked 2 only)
available_images_by_subj = {}  # subj -> set(nsdId meeting STRICT2 criterion)

for subj in SUBJECTS:
    print(f"\n[PASS 1] Scanning valid repeats for subject {subj:02d}...")

    subj_trials = trial_df[
        (trial_df["subject"] == subj) &
        (trial_df["nsdId"].isin(design_common_images))
    ].copy()

    lh_dir = BETAS_DIR / f"subj{subj:02d}_betas" / f"subj{subj:02d}_lh"

    # Determine which sessions have existing mgh and how many trials each contains
    session_to_ntrials = {}
    for sess in sorted(subj_trials["session"].unique()):
        mgh_file = lh_dir / f"subj{subj:02d}_lh.betas_session{sess:02d}.mgh"
        if not mgh_file.exists():
            continue
        session_to_ntrials[sess] = _get_mgh_ntrials(mgh_file)

    if len(session_to_ntrials) == 0:
        print(f"⚠️ No session .mgh files found for subject {subj:02d}. Availability is empty.")
        available_images_by_subj[subj] = set()
        continue

    # Keep only trial rows whose session exists AND trial index is in bounds
    valid_rows = []
    for _, row in subj_trials.iterrows():
        sess = int(row["session"])
        tr = int(row["trial"])
        if sess not in session_to_ntrials:
            continue
        ntr = session_to_ntrials[sess]
        if tr < 1 or tr > ntr:
            continue
        valid_rows.append(row)

    if len(valid_rows) == 0:
        print(f"⚠️ No valid (in-bounds) trials found for subject {subj:02d}. Availability is empty.")
        available_images_by_subj[subj] = set()
        continue

    valid_df = pd.DataFrame(valid_rows)

    # For each nsdId: record how many valid repeats exist (diagnostic), then pick 2-from-2-sessions
    avail_set = set()

    # Precompute counts of valid repeats (before picking)
    counts_raw = valid_df.groupby("nsdId").size().to_dict()

    for img_id in design_common_images:
        n_valid = int(counts_raw.get(img_id, 0))
        repeat_summary_rows.append({"subject": subj, "nsdId": img_id, "n_valid_repeats_found": n_valid})

        if img_id not in counts_raw:
            continue

        picked = pick_two_repeats_two_sessions(valid_df[valid_df["nsdId"] == img_id], META_COLS)
        if len(picked) == REPEATS_REQUIRED:
            valid_trials_by_subj_img[subj][img_id] = picked
            avail_set.add(img_id)

    available_images_by_subj[subj] = avail_set
    print(f"Images with STRICT2 (2 repeats from 2 sessions) for subj{subj:02d}: {len(avail_set)} / {len(design_common_images)}")

# ---------------- FINAL COMMON SET: STRICT2 ACROSS ALL 8 ----------------

common_available_images = sorted(list(set.intersection(*[available_images_by_subj[s] for s in SUBJECTS])))

print(f"\nFinal common images across ALL 8 subjects with STRICT2 (2 repeats, 2 sessions): {len(common_available_images)}")
if len(common_available_images) == 0:
    raise RuntimeError(
        "No images meet the STRICT2 criterion across all 8 subjects.\n"
        "Check repeat_counts_designCommonAll8_crossnobis2reps.csv for where repeats drop."
    )

# Save repeat summary + final list
repeat_df = pd.DataFrame(repeat_summary_rows)
repeat_csv = OUT_DIR / "repeat_counts_designCommonAll8_crossnobis2reps.csv"
repeat_df.to_csv(repeat_csv, index=False)
print(f"Wrote repeat summary: {repeat_csv}")

final_list_txt = OUT_DIR / "images_commonAvailableAll8_crossnobis2reps.txt"
final_list_txt.write_text("\n".join(common_available_images))
print(f"Wrote final nsdId list: {final_list_txt}")

# ---------------- PASS 2: EXTRACT ROI BETAS PER REPEAT (NO AVERAGING) ----------------

for subj in SUBJECTS:
    print(f"\n[PASS 2] Extracting per-repeat ROI betas for subject {subj:02d}...")

    lh_dir = BETAS_DIR / f"subj{subj:02d}_betas" / f"subj{subj:02d}_lh"

    # session -> list of (nsdId, meta_row)
    needed_by_session = defaultdict(list)

    for img_id in common_available_images:
        rows = valid_trials_by_subj_img[subj].get(img_id, [])
        if len(rows) != REPEATS_REQUIRED:
            continue
        for r in rows:
            needed_by_session[int(r["session"])].append((img_id, r))

    if len(needed_by_session) == 0:
        raise RuntimeError(f"No sessions to process for subject {subj:02d} in PASS 2. Unexpected state.")

    out_rows = []
    repeat_counter = defaultdict(int)

    for sess in sorted(needed_by_session.keys()):
        mgh_file = lh_dir / f"subj{subj:02d}_lh.betas_session{sess:02d}.mgh"
        if not mgh_file.exists():
            raise RuntimeError(f"Missing expected beta file in PASS 2: {mgh_file}")

        print(f"  Loading session {sess:02d} beta file...")
        data = _load_mgh_as_2d(mgh_file)  # vertices x trials
        n_vertices, n_trials = data.shape

        # Sanity: mask indices fit
        for roi, idx in roi_indices.items():
            if idx.max() >= n_vertices:
                raise RuntimeError(
                    f"ROI mask '{roi}' has index >= n_vertices for subject {subj:02d}, session {sess:02d}.\n"
                    f"mask max idx={idx.max()}, n_vertices={n_vertices}."
                )

        for img_id, meta in needed_by_session[sess]:
            tr_1idx = int(meta["trial"])
            tr_0idx = tr_1idx - 1

            if tr_0idx < 0 or tr_0idx >= n_trials:
                raise RuntimeError(
                    f"Trial index out of bounds in PASS 2: subj{subj:02d} sess{sess:02d} "
                    f"trial={tr_1idx}, n_trials={n_trials}"
                )

            rep_i = repeat_counter[img_id]
            repeat_counter[img_id] += 1

            full_vec = data[:, tr_0idx]

            for roi_name, idx in roi_indices.items():
                beta_roi = full_vec[idx].astype(BETA_DTYPE, copy=False)

                row = {
                    "subject": subj,
                    "ROI": roi_name,
                    "nsdId": str(img_id),
                    "repeat_index": int(rep_i),
                    "session": int(meta["session"]),
                    "trial": int(meta["trial"]),
                    "beta": beta_roi.tolist(),
                }

                # add other metadata columns
                for c in META_COLS:
                    if c in ["nsdId", "session", "trial"]:
                        continue
                    row[c] = meta.get(c, None)

                out_rows.append(row)

    out_df = pd.DataFrame(out_rows)

    # Sanity checks
    expected_rows = len(common_available_images) * REPEATS_REQUIRED * len(roi_indices)
    if len(out_df) != expected_rows:
        print(
            f"⚠️ Row count mismatch for subject {subj:02d}: got {len(out_df)}, expected {expected_rows}.\n"
            "This usually means some images lost repeats due to selection/order. Check the repeat_counts CSV."
        )

    check = (
        out_df.groupby(["nsdId", "ROI"])["repeat_index"].nunique()
        .reset_index(name="n_repeats_written")
    )
    bad = check[check["n_repeats_written"] != REPEATS_REQUIRED]
    if not bad.empty:
        print(f"⚠️ Some (nsdId, ROI) pairs do not have exactly {REPEATS_REQUIRED} repeats written:")
        print(bad.head(20).to_string(index=False))

    outfile = OUT_DIR / f"subj{subj:02d}_betas_commonAvailableAll8_{REPEATS_REQUIRED}reps_ROIs_LH_crossnobis.parquet"
    out_df.to_parquet(outfile, index=False)
    print(f"Wrote {outfile} with {len(out_df)} rows")

print("\n✅ DONE. STRICT2 Crossnobis-ready per-repeat ROI betas saved (no averaging), restricted to 2 repeats from 2 sessions.")