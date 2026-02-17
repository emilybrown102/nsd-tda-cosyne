#!/usr/bin/env python

"""

------------------------------------------
 
Extract left-hemisphere ROI betas for NSD subjects 01–08 (CROSSNOBIS-READY)
 
Goal (Crossnobis pipeline):
 
1) Start with images common to ALL 8 subjects in the TRIAL file (design-level common).
 
2) Further restrict to images ACTUALLY AVAILABLE in the beta files for ALL 8 subjects,

   AND with EXACTLY 3 valid repeats per subject (as requested).
 
   "Valid repeat" = trial row whose session .mgh exists AND whose trial index is within bounds.
 
3) For each subject + ROI + nsdId + repeat (session/trial), extract ROI beta vector

   WITHOUT averaging repeats.
 
4) Save subject-level parquet files with identical nsdId sets across all 8 subjects,

   preserving per-repeat metadata so you can map back to images/categories later.
 
Outputs (written to a separate crossnobis folder; originals untouched):
 
- subjXX_betas_commonAvailableAll8_3reps_ROIs_LH_crossnobis.parquet

  (one row per subject x ROI x nsdId x repeat; includes metadata + beta vector)
 
- repeat_counts_designCommonAll8_crossnobis3reps.csv

  (diagnostics on repeat counts per subject/nsdId)
 
- images_commonAvailableAll8_crossnobis3reps.txt

  (final nsdId list)
 
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
 
# Save separately from originals

OUT_DIR = ROOT / "COGS401" / "COSYNE" / "betas" / "crossnobis" / "unaveraged_responses_3reps"

OUT_DIR.mkdir(parents=True, exist_ok=True)
 
SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]

ROIS = ["V1", "V2", "V3", "V4", "PIT", "PH", "STSva", "MT", "LO2"]
 
# Require EXACTLY this many repeats per subject per image (as requested)

REPEATS_REQUIRED = 3
 
# Option: store beta as float32 to save space (recommended)

BETA_DTYPE = np.float32
 
# ---------------- HELPERS ----------------
 
def _normalize_subject_col(trial_df: pd.DataFrame) -> pd.DataFrame:

    """Normalize trial_df['subject'] into int 1..8, accepting strings like 'subj01'/'subj001'/'subj0X'."""

    if "subject" not in trial_df.columns:

        raise RuntimeError("TRIAL_FILE must contain a 'subject' column.")

    s = trial_df["subject"].astype(str).str.strip()

    # Common patterns: 'subj01', 'subj001', 'subj0X'

    s = s.str.replace("subj", "", regex=False).str.lstrip("0")

    s = s.replace("", np.nan)

    trial_df["subject"] = s.astype(float).astype("Int64")  # nullable int

    if trial_df["subject"].isna().any():

        bad = trial_df.loc[trial_df["subject"].isna()].head(10)

        raise RuntimeError(f"Could not parse some 'subject' values. Example rows:\n{bad}")

    trial_df["subject"] = trial_df["subject"].astype(int)

    return trial_df
 
def _load_roi_masks(mask_dir: Path, rois: list[str]) -> dict[str, np.ndarray]:

    """Load LH ROI masks from npy files named like lh_<ROI>_*.npy or lh_<ROI>.npy.

    Uses stem split logic similar to your original script, but made more robust.

    """

    masks = {}

    npy_files = list(mask_dir.glob("lh_*.npy"))

    if not npy_files:

        raise RuntimeError(f"No ROI mask files found in: {mask_dir}")
 
    # Try to match ROI names anywhere in filename after "lh_"

    for f in npy_files:

        stem = f.stem  # e.g., "lh_V1" or "lh_V1_something"

        parts = stem.split("_")

        if len(parts) < 2:

            continue

        # Candidate roi token is second part by convention

        candidate = parts[1]

        if candidate in rois and candidate not in masks:

            mask = np.load(f)

            if mask.ndim != 1:

                raise RuntimeError(f"Mask {f} is not 1D. Got shape {mask.shape}")

            if mask.dtype != bool:

                # Accept integer index masks too, but normalize to indices for speed later

                pass

            masks[candidate] = mask
 
    if len(masks) == 0:

        raise RuntimeError(f"No matching ROI masks loaded from {mask_dir} for ROIs={rois}")
 
    # Basic sanity

    empty = [k for k, v in masks.items() if (v.sum() if v.dtype == bool else len(v)) == 0]

    if empty:

        raise RuntimeError(f"Some ROI masks are empty (0 vox/verts): {empty}")
 
    return masks
 
def _get_mgh_ntrials(mgh_file: Path) -> int:

    """Return the number of trials in an .mgh file without forcing full data load."""

    img = nib.load(str(mgh_file))

    shape = img.shape

    # NSD mgh commonly: (vertices, 1, 1, trials) OR (vertices, trials)

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

    # If mask is already indices

    mask = np.asarray(mask)

    if mask.ndim != 1:

        raise RuntimeError(f"Mask indices must be 1D; got shape {mask.shape}")

    return mask.astype(int)

# ---------------- LOAD ROI MASKS ----------------
 
roi_masks = _load_roi_masks(MASKS_DIR, ROIS)

roi_indices = {roi: _mask_to_indices(mask) for roi, mask in roi_masks.items()}
 
print(f"Loaded {len(roi_masks)} ROIs:", list(roi_masks.keys()))
 
# ---------------- LOAD FULL TRIAL INFO ----------------
 
trial_df = pd.read_csv(TRIAL_FILE)

trial_df = _normalize_subject_col(trial_df)
 
trial_df = trial_df[trial_df["subject"].isin(SUBJECTS)].copy()
 
if "nsdId" not in trial_df.columns:

    raise RuntimeError("TRIAL_FILE must contain an 'nsdId' column.")

if "session" not in trial_df.columns:

    raise RuntimeError("TRIAL_FILE must contain a 'session' column.")

if "trial" not in trial_df.columns:

    raise RuntimeError("TRIAL_FILE must contain a 'trial' column.")
 
trial_df["nsdId"] = trial_df["nsdId"].astype(str).str.strip()
 
# Ensure session/trial are ints

trial_df["session"] = trial_df["session"].astype(int)

trial_df["trial"] = trial_df["trial"].astype(int)
 
# Keep all metadata columns (for later category mapping)

META_COLS = [c for c in trial_df.columns if c not in ["subject"]]  # we keep all besides subject; subject re-added per row
 
# ---------------- DESIGN-LEVEL COMMON IMAGES (ALL 8) ----------------
 
images_per_subj = [set(trial_df.loc[trial_df["subject"] == s, "nsdId"]) for s in SUBJECTS]

design_common_images = sorted(list(set.intersection(*images_per_subj)))
 
print(f"Design-level common images across all {len(SUBJECTS)} subjects: {len(design_common_images)}")

if len(design_common_images) == 0:

    raise RuntimeError("No common nsdId found across all 8 subjects in TRIAL_FILE.")
 
# ---------------- PASS 1: FIND VALID REPEATS PER SUBJECT (NO BETA EXTRACTION) ----------------

#

# For each subject, count repeats for each design-common nsdId that are actually extractable:

# - session .mgh exists

# - trial index within mgh bounds

#

# We will keep only images with EXACTLY REPEATS_REQUIRED valid repeats per subject.
 
repeat_summary_rows = []

valid_trials_by_subj_img = defaultdict(lambda: defaultdict(list))  # subj -> nsdId -> list[dict(trial metadata row)]
 
available_images_by_subj = {}  # subj -> set(nsdId meeting repeat criterion)
 
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

        try:

            session_to_ntrials[sess] = _get_mgh_ntrials(mgh_file)

        except Exception as e:

            raise RuntimeError(f"Failed reading shape for {mgh_file}: {e}")
 
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

        # trials in file are 1-indexed in your script; convert later

        if tr < 1 or tr > ntr:

            continue

        valid_rows.append(row)
 
    if len(valid_rows) == 0:

        print(f"⚠️ No valid (in-bounds) trials found for subject {subj:02d}. Availability is empty.")

        available_images_by_subj[subj] = set()

        continue
 
    valid_df = pd.DataFrame(valid_rows)
 
    # Group valid repeats by nsdId

    avail_set = set()

    for img_id, g in valid_df.groupby("nsdId"):

        # store rows (all metadata) for later extraction

        rows_as_dict = g[META_COLS].to_dict(orient="records")

        valid_trials_by_subj_img[subj][img_id] = rows_as_dict
 
    # Count repeats for every design-common image for diagnostics and filtering

    for img_id in design_common_images:

        nrep = len(valid_trials_by_subj_img[subj].get(img_id, []))

        repeat_summary_rows.append({"subject": subj, "nsdId": img_id, "n_valid_repeats_found": nrep})

        if nrep == REPEATS_REQUIRED:

            avail_set.add(img_id)
 
    available_images_by_subj[subj] = avail_set

    print(

        f"Images with exactly {REPEATS_REQUIRED} valid repeats for subj{subj:02d}: "

        f"{len(avail_set)} / {len(design_common_images)}"

    )

# ---------------- FINAL COMMON SET: EXACTLY 3 VALID REPEATS ACROSS ALL 8 ----------------
 
common_available_images = sorted(list(set.intersection(*[available_images_by_subj[s] for s in SUBJECTS])))
 
print(

    f"\nFinal common images across ALL 8 subjects "

    f"with exactly {REPEATS_REQUIRED} valid repeats each: {len(common_available_images)}"

)
 
if len(common_available_images) == 0:

    raise RuntimeError(

        f"No images meet the criterion of exactly {REPEATS_REQUIRED} valid repeats per subject across all 8 subjects.\n"

        "If you expected some, check repeat_counts_designCommonAll8_crossnobis3reps.csv to see where repeats drop."

    )
 
# Save repeat summary + final list (reproducibility)

repeat_df = pd.DataFrame(repeat_summary_rows)

repeat_csv = OUT_DIR / "repeat_counts_designCommonAll8_crossnobis3reps.csv"

repeat_df.to_csv(repeat_csv, index=False)

print(f"Wrote repeat summary: {repeat_csv}")
 
final_list_txt = OUT_DIR / "images_commonAvailableAll8_crossnobis3reps.txt"

final_list_txt.write_text("\n".join(common_available_images))

print(f"Wrote final nsdId list: {final_list_txt}")
 
# ---------------- PASS 2: EXTRACT ROI BETAS PER REPEAT (NO AVERAGING) ----------------
 
for subj in SUBJECTS:

    print(f"\n[PASS 2] Extracting per-repeat ROI betas for subject {subj:02d}...")
 
    lh_dir = BETAS_DIR / f"subj{subj:02d}_betas" / f"subj{subj:02d}_lh"
 
    # Build a session -> list of needed (nsdId, meta_row) lookups to process efficiently

    needed_by_session = defaultdict(list)

    for img_id in common_available_images:

        rows = valid_trials_by_subj_img[subj].get(img_id, [])

        if len(rows) != REPEATS_REQUIRED:

            # This shouldn't happen given the filtering, but keep guardrails

            continue

        for r in rows:

            needed_by_session[int(r["session"])].append((img_id, r))
 
    if len(needed_by_session) == 0:

        raise RuntimeError(f"No sessions to process for subject {subj:02d} in PASS 2. Unexpected state.")
 
    out_rows = []

    # Track repeat_index per image (stable ordering)

    repeat_counter = defaultdict(int)
 
    for sess in sorted(needed_by_session.keys()):

        mgh_file = lh_dir / f"subj{subj:02d}_lh.betas_session{sess:02d}.mgh"

        if not mgh_file.exists():

            raise RuntimeError(f"Missing expected beta file in PASS 2: {mgh_file}")
 
        print(f"  Loading session {sess:02d} beta file...")

        data = _load_mgh_as_2d(mgh_file)  # vertices x trials

        n_vertices, n_trials = data.shape
 
        # Sanity check masks length vs vertices

        for roi, idx in roi_indices.items():

            if idx.max() >= n_vertices:

                raise RuntimeError(

                    f"ROI mask '{roi}' has index >= n_vertices for subject {subj:02d}, session {sess:02d}.\n"

                    f"mask max idx={idx.max()}, n_vertices={n_vertices}."

                )
 
        # Process all required trials in this session

        # We keep betas per ROI per repeat.

        for img_id, meta in needed_by_session[sess]:

            tr_1idx = int(meta["trial"])

            tr_0idx = tr_1idx - 1

            if tr_0idx < 0 or tr_0idx >= n_trials:

                raise RuntimeError(

                    f"Trial index out of bounds in PASS 2 (should have been filtered in PASS 1): "

                    f"subj{subj:02d} sess{sess:02d} trial={tr_1idx}, n_trials={n_trials}"

                )
 
            # Determine repeat_index in a stable way: count repeats as encountered

            rep_i = repeat_counter[img_id]

            repeat_counter[img_id] += 1
 
            # Extract full vertex vector once, then mask for each ROI

            full_vec = data[:, tr_0idx]
 
            for roi_name, idx in roi_indices.items():

                beta_roi = full_vec[idx].astype(BETA_DTYPE, copy=False)
 
                row = {

                    "subject": subj,

                    "ROI": roi_name,

                    "nsdId": str(img_id),

                    "repeat_index": int(rep_i),

                    # Preserve key mapping metadata

                    "session": int(meta["session"]),

                    "trial": int(meta["trial"]),

                    # Store the vector as a Python list for parquet compatibility

                    "beta": beta_roi.tolist(),

                }
 
                # Add all additional metadata columns (categories, etc.) if present

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

            "This usually means some images lost repeats due to ordering/session filtering. "

            "Check the repeat_counts CSV."

        )
 
    # Ensure each image has exactly 3 repeats recorded (per ROI)

    # (This check is conservative; it doesn't stop run but will warn loudly.)

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
 
print("\n✅ DONE. Crossnobis-ready per-repeat ROI betas saved (no averaging), restricted to exactly 3 repeats.")
 