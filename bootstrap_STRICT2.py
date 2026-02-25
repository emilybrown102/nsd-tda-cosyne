#!/usr/bin/env python
r"""
===========================================================
Persistent homology (H0/H1/H2) on CROSSNOBIS RDMs using ripser
(Vietoris–Rips filtration on a distance matrix), for STRICT2 outputs.

Same as original STRICT2 script EXCEPT:
- Does NOT generate persistence diagrams
- Does NOT generate Betti curves
- Does NOT create figs/ directory

All PH computations and summaries remain identical.
===========================================================
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
ROOT = Path(r"D:\Emily\NSD_cogsci_sandbox")
RDM_ROOT = ROOT / "COGS401" / "COSYNE" / "RDMs_Final" / "subjects"

OUT_ROOT = ROOT / "COGS401" / "COSYNE" / "RDMs_Final" / "PH_VR_ripser_STRICT2_NOFIGS"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SUBJECTS = [1, 2, 3, 4, 5]
MAXDIM = 2

SHIFT_EPS = 1e-9
SYM_TOL = 1e-6
DIAG_TOL = 1e-8
USE_MAXDIST_QUANTILE = None

ROI_RE = re.compile(r"ROI-([A-Za-z0-9]+)_crossnobis_STRICT2_sessionSplit_raw\.npy$")

try:
    from ripser import ripser
except Exception as e:
    raise RuntimeError(
        "Missing dependency ripser. Install: python -m pip install ripser\n"
        f"Import error: {e}"
    )


# ---------------- Helpers ----------------

def discover_rdms_for_subject(subj: int):
    subj_dir = RDM_ROOT / f"subj{subj:02d}"
    if not subj_dir.exists():
        raise FileNotFoundError(subj_dir)

    out = []
    for f in sorted(subj_dir.glob("ROI-*_crossnobis_STRICT2_sessionSplit_raw.npy")):
        m = ROI_RE.search(f.name)
        if m:
            out.append((m.group(1), f))

    if not out:
        raise RuntimeError(f"No STRICT2 RAW RDMs found for subj{subj:02d}")
    return out


def basic_checks(D: np.ndarray, fpath: Path):
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise RuntimeError(f"{fpath}: matrix not square")

    if not np.isfinite(D).all():
        raise RuntimeError(f"{fpath}: non-finite entries")

    if np.max(np.abs(D - D.T)) > SYM_TOL:
        raise RuntimeError(f"{fpath}: not symmetric")

    if np.max(np.abs(np.diag(D))) > DIAG_TOL:
        raise RuntimeError(f"{fpath}: diagonal not ~0")


def shift_to_nonnegative(D: np.ndarray, eps: float):
    n = D.shape[0]
    off = D[~np.eye(n, dtype=bool)]
    min_off = float(np.min(off))

    if min_off >= 0:
        return D.copy(), 0.0

    shift = (-min_off) + eps
    return D + shift, shift


def choose_maxdist(D_shift: np.ndarray, q):
    if q is None:
        return None

    n = D_shift.shape[0]
    off = D_shift[~np.eye(n, dtype=bool)]
    return float(np.quantile(off, q))


def summarize_diagram(dgm: np.ndarray):
    if dgm.size == 0:
        return {"n": 0, "n_finite": 0, "max_persistence": 0.0, "median_persistence": 0.0}

    b = dgm[:, 0]
    d = dgm[:, 1]
    finite = np.isfinite(d)

    if finite.sum() == 0:
        return {"n": int(dgm.shape[0]), "n_finite": 0, "max_persistence": 0.0, "median_persistence": 0.0}

    pers = d[finite] - b[finite]

    return {
        "n": int(dgm.shape[0]),
        "n_finite": int(finite.sum()),
        "max_persistence": float(np.max(pers)) if pers.size else 0.0,
        "median_persistence": float(np.median(pers)) if pers.size else 0.0,
    }


def run_ripser_distance(D_shift: np.ndarray, maxdim: int, thresh):
    kwargs = dict(distance_matrix=True, maxdim=maxdim)
    if thresh is not None:
        kwargs["thresh"] = float(thresh)
    return ripser(D_shift, **kwargs)


# ---------------- MAIN ----------------

def main():
    rows = []

    for subj in SUBJECTS:
        print(f"\n=== Subject {subj:02d} ===")
        roi_files = discover_rdms_for_subject(subj)

        subj_out = OUT_ROOT / f"subj{subj:02d}"
        subj_out.mkdir(parents=True, exist_ok=True)

        for roi, fpath in roi_files:
            print(f"  -> ROI {roi}")

            D_raw = np.load(fpath).astype(np.float64, copy=False)
            basic_checks(D_raw, fpath)

            D_shift, shift_added = shift_to_nonnegative(D_raw, eps=SHIFT_EPS)
            np.fill_diagonal(D_shift, 0.0)

            maxdist = choose_maxdist(D_shift, USE_MAXDIST_QUANTILE)
            res = run_ripser_distance(D_shift, maxdim=MAXDIM, thresh=maxdist)

            dgms = res["dgms"]

            roi_out = subj_out / f"ROI-{roi}"
            roi_out.mkdir(parents=True, exist_ok=True)

            np.save(roi_out / "used_distance_matrix_SHIFTED.npy", D_shift)

            for dim in range(MAXDIM + 1):
                np.save(roi_out / f"dgm_H{dim}.npy", dgms[dim])

            s0 = summarize_diagram(dgms[0])
            s1 = summarize_diagram(dgms[1])
            s2 = summarize_diagram(dgms[2])

            summary = {
                "subject": subj,
                "roi": roi,
                "n_nodes": int(D_raw.shape[0]),
                "strict_version": "STRICT2",
                "shift_added_to_make_nonnegative": float(shift_added),
                "maxdist_thresh": None if maxdist is None else float(maxdist),
                "H0": s0,
                "H1": s1,
                "H2": s2,
            }

            (roi_out / "summary.json").write_text(json.dumps(summary, indent=2))

            rows.append({
                "subject": subj,
                "roi": roi,
                "n_nodes": int(D_raw.shape[0]),
                "H1_n": s1["n"],
                "H1_max_persistence": s1["max_persistence"],
                "H1_median_persistence": s1["median_persistence"],
                "H2_n": s2["n"],
                "H2_max_persistence": s2["max_persistence"],
                "H2_median_persistence": s2["median_persistence"],
                "out_dir": str(roi_out),
            })

    out_csv = OUT_ROOT / "ph_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"\n✅ DONE. Master summary: {out_csv}")


if __name__ == "__main__":
    main()
