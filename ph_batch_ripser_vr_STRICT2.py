#!/usr/bin/env python
r"""
ph_batch_ripser_vr_STRICT2.py
===========================================================
Persistent homology (H0/H1/H2) on CROSSNOBIS RDMs using ripser
(Vietoris–Rips filtration on a distance matrix), for STRICT2 outputs.

Reads RAW crossnobis distance matrices:
    ROI-<roi>_crossnobis_STRICT2_sessionSplit_raw.npy

Checks:
- square, finite, symmetric, diagonal ~0

Negatives:
- shifts by a constant so min off-diagonal becomes 0+eps:
    D_shift = D + (-min_offdiag + eps)
  This preserves ordering and persistence (death-birth).

Outputs:
  RDMs_Final/PH_VR_ripser_STRICT2/
    subjXX/
      ROI-<roi>/
        used_distance_matrix_SHIFTED.npy
        dgm_H0.npy, dgm_H1.npy, dgm_H2.npy
        summary.json
        figs/
          diagram_H0.png, diagram_H1.png, diagram_H2.png
          betti_curves.png
    ph_summary.csv  (master table)

Run:
  cd D:\Emily\NSD_cogsci_sandbox\COGS401\COSYNE
  python ph_batch_ripser_vr_STRICT2.py
===========================================================
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
ROOT = Path(r"D:\Emily\NSD_cogsci_sandbox")
RDM_ROOT = ROOT / "COGS401" / "COSYNE" / "RDMs_Final" / "subjects"

# NEW output root so you don't overwrite STRICT3 PH results
OUT_ROOT = ROOT / "COGS401" / "COSYNE" / "RDMs_Final" / "PH_VR_ripser_STRICT2"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Subjects you actually computed STRICT2 RDMs for
SUBJECTS = [1, 2, 3, 4, 5]
MAXDIM = 2

# Shift so min off-diagonal becomes 0+eps (safe for PH)
SHIFT_EPS = 1e-9

# Numerical tolerances
SYM_TOL = 1e-6
DIAG_TOL = 1e-8

# Optional filtration cap:
# - None: use full range (recommended for correctness)
# - 0.95: cap at 95th percentile off-diagonal (speed/consistency)
USE_MAXDIST_QUANTILE = None  # e.g. 0.95

# STRICT2 filename pattern
ROI_RE = re.compile(r"ROI-([A-Za-z0-9]+)_crossnobis_STRICT2_sessionSplit_raw\.npy$")

# ---------------- Imports ----------------
try:
    from ripser import ripser
except Exception as e:
    raise RuntimeError(
        "Missing dependency ripser. Install: python -m pip install ripser\n"
        f"Import error: {e}"
    )


# ---------------- Helpers ----------------

def discover_rdms_for_subject(subj: int) -> list[tuple[str, Path]]:
    subj_dir = RDM_ROOT / f"subj{subj:02d}"
    if not subj_dir.exists():
        raise FileNotFoundError(subj_dir)

    out: list[tuple[str, Path]] = []
    for f in sorted(subj_dir.glob("ROI-*_crossnobis_STRICT2_sessionSplit_raw.npy")):
        m = ROI_RE.search(f.name)
        if m:
            out.append((m.group(1), f))
    if not out:
        raise RuntimeError(f"No STRICT2 RAW RDMs found for subj{subj:02d} in {subj_dir}")
    return out


def basic_checks(D: np.ndarray, fpath: Path) -> None:
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise RuntimeError(f"{fpath}: matrix not square: {D.shape}")

    if not np.isfinite(D).all():
        n_nan = int(np.isnan(D).sum())
        n_inf = int(np.isinf(D).sum())
        raise RuntimeError(f"{fpath}: non-finite entries: NaN={n_nan}, Inf={n_inf}")

    sym_err = float(np.max(np.abs(D - D.T)))
    if sym_err > SYM_TOL:
        raise RuntimeError(f"{fpath}: not symmetric: max|D-DT|={sym_err:.3e}")

    diag_err = float(np.max(np.abs(np.diag(D))))
    if diag_err > DIAG_TOL:
        raise RuntimeError(f"{fpath}: diagonal not ~0: max|diag|={diag_err:.3e}")


def shift_to_nonnegative(D: np.ndarray, eps: float = 0.0) -> tuple[np.ndarray, float]:
    n = D.shape[0]
    off = D[~np.eye(n, dtype=bool)]
    min_off = float(np.min(off))
    if min_off >= 0:
        return D.copy(), 0.0
    shift = (-min_off) + eps
    return D + shift, shift


def choose_maxdist(D_shift: np.ndarray, q: float | None) -> float | None:
    if q is None:
        return None
    n = D_shift.shape[0]
    off = D_shift[~np.eye(n, dtype=bool)]
    return float(np.quantile(off, q))


def summarize_diagram(dgm: np.ndarray) -> dict:
    if dgm.size == 0:
        return {"n": 0, "n_finite": 0, "max_persistence": 0.0, "median_persistence": 0.0}

    b = dgm[:, 0]
    d = dgm[:, 1]
    finite = np.isfinite(d)

    if finite.sum() == 0:
        return {"n": int(dgm.shape[0]), "n_finite": 0, "max_persistence": 0.0, "median_persistence": 0.0}

    pers = (d[finite] - b[finite])
    return {
        "n": int(dgm.shape[0]),
        "n_finite": int(finite.sum()),
        "max_persistence": float(np.max(pers)) if pers.size else 0.0,
        "median_persistence": float(np.median(pers)) if pers.size else 0.0,
    }


def plot_diagram(dgm: np.ndarray, title: str, out_png: Path, maxdist_used: float | None):
    plt.figure(figsize=(6.4, 5.6))
    if dgm.size == 0:
        plt.text(0.5, 0.5, "No features", ha="center", va="center")
        plt.axis("off")
    else:
        b = dgm[:, 0]
        d = dgm[:, 1]
        finite = np.isfinite(d)

        if finite.any():
            plt.scatter(b[finite], d[finite], s=16, alpha=0.75)

            lo = float(min(np.min(b[finite]), np.min(d[finite]), 0.0))
            hi = float(max(np.max(b[finite]), np.max(d[finite])))

            if maxdist_used is not None:
                hi = max(hi, float(maxdist_used))

            plt.plot([lo, hi], [lo, hi], linewidth=1)
            plt.xlabel("Birth (distance)")
            plt.ylabel("Death (distance)")
        else:
            plt.text(0.5, 0.5, "All deaths are infinite (likely H0)", ha="center", va="center")
            plt.axis("off")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def betti_curve(dgm: np.ndarray, ts: np.ndarray) -> np.ndarray:
    if dgm.size == 0:
        return np.zeros_like(ts)

    b = dgm[:, 0]
    d = dgm[:, 1]
    finite = np.isfinite(d)
    b = b[finite]
    d = d[finite]
    if b.size == 0:
        return np.zeros_like(ts)

    out = np.zeros_like(ts, dtype=int)
    for i, t in enumerate(ts):
        out[i] = int(np.sum((b <= t) & (t < d)))
    return out


def plot_betti(dgms: list[np.ndarray], title: str, out_png: Path, maxdist_used: float | None):
    if maxdist_used is None:
        deaths = []
        for dgm in dgms:
            if dgm.size:
                d = dgm[:, 1]
                deaths.append(d[np.isfinite(d)])
        if deaths and np.concatenate(deaths).size:
            hi = float(np.max(np.concatenate(deaths)))
        else:
            hi = 1.0
    else:
        hi = float(maxdist_used)

    ts = np.linspace(0.0, hi, 250)

    plt.figure(figsize=(7.6, 4.8))
    for dim, dgm in enumerate(dgms):
        bc = betti_curve(dgm, ts)
        plt.plot(ts, bc, label=f"Betti_{dim}")

    plt.title(title)
    plt.xlabel("Filtration threshold (distance)")
    plt.ylabel("Betti number")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def run_ripser_distance(D_shift: np.ndarray, maxdim: int, thresh: float | None):
    """
    ripser() type-safe wrapper:
    - only passes thresh when it is a real float
    """
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
            dgms = res["dgms"]  # [H0, H1, H2]

            roi_out = subj_out / f"ROI-{roi}"
            fig_out = roi_out / "figs"
            roi_out.mkdir(parents=True, exist_ok=True)
            fig_out.mkdir(parents=True, exist_ok=True)

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
                "input_file": str(fpath),
                "shift_added_to_make_nonnegative": float(shift_added),
                "maxdist_thresh": None if maxdist is None else float(maxdist),
                "H0": s0,
                "H1": s1,
                "H2": s2,
            }
            (roi_out / "summary.json").write_text(json.dumps(summary, indent=2))

            plot_diagram(dgms[0], f"subj{subj:02d} • {roi} • H0 diagram", fig_out / "diagram_H0.png", maxdist)
            plot_diagram(dgms[1], f"subj{subj:02d} • {roi} • H1 diagram (loops)", fig_out / "diagram_H1.png", maxdist)
            plot_diagram(dgms[2], f"subj{subj:02d} • {roi} • H2 diagram", fig_out / "diagram_H2.png", maxdist)

            plot_betti(dgms, f"subj{subj:02d} • {roi} • Betti curves", fig_out / "betti_curves.png", maxdist)

            rows.append({
                "subject": subj,
                "roi": roi,
                "n_nodes": int(D_raw.shape[0]),
                "strict_version": "STRICT2",
                "shift_added": float(shift_added),
                "maxdist_thresh": None if maxdist is None else float(maxdist),
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