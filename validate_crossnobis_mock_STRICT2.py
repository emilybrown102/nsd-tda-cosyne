#!/usr/bin/env python
r"""
validate_crossnobis_mock_STRICT2.py
===========================================================
END-TO-END VALIDATION FOR STRICT2 CROSSNOBIS PIPELINE

STRICT2 design:
- exactly 2 repeats per image
- repeats come from 2 distinct sessions (here: sessions 1 and 2)
- folds are balanced 1v1: repeat_index 0 vs 1

What this validates
-------------------
This validates the core steps of your STRICT2 pipeline:
  1) STRICT2 filtering: 2 repeats in 2 distinct sessions
  2) Fold building: repeat0 vs repeat1 (balanced; no averaging)
  3) Diagonal noise variance from repeat residuals
  4) Diagonal whitening
  5) Crossnobis distance computation (2-fold; /P scaling)
  6) Across-ROI RSA similarity = correlation of ROI RDM upper triangles

Tests
-----
TEST_0_deterministic_tinyNoise:
  - known structure + tiny noise
  - EXPECT: corr(recovered, weighted_truth) ~ 1

TEST_1_pure_noise:
  - no structure at all
  - EXPECT: NO spurious stable geometry (across-ROI RSA correlations ~0 on average)
  - IMPORTANT: With plug-in whitening (w estimated from same repeats) + shrinkage,
               pure-noise mean distances can be slightly > 0. So we do NOT use
               "mean≈0" as a strict pass/fail criterion.

TEST_2_two_cluster_strongSignal:
  - clear structure + moderate noise
  - EXPECT: recovered RDM correlates strongly with weighted ground truth

Across-ROI expectation (strong-signal)
--------------------------------------
ROIs with known relationships:
  ROI_A: base structure
  ROI_B: identical to A          -> corr(RDM_A, RDM_B) high +
  ROI_C: independent geometry    -> corr(RDM_A, RDM_C) ~ 0
  ROI_D: permuted labels of A    -> corr(RDM_A, RDM_D) ~ 0

Outputs
-------
.../RDMs_MockValidation_STRICT2/
  mock_parquets/
  figures/
  metrics_summary.json
  metrics_summary.txt
===========================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# =========================
# CONFIG
# =========================

ROOT = Path(r"D:\Emily\NSD_cogsci_sandbox")
OUT_DIR = ROOT / "COGS401" / "COSYNE" / "RDMs_MockValidation_STRICT2"
BETAS_OUT = OUT_DIR / "mock_parquets"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BETAS_OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUBJ_MOCK = 99

N_CONDITIONS = 60      # e.g., 60 / 157 / 200
P_FEATURES = 800       # number of voxels/features

# STRICT2: 2 repeats in 2 sessions
SESSIONS = [1, 2]
REPEAT_INDEX_BY_SESSION = {1: 0, 2: 1}

ROIS = ["ROI_A", "ROI_B", "ROI_C", "ROI_D"]

# Match real pipeline settings
SHRINK_ALPHA = 0.10
EPS_VAR = 1e-6
EPS_SHIFT = 1e-6

# Visuals
RDM_CMAP = "viridis"
SIM_CMAP = "bwr"
FIG_DPI = 300
TITLE_FONTSIZE = 16
TICK_FONTSIZE = 10
VALUE_FONTSIZE = 10


# =========================
# Plot helpers
# =========================

def _nice_defaults():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titleweight": "bold",
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
    })

def _wrap_title(title: str, max_chars: int = 46) -> str:
    words = title.split()
    lines, cur, cur_len = [], [], 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur_len + add > max_chars:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)

def plot_heatmap(
    mat: np.ndarray,
    title: str,
    out_png: Path,
    cmap: str,
    tick_labels: List[str] | None = None,
    show_values: bool = False,
    value_fmt: str = "{:.2f}",
    cbar_label: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    _nice_defaults()
    n = mat.shape[0]
    figsize = (10.2, 9.0) if tick_labels is not None else (8.2, 7.2)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(mat, interpolation="nearest", cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_title(_wrap_title(title), fontsize=TITLE_FONTSIZE, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.02)
    if cbar_label:
        cbar.ax.set_ylabel(cbar_label, rotation=90)

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
                ax.text(j, i, value_fmt.format(mat[i, j]),
                        ha="center", va="center",
                        fontsize=VALUE_FONTSIZE, color="black")

    fig.savefig(out_png, dpi=FIG_DPI)
    plt.close(fig)

def plot_scatter(x: np.ndarray, y: np.ndarray, title: str, out_png: Path) -> None:
    _nice_defaults()
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    ax.scatter(x, y, s=10, alpha=0.5)
    ax.set_title(_wrap_title(title), fontsize=TITLE_FONTSIZE, pad=10)
    ax.set_xlabel("Ground-truth distance (upper triangle)")
    ax.set_ylabel("Recovered crossnobis distance (upper triangle)")
    m = min(float(x.min()), float(y.min()))
    M = max(float(x.max()), float(y.max()))
    ax.plot([m, M], [m, M], linewidth=1)
    fig.savefig(out_png, dpi=FIG_DPI)
    plt.close(fig)


# =========================
# Pipeline functions (mirror STRICT2 code)
# =========================

def shift_to_nonnegative(rdm: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = float(np.min(rdm))
    return rdm.copy() if m >= 0 else (rdm + (-m + eps))

def strict_images_2reps_2sessions(df_anyroi: pd.DataFrame, nsdids_order: List[str]) -> List[str]:
    """Keep only nsdIds that have exactly 2 repeats and 2 distinct sessions."""
    keep = []
    g = df_anyroi.groupby("nsdId")
    for img in nsdids_order:
        if img not in g.groups:
            continue
        rows = g.get_group(img)
        if len(rows) != 2:
            continue
        sess = rows["session"].to_numpy()
        if len(np.unique(sess)) != 2:
            continue
        keep.append(img)
    return keep

def build_repeat_split_folds(df_subj_roi: pd.DataFrame, ids: List[str]) -> np.ndarray:
    """STRICT2: Fold A = repeat_index 0, Fold B = repeat_index 1 (balanced 1v1)."""
    first_beta = df_subj_roi.iloc[0]["beta"]
    P = len(first_beta)
    grouped = df_subj_roi.groupby(["nsdId", "repeat_index"])

    X = np.zeros((2, len(ids), P), dtype=np.float32)
    for i, img in enumerate(ids):
        a = np.asarray(grouped.get_group((img, 0)).iloc[0]["beta"], dtype=np.float32)
        b = np.asarray(grouped.get_group((img, 1)).iloc[0]["beta"], dtype=np.float32)
        X[0, i, :] = a
        X[1, i, :] = b
    return X

def diag_noise_from_repeat_residuals(
    df_subj_roi: pd.DataFrame,
    ids: List[str],
    shrink_alpha: float = 0.10,
    eps: float = 1e-6
) -> np.ndarray:
    """Estimate diagonal noise variance per voxel from repeat residuals (STRICT2)."""
    grouped = df_subj_roi.groupby("nsdId")
    residuals = []
    for img in ids:
        rows = grouped.get_group(img)
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

def crossnobis_rdm_fast_diagwhiten(X_folds: np.ndarray, w: np.ndarray) -> np.ndarray:
    X = X_folds.astype(np.float32, copy=False)
    K, N, _ = X.shape
    if K < 2:
        raise RuntimeError("Need at least 2 folds for crossnobis.")
    Y = X * w.reshape(1, 1, -1)

    Ya, Yb = Y[0], Y[1]
    P = Ya.shape[1]
    G = ((Ya @ Yb.T) / P).astype(np.float64)

    diagG = np.diag(G)
    D = diagG[:, None] + diagG[None, :] - G - G.T
    np.fill_diagonal(D, 0.0)
    return D

def across_roi_corr_similarity(rdms_by_roi_raw: Dict[str, np.ndarray], roi_order: List[str]) -> np.ndarray:
    vecs = []
    for roi in roi_order:
        rdm = rdms_by_roi_raw[roi]
        iu = np.triu_indices(rdm.shape[0], k=1)
        vecs.append(rdm[iu].astype(np.float64))
    V = np.vstack(vecs)
    C = np.corrcoef(V)
    np.fill_diagonal(C, 1.0)
    return C


# =========================
# Synthetic generation
# =========================

@dataclass
class MockSpec:
    name: str
    signal_scale: float
    noise_sd: float
    structure: str        # "noise" or "two_cluster"
    separation: float = 2.0


def make_nsd_ids(n: int) -> List[str]:
    return [f"mock{idx:04d}" for idx in range(1, n + 1)]


def make_two_cluster_mu(n: int, p: int, separation: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=int)
    labels[n // 2:] = 1
    rng.shuffle(labels)

    base = rng.normal(0, 1.0, size=(n, p)).astype(np.float32)

    d = rng.normal(0, 1.0, size=(p,)).astype(np.float32)
    d = d / (np.std(d) + 1e-12)

    shift = ((labels * 2 - 1).astype(np.float32)[:, None]) * (separation / 2.0) * d[None, :]
    mu = base + shift
    return mu


def true_distance_rdm_weighted(mu: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Ground truth matched to the pipeline: expected (whitened) squared distance / P."""
    muw = (mu.astype(np.float64) * w.astype(np.float64)[None, :])
    n, p = muw.shape
    ss = np.sum(muw * muw, axis=1)
    G = (muw @ muw.T) / p
    D = (ss[:, None] + ss[None, :]) / p - 2.0 * G
    np.fill_diagonal(D, 0.0)
    return D


def write_mock_parquet(
    out_file: Path,
    subject: int,
    nsd_ids: List[str],
    spec: MockSpec,
    seed: int
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, p = len(nsd_ids), P_FEATURES

    if spec.structure == "two_cluster":
        mu_A = make_two_cluster_mu(n, p, separation=spec.separation, seed=seed + 10)
    else:
        mu_A = np.zeros((n, p), dtype=np.float32)

    mu_B = mu_A.copy()
    mu_C = rng.normal(0, 1.0, size=(n, p)).astype(np.float32) if spec.structure == "two_cluster" else np.zeros((n, p), dtype=np.float32)
    perm = rng.permutation(n)
    mu_D = mu_A[perm].copy()

    mu_by_roi = {"ROI_A": mu_A, "ROI_B": mu_B, "ROI_C": mu_C, "ROI_D": mu_D}

    rows = []
    for i, nsd in enumerate(nsd_ids):
        for sess in SESSIONS:
            rep_i = REPEAT_INDEX_BY_SESSION[sess]
            for roi in ROIS:
                mu = mu_by_roi[roi][i]
                noise = rng.normal(0, spec.noise_sd, size=p).astype(np.float32)
                beta = (spec.signal_scale * mu + noise).astype(np.float32)
                rows.append({
                    "subject": subject,
                    "ROI": roi,
                    "nsdId": nsd,
                    "repeat_index": int(rep_i),
                    "session": int(sess),
                    "beta": beta.tolist(),
                })

    pd.DataFrame(rows).to_parquet(out_file, index=False)
    return mu_by_roi


# =========================
# Metrics
# =========================

def upper_tri_vec(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu]

def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float((a @ b) / denom)

def summarize_rdm(rdm: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(rdm)),
        "max": float(np.max(rdm)),
        "mean": float(np.mean(rdm)),
        "fraction_negative": float(np.mean(rdm < 0)),
    }

def mean_sd(vals: List[float]) -> Tuple[float, float]:
    x = np.asarray(vals, dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


# =========================
# Run one spec
# =========================

def run_one(spec: MockSpec, seed: int, make_figures: bool) -> Dict:
    nsd_ids = make_nsd_ids(N_CONDITIONS)
    parquet_file = BETAS_OUT / f"subj{SUBJ_MOCK:02d}_{spec.name}_seed{seed}_mock_STRICT2.parquet"
    mu_by_roi = write_mock_parquet(parquet_file, SUBJ_MOCK, nsd_ids, spec, seed)

    df = pd.read_parquet(parquet_file)
    df["nsdId"] = df["nsdId"].astype(str)
    df["ROI"] = df["ROI"].astype(str)
    df["session"] = df["session"].astype(int)
    df["repeat_index"] = df["repeat_index"].astype(int)

    ids_strict = strict_images_2reps_2sessions(df[df["ROI"] == "ROI_A"].copy(), nsd_ids)
    if len(ids_strict) != len(nsd_ids):
        raise RuntimeError(f"STRICT2 filter kept {len(ids_strict)} / {len(nsd_ids)} (unexpected in this mock).")

    rdms_raw: Dict[str, np.ndarray] = {}
    truths_w: Dict[str, np.ndarray] = {}

    for roi in ROIS:
        df_roi = df[df["ROI"] == roi].copy()
        w = diag_noise_from_repeat_residuals(df_roi, ids_strict, shrink_alpha=SHRINK_ALPHA, eps=EPS_VAR)
        X = build_repeat_split_folds(df_roi, ids_strict)
        rdm = crossnobis_rdm_fast_diagwhiten(X, w)
        rdms_raw[roi] = rdm
        truths_w[roi] = true_distance_rdm_weighted(mu_by_roi[roi], w)

    across_sim = across_roi_corr_similarity(rdms_raw, ROIS)

    # ROI_A proof vs weighted truth
    rdmA = rdms_raw["ROI_A"]
    tA = truths_w["ROI_A"]
    corr_A = corr(upper_tri_vec(tA), upper_tri_vec(rdmA))
    max_abs_diff = float(np.max(np.abs(rdmA - tA)))

    idx = {r: i for i, r in enumerate(ROIS)}
    def sim(a: str, b: str) -> float:
        return float(across_sim[idx[a], idx[b]])

    if make_figures:
        plot_heatmap(
            shift_to_nonnegative(rdmA, eps=EPS_SHIFT),
            title=f"{spec.name} (seed {seed}): ROI_A recovered crossnobis RDM [visual shift; STRICT2]",
            out_png=FIG_DIR / f"{spec.name}_seed{seed}_ROI_A_recovered_STRICT2.png",
            cmap=RDM_CMAP,
            cbar_label="Crossnobis (shifted for display)"
        )
        plot_heatmap(
            tA,
            title=f"{spec.name} (seed {seed}): ROI_A ground truth (weighted to match whitening; STRICT2)",
            out_png=FIG_DIR / f"{spec.name}_seed{seed}_ROI_A_truth_weighted_STRICT2.png",
            cmap=RDM_CMAP,
            cbar_label="True distance (weighted; squared / P)"
        )
        plot_scatter(
            upper_tri_vec(tA),
            upper_tri_vec(rdmA),
            title=f"{spec.name} (seed {seed}): ROI_A recovered vs ground truth (STRICT2)",
            out_png=FIG_DIR / f"{spec.name}_seed{seed}_ROI_A_scatter_STRICT2.png"
        )
        plot_heatmap(
            across_sim,
            title=f"{spec.name} (seed {seed}): Across-ROI RSA similarity (corr of RDMs; STRICT2)",
            out_png=FIG_DIR / f"{spec.name}_seed{seed}_acrossROI_corr_VALUES_STRICT2.png",
            cmap=SIM_CMAP,
            tick_labels=ROIS,
            show_values=True,
            value_fmt="{:.2f}",
            cbar_label="Correlation (similarity)",
            vmin=-1.0,
            vmax=1.0,
        )

    return {
        "spec": spec.__dict__,
        "seed": seed,
        "strict_images_kept": len(ids_strict),
        "roi_A_rdm_summary": summarize_rdm(rdmA),
        "roi_A_corr_recovered_vs_truth_WEIGHTED": float(corr_A),
        "roi_A_maxabsdiff_recovered_minus_truth_WEIGHTED": float(max_abs_diff),
        "observed_key_similarities": {
            "corr(ROI_A, ROI_B)": sim("ROI_A", "ROI_B"),
            "corr(ROI_A, ROI_C)": sim("ROI_A", "ROI_C"),
            "corr(ROI_A, ROI_D)": sim("ROI_A", "ROI_D"),
        },
    }


def main():
    seed = 123

    # Single-seed headline tests (with figures)
    tests = [
        MockSpec(name="TEST_0_deterministic_tinyNoise", signal_scale=1.0, noise_sd=1e-3, structure="two_cluster", separation=2.0),
        MockSpec(name="TEST_1_pure_noise", signal_scale=0.0, noise_sd=1.0, structure="noise", separation=2.0),
        MockSpec(name="TEST_2_two_cluster_strongSignal", signal_scale=1.0, noise_sd=0.5, structure="two_cluster", separation=2.0),
    ]

    results_single = {}
    for spec in tests:
        print(f"\nRunning {spec.name} (seed {seed}) ...")
        results_single[spec.name] = run_one(spec, seed=seed, make_figures=True)

    # Monte Carlo robustness (moderate SNR)
    spec_mc = MockSpec(name="MC_moderateSNR", signal_scale=1.0, noise_sd=1.0, structure="two_cluster", separation=2.0)
    seeds = list(range(200, 220))  # 20 seeds
    mc = []
    print(f"\nRunning Monte Carlo robustness (moderate SNR): {len(seeds)} seeds ...")
    for s in seeds:
        mc.append(run_one(spec_mc, seed=s, make_figures=False))

    corr_vals = [r["roi_A_corr_recovered_vs_truth_WEIGHTED"] for r in mc]
    ab_vals = [r["observed_key_similarities"]["corr(ROI_A, ROI_B)"] for r in mc]
    ac_vals = [r["observed_key_similarities"]["corr(ROI_A, ROI_C)"] for r in mc]
    ad_vals = [r["observed_key_similarities"]["corr(ROI_A, ROI_D)"] for r in mc]

    mc_summary = {
        "n_seeds": len(seeds),
        "corr_A_vs_truth_WEIGHTED_mean_sd": mean_sd(corr_vals),
        "corr(A,B)_mean_sd": mean_sd(ab_vals),
        "corr(A,C)_mean_sd": mean_sd(ac_vals),
        "corr(A,D)_mean_sd": mean_sd(ad_vals),
    }

    # Monte Carlo pure noise (principled no-structure check)
    spec_noise_mc = MockSpec(name="MC_pure_noise", signal_scale=0.0, noise_sd=1.0, structure="noise", separation=2.0)
    noise_seeds = list(range(300, 350))  # 50 seeds
    noise_mc = []
    print(f"\nRunning PURE NOISE Monte Carlo: {len(noise_seeds)} seeds ...")
    for s in noise_seeds:
        noise_mc.append(run_one(spec_noise_mc, seed=s, make_figures=False))

    noise_means = [r["roi_A_rdm_summary"]["mean"] for r in noise_mc]
    noise_fracneg = [r["roi_A_rdm_summary"]["fraction_negative"] for r in noise_mc]

    noise_mc_summary = {
        "n_seeds": len(noise_seeds),
        "mean_of_means": float(np.mean(noise_means)),
        "sd_of_means": float(np.std(noise_means, ddof=1)),
        "mean_fraction_negative": float(np.mean(noise_fracneg)),
        "sd_fraction_negative": float(np.std(noise_fracneg, ddof=1)),
    }

    # Additional noise metrics: across-ROI correlations should be ~0
    noise_ab = [r["observed_key_similarities"]["corr(ROI_A, ROI_B)"] for r in noise_mc]
    noise_ac = [r["observed_key_similarities"]["corr(ROI_A, ROI_C)"] for r in noise_mc]
    noise_ad = [r["observed_key_similarities"]["corr(ROI_A, ROI_D)"] for r in noise_mc]

    noise_ab_mean_sd = mean_sd(noise_ab)
    noise_ac_mean_sd = mean_sd(noise_ac)
    noise_ad_mean_sd = mean_sd(noise_ad)

    # Informational centering diagnostic (NOT used for pass/fail)
    center_z = abs(noise_mc_summary["mean_of_means"]) / (noise_mc_summary["sd_of_means"] + 1e-12)

    # PASS/FAIL
    pf = {
        "TEST_0_near_perfect_corr": {
            "pass": results_single["TEST_0_deterministic_tinyNoise"]["roi_A_corr_recovered_vs_truth_WEIGHTED"] > 0.99,
            "corr": results_single["TEST_0_deterministic_tinyNoise"]["roi_A_corr_recovered_vs_truth_WEIGHTED"],
        },
        "TEST_1_pure_noise_no_spurious_structure_MonteCarlo": {
            "pass": (
                abs(noise_ab_mean_sd[0]) < 0.20
                and abs(noise_ac_mean_sd[0]) < 0.20
                and abs(noise_ad_mean_sd[0]) < 0.20
                and (0.20 <= noise_mc_summary["mean_fraction_negative"] <= 0.80)
            ),
            "single_seed_mean_descriptive": results_single["TEST_1_pure_noise"]["roi_A_rdm_summary"]["mean"],
            "single_seed_fracneg_descriptive": results_single["TEST_1_pure_noise"]["roi_A_rdm_summary"]["fraction_negative"],
            "mc_summary": noise_mc_summary,
            "mc_acrossROI_mean_sd": {
                "corr(A,B)_mean_sd": noise_ab_mean_sd,
                "corr(A,C)_mean_sd": noise_ac_mean_sd,
                "corr(A,D)_mean_sd": noise_ad_mean_sd,
            },
            "criteria": {
                "abs(mean corr(A,B)) < 0.20": abs(noise_ab_mean_sd[0]) < 0.20,
                "abs(mean corr(A,C)) < 0.20": abs(noise_ac_mean_sd[0]) < 0.20,
                "abs(mean corr(A,D)) < 0.20": abs(noise_ad_mean_sd[0]) < 0.20,
                "0.20 <= mean_fraction_negative <= 0.80": (0.20 <= noise_mc_summary["mean_fraction_negative"] <= 0.80),
                "centering_z = |mean_of_means|/sd_of_means (informational)": float(center_z),
            },
        },
        "TEST_2_strong_signal_corr_positive": {
            "pass": results_single["TEST_2_two_cluster_strongSignal"]["roi_A_corr_recovered_vs_truth_WEIGHTED"] > 0.6,
            "corr": results_single["TEST_2_two_cluster_strongSignal"]["roi_A_corr_recovered_vs_truth_WEIGHTED"],
        },
        "AcrossROI_expected_pattern_strong_signal": {
            "pass": (
                results_single["TEST_2_two_cluster_strongSignal"]["observed_key_similarities"]["corr(ROI_A, ROI_B)"] > 0.6
                and abs(results_single["TEST_2_two_cluster_strongSignal"]["observed_key_similarities"]["corr(ROI_A, ROI_C)"]) < 0.2
                and abs(results_single["TEST_2_two_cluster_strongSignal"]["observed_key_similarities"]["corr(ROI_A, ROI_D)"]) < 0.2
            ),
            "A_B": results_single["TEST_2_two_cluster_strongSignal"]["observed_key_similarities"]["corr(ROI_A, ROI_B)"],
            "A_C": results_single["TEST_2_two_cluster_strongSignal"]["observed_key_similarities"]["corr(ROI_A, ROI_C)"],
            "A_D": results_single["TEST_2_two_cluster_strongSignal"]["observed_key_similarities"]["corr(ROI_A, ROI_D)"],
        },
        "MonteCarlo_moderateSNR_corr_positive_on_average": {
            "pass": mc_summary["corr_A_vs_truth_WEIGHTED_mean_sd"][0] > 0.2,
            "mean_sd": mc_summary["corr_A_vs_truth_WEIGHTED_mean_sd"],
        },
    }

    payload = {
        "single_seed_results": results_single,
        "monte_carlo_spec_moderateSNR": spec_mc.__dict__,
        "monte_carlo_summary_moderateSNR": mc_summary,
        "monte_carlo_spec_pure_noise": spec_noise_mc.__dict__,
        "monte_carlo_summary_pure_noise": noise_mc_summary,
        "pass_fail": pf,
        "interpretation_simple": [
            "When we inject known representational structure, recovered STRICT2 crossnobis RDM matches the expected (weighted) ground truth.",
            "For pure noise (no structure), we test that the pipeline does not produce stable geometry (across-ROI RSA correlations ~0 on average).",
            "Across-ROI RSA behaves meaningfully under signal: ROI_B matches ROI_A; ROI_C and ROI_D are ~0 relative to ROI_A."
        ],
    }

    out_json = OUT_DIR / "metrics_summary.json"
    out_txt = OUT_DIR / "metrics_summary.txt"
    out_json.write_text(json.dumps(payload, indent=2))

    # Human-readable summary text
    lines = []
    for name, res in results_single.items():
        s = res["roi_A_rdm_summary"]
        lines.append(f"=== {name} ===")
        lines.append(f"Spec: {res['spec']}")
        lines.append(f"ROI_A stats: mean={s['mean']:.4f}, frac_neg={s['fraction_negative']:.3f}, min={s['min']:.3f}, max={s['max']:.3f}")
        lines.append(f"Corr(recovered ROI_A, truth ROI_A) [WEIGHTED]: {res['roi_A_corr_recovered_vs_truth_WEIGHTED']:.3f}")
        sim = res["observed_key_similarities"]
        lines.append("Across-ROI RSA similarity (corr of RDMs):")
        lines.append(f"  corr(A,B) expected + : {sim['corr(ROI_A, ROI_B)']:.3f}")
        lines.append(f"  corr(A,C) expected ~0: {sim['corr(ROI_A, ROI_C)']:.3f}")
        lines.append(f"  corr(A,D) expected ~0 (permuted): {sim['corr(ROI_A, ROI_D)']:.3f}")
        lines.append("")

    lines.append("=== MONTE CARLO (moderate SNR) ===")
    lines.append(f"n_seeds: {mc_summary['n_seeds']}")
    lines.append(f"Corr(A vs truth) mean±sd: {mc_summary['corr_A_vs_truth_WEIGHTED_mean_sd'][0]:.3f} ± {mc_summary['corr_A_vs_truth_WEIGHTED_mean_sd'][1]:.3f}")
    lines.append(f"corr(A,B) mean±sd: {mc_summary['corr(A,B)_mean_sd'][0]:.3f} ± {mc_summary['corr(A,B)_mean_sd'][1]:.3f}")
    lines.append(f"corr(A,C) mean±sd: {mc_summary['corr(A,C)_mean_sd'][0]:.3f} ± {mc_summary['corr(A,C)_mean_sd'][1]:.3f}")
    lines.append(f"corr(A,D) mean±sd: {mc_summary['corr(A,D)_mean_sd'][0]:.3f} ± {mc_summary['corr(A,D)_mean_sd'][1]:.3f}")
    lines.append("")

    lines.append("=== MONTE CARLO (pure noise no-structure check) ===")
    lines.append(f"n_seeds: {noise_mc_summary['n_seeds']}")
    lines.append(f"mean_of_means (descriptive): {noise_mc_summary['mean_of_means']:.4f} (sd={noise_mc_summary['sd_of_means']:.4f})")
    lines.append(f"mean_fraction_negative: {noise_mc_summary['mean_fraction_negative']:.3f} (sd={noise_mc_summary['sd_fraction_negative']:.3f})")
    lines.append(f"corr(A,B) mean±sd: {noise_ab_mean_sd[0]:.3f} ± {noise_ab_mean_sd[1]:.3f}")
    lines.append(f"corr(A,C) mean±sd: {noise_ac_mean_sd[0]:.3f} ± {noise_ac_mean_sd[1]:.3f}")
    lines.append(f"corr(A,D) mean±sd: {noise_ad_mean_sd[0]:.3f} ± {noise_ad_mean_sd[1]:.3f}")
    lines.append(f"centering_z (informational): {center_z:.2f}")
    lines.append("")

    lines.append("=== PASS/FAIL SUMMARY ===")
    for k, v in pf.items():
        details = {kk: vv for kk, vv in v.items() if kk != "pass"}
        lines.append(f"{k}: {'PASS' if v['pass'] else 'FAIL'}  ({details})")

    out_txt.write_text("\n".join(lines))

    print("\n✅ Done (STRICT2 mock validation).")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_txt}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()