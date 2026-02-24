#!/usr/bin/env python
r"""
graph_sanity_sweep_batch_STRICT2.py
===========================================================
Batch "PH-readiness" / graph sanity sweep for your REAL STRICT2 crossnobis RDMs.

Auto-discovers matrices:
  D:\Emily\NSD_cogsci_sandbox\COGS401\COSYNE\RDMs_Final\subjects\subjXX\
    ROI-*_crossnobis_STRICT2_sessionSplit_raw.npy

For EACH subject×ROI matrix (RAW only):
  1) Basic validity checks:
     - finite, symmetric (approx), diagonal ~ 0
  2) kNN graph sweep (undirected kNN):
     - edges, density, avg_degree, #components, largest_component
  3) epsilon sweep using DISTANCE quantiles on off-diagonal entries:
     - eps = quantile(dist_offdiag, q)
     - same metrics as above

Outputs:
  ...\RDMs_Final\graph_sanity_batch_STRICT2\
    - summary_knn.csv
    - summary_eps.csv
    - summary_matrix_checks.csv
    - WORST_CASE_* plots

Notes
-----
- Uses ONLY *_raw.npy (never VISUAL).
- No external deps (no networkx). Uses union-find for components.
===========================================================
"""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================

ROOT = Path(r"D:\Emily\NSD_cogsci_sandbox")
RDM_ROOT = ROOT / "COGS401" / "COSYNE" / "RDMs_Final"
SUBJ_ROOT = RDM_ROOT / "subjects"

OUT_DIR = RDM_ROOT / "graph_sanity_batch_STRICT2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# kNN values to test
K_LIST = [2, 5, 10, 20, 40]

# epsilon sweep as quantiles of off-diagonal distances
EPS_QUANTILES = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

# How many "worst cases" to plot
N_WORST_TO_PLOT = 6

# Tolerances for checks
SYM_TOL = 1e-5
DIAG_TOL = 1e-10

# Exclusions (match your group analysis practice)
EXCLUDE_SUBJECTS = {6, 8}


# =========================
# Union-Find (components)
# =========================

class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=int)
        self.size = np.ones(n, dtype=int)

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def component_sizes(self) -> np.ndarray:
        roots = np.array([self.find(i) for i in range(len(self.parent))], dtype=int)
        _, counts = np.unique(roots, return_counts=True)
        return np.sort(counts)[::-1]


# =========================
# Graph builders + metrics
# =========================

def metrics_from_edges(n: int, edges: list[tuple[int, int]]) -> dict:
    uf = UnionFind(n)
    for i, j in edges:
        uf.union(i, j)

    comps = uf.component_sizes()
    n_components = int(len(comps))
    largest = int(comps[0]) if len(comps) else 0

    E = len(edges)
    maxE = n * (n - 1) / 2
    density = float(E / maxE) if maxE > 0 else 0.0
    avg_degree = float(2 * E / n) if n > 0 else 0.0

    return {
        "edges": int(E),
        "density": density,
        "avg_degree": avg_degree,
        "n_components": n_components,
        "largest_component": largest,
    }


def knn_undirected_edges(D: np.ndarray, k: int) -> list[tuple[int, int]]:
    n = D.shape[0]
    Dwork = D.copy()
    np.fill_diagonal(Dwork, np.inf)

    edges = set()
    for i in range(n):
        nn = np.argpartition(Dwork[i], kth=k-1)[:k]
        for j in nn:
            j = int(j)
            a, b = (i, j) if i < j else (j, i)
            if a != b:
                edges.add((a, b))
    return sorted(edges)


def eps_threshold_edges(D: np.ndarray, eps: float) -> list[tuple[int, int]]:
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    mask = D[iu] <= eps
    ii = iu[0][mask]
    jj = iu[1][mask]
    return list(zip(ii.tolist(), jj.tolist()))


# =========================
# IO discovery helpers
# =========================

ROI_RE = re.compile(r"ROI-([A-Za-z0-9]+)_crossnobis_STRICT2_sessionSplit_raw\.npy$")

def discover_rdms() -> list[tuple[int, str, Path]]:
    found = []
    if not SUBJ_ROOT.exists():
        raise FileNotFoundError(f"Missing subjects folder: {SUBJ_ROOT}")

    for subj_dir in sorted(SUBJ_ROOT.glob("subj[0-9][0-9]")):
        msub = re.match(r"subj(\d\d)$", subj_dir.name)
        if not msub:
            continue
        subj = int(msub.group(1))
        if subj in EXCLUDE_SUBJECTS:
            continue

        for f in sorted(subj_dir.glob("ROI-*_crossnobis_STRICT2_sessionSplit_raw.npy")):
            m = ROI_RE.search(f.name)
            if not m:
                continue
            roi = m.group(1)
            found.append((subj, roi, f))

    if not found:
        raise RuntimeError(f"No matching STRICT2 RAW RDMs found under: {SUBJ_ROOT}")
    return found


# =========================
# Plotting worst-cases
# =========================

def plot_knn_curve(rows: pd.DataFrame, out_png: Path, title: str) -> None:
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(rows["k"], rows["n_components"], marker="o")
    plt.xlabel("k (kNN)")
    plt.ylabel("# connected components")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()

def plot_eps_curve(rows: pd.DataFrame, out_png: Path, title: str) -> None:
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(rows["eps_quantile"], rows["n_components"], marker="o")
    plt.xlabel("epsilon quantile (of off-diagonal distances)")
    plt.ylabel("# connected components")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


# =========================
# Main sweep
# =========================

def main() -> None:
    items = discover_rdms()
    print(f"Discovered {len(items)} STRICT2 RAW RDMs.")

    checks_rows = []
    knn_rows = []
    eps_rows = []

    for subj, roi, fpath in items:
        D = np.load(fpath)

        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise RuntimeError(f"Matrix is not square: {fpath} shape={D.shape}")

        n = D.shape[0]
        finite_ok = bool(np.isfinite(D).all())
        sym_max_abs = float(np.max(np.abs(D - D.T)))
        sym_ok = bool(sym_max_abs <= SYM_TOL)
        diag_max_abs = float(np.max(np.abs(np.diag(D))))
        diag_ok = bool(diag_max_abs <= DIAG_TOL)

        iu = np.triu_indices(n, k=1)
        off = D[iu]
        off_finite = off[np.isfinite(off)]
        if len(off_finite) == 0:
            raise RuntimeError(f"No finite off-diagonal entries: {fpath}")

        checks_rows.append({
            "subject": subj,
            "roi": roi,
            "file": str(fpath),
            "n": n,
            "finite_ok": finite_ok,
            "sym_ok": sym_ok,
            "sym_max_abs": sym_max_abs,
            "diag_ok": diag_ok,
            "diag_max_abs": diag_max_abs,
            "min": float(np.min(D)),
            "max": float(np.max(D)),
            "mean": float(np.mean(D)),
            "frac_negative": float(np.mean(D < 0)),
        })

        if not finite_ok:
            print(f"⚠️ Non-finite entries in subj{subj:02d} {roi} — skipping graph sweeps.")
            continue

        # kNN sweep
        for k in K_LIST:
            edges = knn_undirected_edges(D, k=k)
            m = metrics_from_edges(n, edges)
            knn_rows.append({"subject": subj, "roi": roi, "k": k, **m})

        # epsilon sweep
        for q in EPS_QUANTILES:
            eps = float(np.quantile(off_finite, q))
            edges = eps_threshold_edges(D, eps=eps)
            m = metrics_from_edges(n, edges)
            eps_rows.append({
                "subject": subj,
                "roi": roi,
                "eps_quantile": q,
                "eps_value": eps,
                **m
            })

        print(f"Done subj{subj:02d} {roi}: n={n}")

    checks_df = pd.DataFrame(checks_rows)
    knn_df = pd.DataFrame(knn_rows)
    eps_df = pd.DataFrame(eps_rows)

    checks_csv = OUT_DIR / "summary_matrix_checks.csv"
    knn_csv = OUT_DIR / "summary_knn.csv"
    eps_csv = OUT_DIR / "summary_eps.csv"

    checks_df.to_csv(checks_csv, index=False)
    knn_df.to_csv(knn_csv, index=False)
    eps_df.to_csv(eps_csv, index=False)

    print("\nSaved:")
    print(f"  {checks_csv}")
    print(f"  {knn_csv}")
    print(f"  {eps_csv}")

    # Worst cases
    if not knn_df.empty:
        worst_knn = (
            knn_df[knn_df["k"] == min(K_LIST)]
            .sort_values(["n_components", "largest_component"], ascending=[False, True])
            .head(N_WORST_TO_PLOT)
        )
    else:
        worst_knn = pd.DataFrame()

    if not eps_df.empty:
        worst_eps = (
            eps_df[eps_df["eps_quantile"] == min(EPS_QUANTILES)]
            .sort_values(["n_components", "largest_component"], ascending=[False, True])
            .head(N_WORST_TO_PLOT)
        )
    else:
        worst_eps = pd.DataFrame()

    fig_out = OUT_DIR / "worst_case_plots"
    fig_out.mkdir(exist_ok=True)

    for _, row in worst_knn.iterrows():
        subj, roi = int(row["subject"]), str(row["roi"])
        rows = knn_df[(knn_df["subject"] == subj) & (knn_df["roi"] == roi)].sort_values("k")
        out_png = fig_out / f"WORST_kNN_subj{subj:02d}_{roi}_STRICT2.png"
        plot_knn_curve(rows, out_png, title=f"subj{subj:02d} {roi} — kNN components vs k (STRICT2)")
    for _, row in worst_eps.iterrows():
        subj, roi = int(row["subject"]), str(row["roi"])
        rows = eps_df[(eps_df["subject"] == subj) & (eps_df["roi"] == roi)].sort_values("eps_quantile")
        out_png = fig_out / f"WORST_eps_subj{subj:02d}_{roi}_STRICT2.png"
        plot_eps_curve(rows, out_png, title=f"subj{subj:02d} {roi} — eps components vs quantile (STRICT2)")

    print(f"\nWorst-case plots saved to: {fig_out}")

    bad = checks_df[(~checks_df["finite_ok"]) | (~checks_df["sym_ok"]) | (~checks_df["diag_ok"])]
    if len(bad) > 0:
        print("\n⚠️ Matrices failing basic checks:")
        print(bad[["subject", "roi", "finite_ok", "sym_ok", "sym_max_abs", "diag_ok", "diag_max_abs", "file"]].to_string(index=False))
    else:
        print("\n✅ All matrices passed basic finite/symmetry/diagonal checks.")

    if not knn_df.empty:
        tmp = knn_df[knn_df["k"] == min(K_LIST)]
        print("\n--- kNN (k=2) connectivity summary (median by ROI) ---")
        print(tmp.groupby("roi")[["n_components", "largest_component", "avg_degree"]].median().round(3))
    if not eps_df.empty:
        tmp = eps_df[eps_df["eps_quantile"] == min(EPS_QUANTILES)]
        print("\n--- Epsilon (lowest quantile) connectivity summary (median by ROI) ---")
        print(tmp.groupby("roi")[["n_components", "largest_component", "avg_degree"]].median().round(3))


if __name__ == "__main__":
    main()