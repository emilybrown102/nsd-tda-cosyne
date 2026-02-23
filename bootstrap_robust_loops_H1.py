#!/usr/bin/env python
r"""
bootstrap_h1_subsample_consensus_cycles.py
===========================================================
COSYNE-defensible pipeline for ROI-specific persistent H1 loops
with *visualizable* VR cycles (closed loops) and image mapping.

CRITICAL FIX vs "bootstrap with replacement"
--------------------------------------------
We DO NOT resample indices with replacement (which creates duplicate points and
0-distance artifacts in a distance matrix VR complex). Instead we use:

  - repeated SUBSAMPLING WITHOUT REPLACEMENT

This measures stability under missing stimuli (standard “subsampling stability”),
and avoids the degenerate topology that duplicates create.

High-level steps (per subject × ROI)
------------------------------------
1) Load full distance matrix D (N×N), enforce symmetry, set diagonal to 0
2) Shift to nonnegative (if needed) for ripser
3) Repeat b=1..B:
     - pick m = round(SUBSAMPLE_FRAC * N) distinct stimuli (replace=False)
     - compute H1 diagram on the m×m submatrix
     - keep only TOP_L_PER_SUBSAMPLE most persistent H1 features
       after applying a persistence floor (noise control)
     - store (birth, death, persistence) with bootstrap id
4) Cluster all kept points in (birth, death) space:
     - clusters are “features”
     - cluster hit-rate = number of bootstraps contributing ≥1 point / B
     - robust if hit-rate ≥ ROBUST_CLUSTER_HITRATE_MIN
5) For each robust cluster:
     - compute full ripser on the full N×N D_shift with cocycles
     - find nearest H1 feature to (birth_med, death_med) and get strongest cocycle edge (u,v)
     - build VR graph at eps = birth + CYCLE_EPS_FRAC_PERSIST * persistence (clamped below death)
     - remove edge (u,v), find shortest path u->v, then add (u,v) back -> a SIMPLE GRAPH CYCLE
     - require cycle length >= MIN_CYCLE_LEN
     - output ordered cycle nodes with nsdId for image mapping

Outputs
-------
RDMs_Final/PH_VR_ripser/bootstrap200_subsample_consensus/
  H1_points_long.csv              (all kept points from subsamples)
  robust_clusters_H1.csv          (robust consensus clusters)
  robust_cycles_nodes.csv         (ordered cycle nodes + nsdId for images)
  bootstrap_config.json           (full provenance)

===========================================================
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from ripser import ripser


# =========================
# CONFIG (EDIT HERE)
# =========================

ROOT = Path(r"D:\Emily\NSD_cogsci_sandbox")
RDM_ROOT = ROOT / "COGS401" / "COSYNE" / "RDMs_Final"
PH_ROOT = RDM_ROOT / "PH_VR_ripser"

OUT_DIR = PH_ROOT / "bootstrap200_subsample_consensus"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = [1, 2, 3, 4, 5]
ROIS = ["V1","V2","V3","V4","PIT","PH","STSva","MT","LO2"]

# Resampling settings
B = 200
RNG_SEED = 20260219

# Subsampling without replacement (key fix)
SUBSAMPLE_FRAC = 0.80  # e.g., 0.75–0.85 are common; 0.80 is a good default

# Keep distances nonnegative
EPS_SHIFT = 1e-6

# Ripser settings
MAXDIM = 1

# Strictness knobs (defensible defaults)
SUBSAMPLE_FRAC = 0.80          # keep this
TOP_L_PER_SUBSAMPLE = 10       # was 5 (too strict)
PERSIST_MIN_FRAC_OF_MAXDIST = 0.05   # was 0.05 (too strict)
CLUSTER_EPS_FRAC_OF_MAXDIST = 0.012  # was 0.008 (too strict)
ROBUST_CLUSTER_HITRATE_MIN = 0.80    # was 0.90 (too strict)

# Cycle extraction (visual loop)
CYCLE_EPS_FRAC_PERSIST = 0.05              # eps = birth + frac * persistence (slightly above birth)
MIN_CYCLE_LEN = 4                          # enforce visually meaningful cycle (>=4 vertices)

# Safety checks
CHECK_ORDER_IDENTITY_ACROSS_SUBJECTS = True  # ensures nsdId order is identical across subjects


# =========================
# Helpers
# =========================

def load_order_file(subj: int) -> List[str]:
    f = RDM_ROOT / "subjects" / f"subj{subj:02d}" / f"subj{subj:02d}_nsdId_order_STRICT3_sessionSplit_oddEven.txt"
    if not f.exists():
        raise FileNotFoundError(f"Missing nsdId order file for subj{subj:02d}: {f}")
    ids = [line.strip() for line in f.read_text().splitlines() if line.strip()]
    if len(ids) == 0:
        raise RuntimeError(f"Order file empty: {f}")
    return ids

def load_rdm_raw(subj: int, roi: str) -> np.ndarray:
    f = RDM_ROOT / "subjects" / f"subj{subj:02d}" / f"ROI-{roi}_crossnobis_STRICT3_sessionSplit_raw.npy"
    if not f.exists():
        raise FileNotFoundError(f"Missing RDM raw file: {f}")
    D = np.load(f)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise RuntimeError(f"RDM not square: {f} shape={D.shape}")
    return D.astype(np.float64)

def sanitize_distance_matrix(D: np.ndarray) -> np.ndarray:
    """
    Enforce properties we rely on:
      - finite off-diagonal entries
      - symmetry (average with transpose)
      - diagonal = 0
    """
    D = np.array(D, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(D)):
        # allow diagonal infinities? no; be strict
        bad = np.sum(~np.isfinite(D))
        raise RuntimeError(f"Distance matrix contains {bad} non-finite values (NaN/Inf).")

    # Symmetrize (in case of tiny numerical asymmetry)
    D = 0.5 * (D + D.T)

    # Force diagonal exactly 0 for VR behavior
    np.fill_diagonal(D, 0.0)

    return D

def shift_nonneg(D: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, float]:
    D = np.array(D, dtype=np.float64, copy=True)
    n = D.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off = D[mask]
    off = off[np.isfinite(off)]
    if off.size == 0:
        raise RuntimeError("No finite off-diagonal entries.")
    m = float(off.min())
    if m >= 0:
        return D, 0.0
    shift = (-m + eps)
    return D + shift, shift

def finite_max_offdiag(D: np.ndarray) -> float:
    n = D.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off = D[mask]
    off = off[np.isfinite(off)]
    if off.size == 0:
        raise RuntimeError("No finite off-diagonal entries for maxdist.")
    return float(off.max())

def ripser_call(D_shift: np.ndarray, thresh: float, do_cocycles: bool) -> Dict:
    return ripser(
        D_shift,
        distance_matrix=True,
        maxdim=MAXDIM,
        thresh=float(thresh),
        do_cocycles=do_cocycles
    )

def filter_h1_points(dgm_h1: np.ndarray, persist_min: float) -> np.ndarray:
    """
    Return array (k,3): birth, death, persistence for finite death points above persist_min.
    """
    if dgm_h1.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    birth = dgm_h1[:, 0]
    death = dgm_h1[:, 1]
    finite = np.isfinite(death)
    birth = birth[finite]
    death = death[finite]
    if birth.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    pers = death - birth
    keep = pers >= persist_min
    out = np.column_stack([birth[keep], death[keep], pers[keep]]).astype(np.float64)
    return out

@dataclass
class Cluster:
    center_birth: float
    center_death: float
    points: List[Tuple[int, float, float, float]]  # (bootstrap_idx, birth, death, pers)
    bootstraps_seen: Set[int]

def cluster_points(points: List[Tuple[int, float, float, float]], eps_match: float) -> List[Cluster]:
    """
    Greedy clustering in (birth,death).
    Robustness is later computed by unique bootstraps contributing points to each cluster.
    """
    clusters: List[Cluster] = []
    eps2 = eps_match * eps_match

    for (b, birth, death, pers) in points:
        assigned = False
        for c in clusters:
            db = birth - c.center_birth
            dd = death - c.center_death
            if (db*db + dd*dd) <= eps2:
                c.points.append((b, birth, death, pers))
                c.bootstraps_seen.add(b)
                k = len(c.points)
                c.center_birth += (birth - c.center_birth) / k
                c.center_death += (death - c.center_death) / k
                assigned = True
                break

        if not assigned:
            clusters.append(Cluster(
                center_birth=float(birth),
                center_death=float(death),
                points=[(b, float(birth), float(death), float(pers))],
                bootstraps_seen={int(b)}
            ))
    return clusters

def choose_representative(cluster: Cluster) -> Tuple[float, float, float]:
    births = np.array([p[1] for p in cluster.points], dtype=np.float64)
    deaths = np.array([p[2] for p in cluster.points], dtype=np.float64)
    pers  = np.array([p[3] for p in cluster.points], dtype=np.float64)
    return float(np.median(births)), float(np.median(deaths)), float(np.median(pers))

def cocycle_strong_edge(res_full: Dict, target_birth: float, target_death: float) -> Optional[Tuple[int,int]]:
    """
    Find the H1 feature in the full diagram nearest to (target_birth,target_death),
    then return the (u,v) edge with maximal |coef| from its cocycle.
    """
    dgms = res_full.get("dgms", [])
    if len(dgms) < 2 or dgms[1].size == 0:
        return None

    full_dgm = dgms[1]
    dist2 = (full_dgm[:, 0] - target_birth)**2 + (full_dgm[:, 1] - target_death)**2
    j = int(np.argmin(dist2))

    cocycles = res_full.get("cocycles", None)
    if cocycles is None or len(cocycles) < 2:
        return None
    if j >= len(cocycles[1]):
        return None

    coc = cocycles[1][j]
    if coc is None or len(coc) == 0:
        return None

    coc = np.asarray(coc)
    coefs = np.abs(coc[:, 2])
    k = int(np.argmax(coefs))
    u = int(coc[k, 0])
    v = int(coc[k, 1])
    if u == v:
        return None
    return (u, v) if u < v else (v, u)

def build_vr_adj(D_shift: np.ndarray, eps: float) -> List[List[int]]:
    n = D_shift.shape[0]
    adj = [[] for _ in range(n)]
    for i in range(n):
        di = D_shift[i]
        for j in range(i+1, n):
            if di[j] <= eps:
                adj[i].append(j)
                adj[j].append(i)
    return adj

def bfs_shortest_path(adj: List[List[int]], src: int, dst: int, banned_edge: Optional[Tuple[int,int]] = None) -> Optional[List[int]]:
    from collections import deque
    n = len(adj)
    parent = [-1] * n
    parent[src] = src
    q = deque([src])

    bu, bv = (-1, -1) if banned_edge is None else banned_edge

    while q:
        u = q.popleft()
        if u == dst:
            break
        for v in adj[u]:
            if banned_edge is not None and ((u == bu and v == bv) or (u == bv and v == bu)):
                continue
            if parent[v] != -1:
                continue
            parent[v] = u
            q.append(v)

    if parent[dst] == -1:
        return None

    path = [dst]
    while path[-1] != src:
        path.append(parent[path[-1]])
    path.reverse()
    return path

def extract_simple_cycle_full_graph(
    D_shift_full: np.ndarray,
    res_full: Dict,
    birth: float,
    death: float,
    pers: float
) -> Optional[Tuple[List[int], float]]:
    """
    Extract a simple cycle on the FULL N-node VR graph at eps slightly above birth.
    Returns (cycle_nodes_closed, eps_used) where cycle_nodes_closed ends with start node repeated.
    """
    if pers <= 0 or not np.isfinite(birth) or not np.isfinite(death):
        return None

    eps = birth + CYCLE_EPS_FRAC_PERSIST * pers
    eps = min(eps, death - 1e-9)  # stay within feature lifetime
    if not np.isfinite(eps) or eps <= 0:
        return None

    edge = cocycle_strong_edge(res_full, target_birth=birth, target_death=death)
    if edge is None:
        return None

    u, v = edge
    n = D_shift_full.shape[0]
    if not (0 <= u < n and 0 <= v < n):
        return None

    adj = build_vr_adj(D_shift_full, eps=eps)

    # Find shortest u->v path with (u,v) removed; add it back to close a cycle
    path = bfs_shortest_path(adj, src=u, dst=v, banned_edge=(u, v))
    if path is None:
        return None

    cycle = path + [u]  # closes via the (v,u) edge
    unique_vertices = list(dict.fromkeys(cycle))
    if len(unique_vertices) < MIN_CYCLE_LEN:
        return None

    return cycle, float(eps)


# =========================
# Main
# =========================

def main():
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)

    # Load nsdId orders + check identity if requested
    subj_order: Dict[int, List[str]] = {s: load_order_file(s) for s in SUBJECTS}
    N = len(subj_order[SUBJECTS[0]])
    for s in SUBJECTS[1:]:
        if len(subj_order[s]) != N:
            raise RuntimeError(f"Order length mismatch: subj{s:02d} has {len(subj_order[s])} vs {N}")

    if CHECK_ORDER_IDENTITY_ACROSS_SUBJECTS:
        base_ids = subj_order[SUBJECTS[0]]
        for s in SUBJECTS[1:]:
            if subj_order[s] != base_ids:
                raise RuntimeError(
                    f"nsdId order identity mismatch between subj{SUBJECTS[0]:02d} and subj{s:02d}. "
                    "This would break cross-subject image mapping; fix your order files."
                )

    # Determine subsample size
    m = int(round(SUBSAMPLE_FRAC * N))
    if not (3 <= m <= N):
        raise RuntimeError(f"Invalid subsample size m={m} from SUBSAMPLE_FRAC={SUBSAMPLE_FRAC} and N={N}")

    # Save config provenance
    config = dict(
        B=B,
        RNG_SEED=RNG_SEED,
        SUBJECTS=SUBJECTS,
        ROIS=ROIS,
        SUBSAMPLE_FRAC=SUBSAMPLE_FRAC,
        SUBSAMPLE_SIZE=m,
        EPS_SHIFT=EPS_SHIFT,
        MAXDIM=MAXDIM,
        TOP_L_PER_SUBSAMPLE=TOP_L_PER_SUBSAMPLE,
        PERSIST_MIN_FRAC_OF_MAXDIST=PERSIST_MIN_FRAC_OF_MAXDIST,
        CLUSTER_EPS_FRAC_OF_MAXDIST=CLUSTER_EPS_FRAC_OF_MAXDIST,
        ROBUST_CLUSTER_HITRATE_MIN=ROBUST_CLUSTER_HITRATE_MIN,
        CYCLE_EPS_FRAC_PERSIST=CYCLE_EPS_FRAC_PERSIST,
        MIN_CYCLE_LEN=MIN_CYCLE_LEN,
        CHECK_ORDER_IDENTITY_ACROSS_SUBJECTS=CHECK_ORDER_IDENTITY_ACROSS_SUBJECTS,
    )
    (OUT_DIR / "bootstrap_config.json").write_text(json.dumps(config, indent=2))

    # Load + sanitize + shift RDMs
    print("Loading, sanitizing, and shifting RDMs...")
    D_full_map: Dict[Tuple[int,str], np.ndarray] = {}
    D_shift_map: Dict[Tuple[int,str], np.ndarray] = {}
    maxdist_map: Dict[Tuple[int,str], float] = {}
    shift_added_map: Dict[Tuple[int,str], float] = {}

    for s in SUBJECTS:
        for roi in ROIS:
            D_raw = load_rdm_raw(s, roi)
            D_raw = sanitize_distance_matrix(D_raw)
            D_shift, shift_added = shift_nonneg(D_raw, eps=EPS_SHIFT)
            maxdist = finite_max_offdiag(D_shift)

            D_full_map[(s,roi)] = D_raw
            D_shift_map[(s,roi)] = D_shift
            maxdist_map[(s,roi)] = maxdist
            shift_added_map[(s,roi)] = shift_added

    long_rows: List[Dict] = []
    robust_rows: List[Dict] = []
    cycle_rows: List[Dict] = []

    print("\nSubsampling WITHOUT replacement and computing H1 diagrams...")
    for s in SUBJECTS:
        ids = subj_order[s]
        for roi in ROIS:
            D_shift_full = D_shift_map[(s,roi)]
            maxdist = maxdist_map[(s,roi)]
            persist_min = PERSIST_MIN_FRAC_OF_MAXDIST * maxdist
            eps_match = CLUSTER_EPS_FRAC_OF_MAXDIST * maxdist

            print(f"  subj{s:02d} ROI {roi}: B={B}, m={m}/{N}, persist_min={persist_min:.4g}, eps_match={eps_match:.4g}")

            pts: List[Tuple[int, float, float, float]] = []

            for b in range(B):
                # KEY FIX: subsample WITHOUT replacement
                idx = rng.choice(N, size=m, replace=False)
                Db = D_shift_full[np.ix_(idx, idx)]

                res = ripser_call(Db, thresh=maxdist, do_cocycles=False)
                dgm_h1 = res["dgms"][1] if len(res["dgms"]) > 1 else np.zeros((0,2))
                kept = filter_h1_points(dgm_h1, persist_min=persist_min)

                # Keep only top-L by persistence (if more)
                if kept.shape[0] > TOP_L_PER_SUBSAMPLE:
                    kept = kept[np.argsort(-kept[:, 2])[:TOP_L_PER_SUBSAMPLE]]

                for (birth, death, pers) in kept:
                    pts.append((b, float(birth), float(death), float(pers)))
                    long_rows.append(dict(
                        subject=s, roi=roi, bootstrap=b,
                        subsample_size=m, subsample_frac=float(SUBSAMPLE_FRAC),
                        birth=float(birth), death=float(death), persistence=float(pers),
                        persist_min=float(persist_min), eps_match=float(eps_match),
                        maxdist_used=float(maxdist), shift_added=float(shift_added_map[(s,roi)]),
                    ))

                if (b + 1) % 25 == 0:
                    print(f"    done {b+1}/{B}")

            # If nothing survived noise floor, then no robust loops here
            if len(pts) == 0:
                print("    (no H1 points above persistence floor; skipping clustering)")
                continue

            # Cluster in (birth,death) space
            clusters = cluster_points(pts, eps_match=eps_match)

            # Summarize clusters
            summaries = []
            for cid, c in enumerate(clusters):
                hit_rate = len(c.bootstraps_seen) / B
                b_med, d_med, p_med = choose_representative(c)

                summaries.append(dict(
                    subject=s, roi=roi,
                    cluster_id=cid,
                    hit_rate=float(hit_rate),
                    n_points=int(len(c.points)),
                    n_bootstraps=int(len(c.bootstraps_seen)),
                    birth_median=float(b_med),
                    death_median=float(d_med),
                    persistence_median=float(p_med),
                    eps_match=float(eps_match),
                    persist_min=float(persist_min),
                    maxdist_used=float(maxdist),
                    shift_added=float(shift_added_map[(s,roi)]),
                    subsample_size=m,
                    subsample_frac=float(SUBSAMPLE_FRAC),
                ))

            cs = pd.DataFrame(summaries).sort_values(
                ["hit_rate", "persistence_median"],
                ascending=[False, False]
            )

            robust_cs = cs[cs["hit_rate"] >= ROBUST_CLUSTER_HITRATE_MIN].copy()
            print(f"    clusters total={len(cs)} robust={len(robust_cs)}")

            if len(robust_cs) == 0:
                continue

            robust_rows.extend(robust_cs.to_dict(orient="records"))

            # Cycle extraction (FULL graph) for each robust cluster
            print(f"    extracting cycles for {len(robust_cs)} robust clusters (full graph)...")
            res_full = ripser_call(D_shift_full, thresh=maxdist, do_cocycles=True)

            for _, r in robust_cs.iterrows():
                birth = float(r["birth_median"])
                death = float(r["death_median"])
                pers  = float(r["persistence_median"])
                cid   = int(r["cluster_id"])
                hrate = float(r["hit_rate"])

                out = extract_simple_cycle_full_graph(
                    D_shift_full=D_shift_full,
                    res_full=res_full,
                    birth=birth,
                    death=death,
                    pers=pers
                )
                if out is None:
                    continue

                cycle, eps_used = out

                # Store ordered nodes with nsdId mapping (cycle already in full graph indices)
                for step, node_idx in enumerate(cycle):
                    cycle_rows.append(dict(
                        subject=s, roi=roi,
                        cluster_id=cid,
                        hit_rate=hrate,
                        birth=birth, death=death, persistence=pers,
                        eps_used=float(eps_used),
                        step=int(step),
                        node_index=int(node_idx),
                        nsdId=ids[int(node_idx)],
                    ))

    # Write outputs
    long_df = pd.DataFrame(long_rows)
    out_long = OUT_DIR / "H1_points_long.csv"
    long_df.to_csv(out_long, index=False)
    print(f"\nWrote: {out_long}")

    robust_df = pd.DataFrame(robust_rows)
    out_robust = OUT_DIR / "robust_clusters_H1.csv"
    robust_df.to_csv(out_robust, index=False)
    print(f"Wrote: {out_robust}")

    cycles_df = pd.DataFrame(cycle_rows)
    out_cycles = OUT_DIR / "robust_cycles_nodes.csv"
    cycles_df.to_csv(out_cycles, index=False)
    print(f"Wrote: {out_cycles}")

    # ROI-level summaries (what you want to report)
    if len(robust_df) > 0:
        print("\nROI-level robust cluster counts (across subjects):")
        print(robust_df.groupby("roi").size().sort_values(ascending=False))
        print("\nPer subject × ROI robust cluster counts:")
        print(robust_df.groupby(["subject","roi"]).size())
        print("\nSubjects with >=1 robust loop per ROI:")
        tmp = robust_df.groupby(["roi","subject"]).size().reset_index(name="n")
        print(tmp.groupby("roi")["subject"].nunique().sort_values(ascending=False))

    dt = time.time() - t0
    print(f"\n✅ DONE in {dt/60:.1f} min.")


if __name__ == "__main__":
    main()