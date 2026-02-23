#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ripser import ripser

# ---------------- CONFIG ----------------
RDM_PATH = Path(r"D:\Emily\NSD_cogsci_sandbox\COGS401\COSYNE\RDMs_Final\subjects\subj01\ROI-V1_crossnobis_STRICT3_sessionSplit_raw.npy")

OUT_DIR = Path(r"D:\Emily\NSD_cogsci_sandbox\COGS401\COSYNE\RDMs_Final\PH_VR_ripser\poster_panels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_NULL = 50
SEED = 123

# If crossnobis includes negatives, we shift to be a proper distance matrix.
SHIFT_IF_NEGATIVE = True

# Null mode: "shuffle_distances" matches what you already ran successfully
NULL_MODE = "shuffle_distances"

# ---------------- HELPERS ----------------
def sanitize_distance_matrix(D: np.ndarray, shift_if_negative: bool = True) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)

    if shift_if_negative:
        iu = np.triu_indices_from(D, k=1)
        m = float(np.min(D[iu]))
        if m < 0:
            D = D + (-m + 1e-6)
            np.fill_diagonal(D, 0.0)
    return D

def max_offdiag(D: np.ndarray) -> float:
    iu = np.triu_indices_from(D, k=1)
    return float(np.max(D[iu]))

# ---------------- MAIN ----------------
def main():
    if not RDM_PATH.exists():
        raise FileNotFoundError(f"RDM file not found:\n{RDM_PATH}")

    D = np.load(RDM_PATH)
    D = sanitize_distance_matrix(D, shift_if_negative=SHIFT_IF_NEGATIVE)

    iu = np.triu_indices_from(D, k=1)
    maxdist = max_offdiag(D)

    # ----- REAL -----
    res_real = ripser(D, distance_matrix=True, maxdim=1, thresh=maxdist)
    dgm1_real = res_real["dgms"][1]
    pers_real = dgm1_real[:, 1] - dgm1_real[:, 0]
    real_max = float(np.max(pers_real)) if len(pers_real) else 0.0

    # ----- NULLS -----
    rng = np.random.default_rng(SEED)
    null_max = []

    if NULL_MODE == "shuffle_distances":
        base_vals = D[iu].copy()
        for _ in range(N_NULL):
            vals = base_vals.copy()
            rng.shuffle(vals)
            Dn = np.zeros_like(D)
            Dn[iu] = vals
            Dn = Dn + Dn.T
            np.fill_diagonal(Dn, 0.0)

            res = ripser(Dn, distance_matrix=True, maxdim=1, thresh=maxdist)
            dgm1 = res["dgms"][1]
            pers = dgm1[:, 1] - dgm1[:, 0]
            null_max.append(float(np.max(pers)) if len(pers) else 0.0)
    else:
        raise ValueError("Unknown NULL_MODE. Use NULL_MODE='shuffle_distances'.")

    null_max = np.array(null_max, dtype=float)
    p_emp = float(np.mean(null_max >= real_max))

    # ----- PLOT -----
    plt.figure(figsize=(7, 4))
    plt.hist(null_max, bins=12, alpha=0.85)
    plt.axvline(real_max, linewidth=3)
    plt.title(
        f"Null sanity check: max H1 persistence\n"
        f"real={real_max:.3f}, null mean={null_max.mean():.3f}±{null_max.std():.3f}, "
        f"max null={null_max.max():.3f}, p={p_emp:.3f}\n"
        f"subj01 V1 crossnobis (STRICT3 sessionSplit)"
    )
    plt.xlabel("max H1 persistence")
    plt.ylabel("count")
    plt.tight_layout()

    out_png = OUT_DIR / "null_sanity_hist_subj01_V1.png"
    plt.savefig(out_png, dpi=300)
    plt.close()

    # ----- TEXT SUMMARY -----
    out_txt = OUT_DIR / "null_sanity_summary_subj01_V1.txt"
    with open(out_txt, "w") as f:
        f.write(f"RDM: {RDM_PATH}\n")
        f.write(f"N_NULL={N_NULL}, SEED={SEED}, NULL_MODE={NULL_MODE}\n")
        f.write(f"SHIFT_IF_NEGATIVE={SHIFT_IF_NEGATIVE}\n\n")
        f.write(f"Max persistence (real): {real_max}\n")
        f.write(f"Null mean: {null_max.mean()}\n")
        f.write(f"Null std: {null_max.std()}\n")
        f.write(f"Max null: {null_max.max()}\n")
        f.write(f"Empirical p-value (null >= real): {p_emp}\n")
        f.write(f"Real ranks higher than {(null_max < real_max).sum()}/{N_NULL} null draws\n")

    print("Saved:", out_png)
    print("Saved:", out_txt)
    print("Max persistence (real):", real_max)
    print("Null mean:", null_max.mean(), "std:", null_max.std(), "max null:", null_max.max())
    print("Empirical p-value:", p_emp)

if __name__ == "__main__":
    main()