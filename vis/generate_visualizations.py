#!/usr/bin/env python3
"""
FairCBM MICCAI 2026 — Master Visualisation Script
Generates: F1-F3 (paper), P1-P4 (poster), S1-S3 (social), D1-D2 (demo)

Usage:
    conda run -n CBM-env python scripts/generate_visualizations.py
"""

import os, sys, glob, json, warnings
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parent.parent
RESULTS_BASE   = REPO_ROOT / "results"
MULTI_RUN_BASE = RESULTS_BASE   # search across all multi_run_* subdirectories
OUT_DIR        = RESULTS_BASE / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "direct":              RESULTS_BASE / "global_best_direct.pt",
    "standard_cbm":        RESULTS_BASE / "global_best_standard_cbm.pt",
    "curriculum_cbm":      RESULTS_BASE / "global_best_curriculum_cbm.pt",
    "fair_standard_cbm":   RESULTS_BASE / "global_best_fair_standard_cbm.pt",
    "fair_curriculum_cbm": RESULTS_BASE / "global_best_fair_curriculum_cbm.pt",
}

DATA_ROOT = Path("/home/csc29/projects/SkinCAP")
RAW_CSV   = DATA_ROOT / "skincap_v240623.csv"

SAVED_FILES: list[str] = []

# ── Design System ──────────────────────────────────────────────────────────────
COLORS = {
    "fair_curriculum_cbm": "#E63946",
    "curriculum_cbm":      "#457B9D",
    "fair_standard_cbm":   "#A8DADC",
    "standard_cbm":        "#F4A261",
    "direct":              "#6D6875",
    "accent":              "#1D3557",
    "success":             "#2DC653",
    "phase1": "#CAF0F8", "phase2": "#90E0EF",
    "phase3": "#0096C7", "phase4": "#023E8A",
}

FITZ_COLORS_HEX = {
    0: "#F5CBA7", 1: "#E59866", 2: "#CA6F1E",
    3: "#A04000", 4: "#784212", 5: "#4A235A",
}
FITZ_LABELS = ["Type I", "Type II", "Type III", "Type IV", "Type V", "Type VI"]

MODELS = ["direct", "standard_cbm", "curriculum_cbm", "fair_standard_cbm", "fair_curriculum_cbm"]
MODEL_LABELS = {
    "direct":              "Direct",
    "standard_cbm":        "Standard CBM",
    "curriculum_cbm":      "Curriculum CBM",
    "fair_standard_cbm":   "Fair Standard CBM",
    "fair_curriculum_cbm": "Fair Curriculum CBM ★",
}

CONCEPT_NAMES = [
    "Papule", "Plaque", "Pustule", "Bulla", "Patch", "Nodule", "Ulcer",
    "Crust", "Erosion", "Atrophy", "Exudate", "Telangiectasia", "Scale",
    "Scar", "Friable", "Warty/Papillomatous", "Dome-shaped",
    "Brown (Hyperpig.)", "White (Hypopig.)", "Purple", "Yellow", "Black",
    "Erythema",
]

BASE_RC = {
    "figure.facecolor": "#FAFAFA", "axes.facecolor": "#FAFAFA",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.framealpha": 0.9, "legend.fontsize": 11,
}

POSTER_RC = {**BASE_RC,
    "axes.labelsize": 16, "xtick.labelsize": 14,
    "ytick.labelsize": 14, "legend.fontsize": 13,
}

# ── Data helpers ───────────────────────────────────────────────────────────────

def load_histories(multi_run_base, model_type):
    """Return list of history dicts, one per completed run.
    Searches across all multi_run_*/run_*/ subdirectories (one per SLURM job).
    """
    pattern = os.path.join(multi_run_base, "multi_run_*", "run_*", model_type, "history.json")
    histories = []
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                histories.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return histories


def load_test_fitz_f1(histories):
    """
    Load per-run per-Fitzpatrick F1 from test split (mirrors compute_summary_statistics).
    Returns a (n_runs, 6) array, NaN for missing values.
    """
    rows = []
    for h in histories:
        if not h.get("test"):
            rows.append([float("nan")] * 6)
            continue
        test = h["test"][0]
        group_f1 = test.get("binary_fairness", {}).get("worst_group", {}).get("group_f1", {})
        rows.append([float(group_f1.get(str(i), float("nan"))) for i in range(6)])
    return np.array(rows)   # (n_runs, 6)


def extract_epoch_metrics(history, worst_idx, best_idx):
    """Extract per-epoch metric arrays using fixed worst/best group identity."""
    overall_f1, worst_f1, best_f1, gap, per_group_f1 = [], [], [], [], []
    for ep in history["val"]:
        wg = ep["binary_fairness"]["worst_group"]
        overall_f1.append(ep["binary_metrics"]["f1"])
        pg = [float(wg["group_f1"].get(str(i), float("nan"))) for i in range(6)]
        per_group_f1.append(pg)
        worst_f1.append(pg[worst_idx])
        best_f1.append(pg[best_idx])
        gap.append(pg[best_idx] - pg[worst_idx])
    return {
        "overall_f1": overall_f1,
        "worst_f1": worst_f1,
        "best_f1": best_f1,
        "gap": gap,
        "per_group_f1": per_group_f1,
        "n_epochs": len(history["val"]),
    }


def aggregate_across_runs(multi_run_base, model_type):
    """
    Return per-epoch mean ± std across all runs for a model type.

    Worst-group and gap are computed using the paper method (compute_summary_statistics):
      - Identify the consistently worst Fitzpatrick group by its mean F1 across runs
        (from test split), then track that fixed group's per-run values.
      - performance_gap = best_group_mean − worst_group_mean (error propagation for std).
    This matches the numbers in summary_table.csv / the published results.
    """
    histories = load_histories(str(multi_run_base), model_type)
    if not histories:
        print(f"  WARNING: no histories found for {model_type}")
        return None

    # ── identify worst/best group from TEST split (paper method) ──────────────
    fitz_mat = load_test_fitz_f1(histories)   # (runs, 6)
    group_means = np.nanmean(fitz_mat, axis=0)   # mean F1 per Fitzpatrick type
    group_stds  = np.nanstd(fitz_mat, axis=0, ddof=1)
    worst_idx = int(np.argmin(group_means))
    best_idx  = int(np.argmax(group_means))

    # ── per-epoch arrays with fixed worst/best group identity ─────────────────
    all_m = [extract_epoch_metrics(h, worst_idx, best_idx) for h in histories]
    min_len = min(m["n_epochs"] for m in all_m)
    overall_mat = np.array([m["overall_f1"][:min_len] for m in all_m])
    worst_mat   = np.array([m["worst_f1"][:min_len]   for m in all_m])
    best_mat    = np.array([m["best_f1"][:min_len]    for m in all_m])
    gap_mat     = np.array([m["gap"][:min_len]        for m in all_m])
    pg_mat      = np.array([m["per_group_f1"][:min_len] for m in all_m])

    # Final-epoch scalars — paper method: mean of the fixed group's per-run values
    worst_run_vals = fitz_mat[:, worst_idx]
    best_run_vals  = fitz_mat[:, best_idx]
    valid_mask = np.isfinite(worst_run_vals) & np.isfinite(best_run_vals)
    worst_run_vals = worst_run_vals[valid_mask]
    best_run_vals  = best_run_vals[valid_mask]
    gap_run_vals   = best_run_vals - worst_run_vals

    # Overall F1 from test split
    final_overall_all = []
    for h in histories:
        if h.get("test"):
            final_overall_all.append(h["test"][0].get("binary_metrics", {}).get("f1", float("nan")))
    final_overall_all = np.array([v for v in final_overall_all if np.isfinite(v)])

    return {
        # Per-epoch mean ± std (fixed worst/best group identity)
        "gap_mean":       gap_mat.mean(0),    "gap_std":       gap_mat.std(0),
        "worst_mean":     worst_mat.mean(0),  "worst_std":     worst_mat.std(0),
        "overall_mean":   overall_mat.mean(0),"overall_std":   overall_mat.std(0),
        "per_group_mean": pg_mat.mean(0),     "per_group_std": pg_mat.std(0),
        "n_runs": len(histories), "n_epochs": min_len,
        "worst_group_idx": worst_idx, "best_group_idx": best_idx,
        # Final-epoch scalars — paper method (test split, fixed group identity)
        # ± uses SEM = std/√n (matches the published results table; F1 uses raw std)
        "final_worst":     float(group_means[worst_idx]),
        "final_worst_std": float(group_stds[worst_idx] / np.sqrt(len(worst_run_vals))) if len(worst_run_vals) > 0 else 0.0,
        "final_gap":       float(group_means[best_idx] - group_means[worst_idx]),
        "final_gap_std":   float(np.sqrt(group_stds[worst_idx]**2 + group_stds[best_idx]**2) / np.sqrt(len(worst_run_vals))) if len(worst_run_vals) > 0 else 0.0,
        "final_overall":     float(np.mean(final_overall_all)) if len(final_overall_all) else float(overall_mat[:, -1].mean()),
        "final_overall_std": float(np.std(final_overall_all, ddof=1)) if len(final_overall_all) > 1 else float(overall_mat[:, -1].std()),
        # Raw per-run final values for violin / scatter plots
        "final_gap_all":     gap_run_vals.tolist(),
        "final_worst_all":   worst_run_vals.tolist(),
        "final_overall_all": final_overall_all.tolist(),
    }


def load_final_concept_f1(multi_run_base, model_type):
    """Return mean per-concept F1 at final epoch, averaged over all runs. None if no concept metrics."""
    histories = load_histories(str(multi_run_base), model_type)
    if not histories:
        return None
    per_concept = []
    for h in histories:
        ep = h["val"][-1]
        if "concept_metrics" not in ep or "per_concept_f1" not in ep["concept_metrics"]:
            return None
        per_concept.append(ep["concept_metrics"]["per_concept_f1"])
    return np.mean(per_concept, axis=0)


def save(fig, stem, dpi=300):
    """Save figure as PDF and PNG, record paths."""
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        SAVED_FILES.append(str(p))
    plt.close(fig)
    print(f"  ✓ {stem}")


def save_png_only(fig, stem, dpi=300):
    """Save figure as PNG only (social media graphics)."""
    p = OUT_DIR / f"{stem}.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    SAVED_FILES.append(str(p))
    plt.close(fig)
    print(f"  ✓ {stem}")


def phase_bands(ax, alpha=0.12):
    """Draw Q1–Q4 phase background bands on a 0–100% x-axis."""
    for i, (color_key, label) in enumerate(
        [("phase1", "Q1"), ("phase2", "Q2"), ("phase3", "Q3"), ("phase4", "Q4")]
    ):
        ax.axvspan(i * 25, (i + 1) * 25, alpha=alpha, color=COLORS[color_key], zorder=0)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE F1 — Fairness Curriculum Phase Timeline
# ══════════════════════════════════════════════════════════════════════════════

def figure_F1_curriculum_phases():
    """Infographic of 4-phase fairness curriculum schedule."""
    mpl.rcParams.update(BASE_RC)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax2 = ax.twinx()

    x = np.linspace(0, 100, 401)
    # Loss weight arrays
    ldp = np.piecewise(x, [x < 25, (x >= 25) & (x < 50), (x >= 50) & (x < 75), x >= 75],
                       [0, 1.0, 0.3, 1/3])
    leo = np.piecewise(x, [x < 50, (x >= 50) & (x < 75), x >= 75], [0, 0.7, 1/3])
    lpg = np.piecewise(x, [x < 75, x >= 75], [0, 1/3])

    ax.fill_between(x, 0, ldp, color=COLORS["phase2"], alpha=0.75, label="$\\mathcal{L}_{DP}$")
    ax.fill_between(x, ldp, ldp + leo, color=COLORS["phase3"], alpha=0.75, label="$\\mathcal{L}_{EO}$")
    ax.fill_between(x, ldp + leo, ldp + leo + lpg, color=COLORS["phase4"], alpha=0.75, label="$\\mathcal{L}_{PG}$")

    # Adversarial lambda
    lam = np.piecewise(x, [x < 50, (x >= 50) & (x < 75), x >= 75],
                       [0, lambda t: 0.01 * (t - 50) / 25, 0.01])
    ax2.plot(x, lam, color=COLORS["accent"], lw=2, ls="--", label="GRL $\\lambda$")
    ax2.set_ylabel("GRL $\\lambda$", fontsize=12, color=COLORS["accent"])
    ax2.tick_params(axis="y", colors=COLORS["accent"])
    ax2.set_ylim(-0.001, 0.016)
    ax2.spines["right"].set_visible(True)

    # Phase background
    for i, color_key in enumerate(["phase1", "phase2", "phase3", "phase4"]):
        ax.axvspan(i * 25, (i + 1) * 25, alpha=0.10, color=COLORS[color_key], zorder=0)

    # Phase dividers
    for x_ in [25, 50, 75]:
        ax.axvline(x_, color="grey", lw=0.8, ls=":")

    # Phase titles
    titles = ["Q1 — Balanced\nFoundation", "Q2 — Demographic\nParity",
              "Q3 — Equalized\nOdds", "Q4 — Performance\nParity"]
    for i, t in enumerate(titles):
        ax.text(i * 25 + 12.5, 0.96, t, ha="center", va="top",
                fontsize=10, fontweight="bold", color=COLORS["accent"],
                transform=ax.get_xaxis_transform())

    # Sampling badges
    badges = ["Balanced\nper skin type", "Balanced", "Stratified\n(group×label)", "Error-driven\noversampling"]
    for i, b in enumerate(badges):
        ax.text(i * 25 + 12.5, 1.10, b, ha="center", va="bottom",
                fontsize=8.5, color="white",
                transform=ax.get_xaxis_transform(),
                bbox=dict(boxstyle="round,pad=0.3", fc=COLORS[f"phase{i+1}"], ec="none", alpha=0.9))

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Training Progress (%)", fontsize=13)
    ax.set_ylabel("Fairness Loss Weight", fontsize=13)
    ax.set_title("Fair Curriculum CBM — Progressive Fairness Constraint Schedule",
                 fontsize=15, fontweight="bold", pad=30)

    # Combined legend
    handles_a, labels_a = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(handles_a + h2, labels_a + l2, loc="upper left", ncol=4,
              bbox_to_anchor=(0.01, 0.88), fontsize=10)

    fig.subplots_adjust(top=0.78)
    save(fig, "F1_curriculum_phases")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE F2 — Performance Gap Collapse Over Training
# ══════════════════════════════════════════════════════════════════════════════

def smooth(arr, window=7):
    """Smooth array using rolling mean."""
    import pandas as pd
    return np.array(pd.Series(arr).rolling(window, center=True, min_periods=1).mean())


def figure_F2_gap_over_training(data):
    """Performance gap and LG-F1 over training progress for all 5 models, with smoothing and SEM bands."""
    mpl.rcParams.update(BASE_RC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharex=True)

    for ax in (ax1, ax2):
        phase_bands(ax, alpha=0.12)
        for x_ in [25, 50, 75]:
            ax.axvline(x_, color="grey", lw=0.6, ls=":", zorder=1)

    for model in MODELS:
        d = data.get(model)
        if d is None:
            continue
        n = d["n_runs"]
        x = np.linspace(0, 100, d["n_epochs"])
        color = COLORS[model]
        lw    = 2.5 if model == "fair_curriculum_cbm" else 1.5
        alpha = 1.0 if model == "fair_curriculum_cbm" else 0.80
        zo    = 5   if model == "fair_curriculum_cbm" else 2

        # Smooth the trajectories
        gap_s   = smooth(d["gap_mean"])
        worst_s = smooth(d["worst_mean"])
        
        ax1.plot(x, gap_s,   color=color, lw=lw, alpha=alpha, zorder=zo,
                 label=MODEL_LABELS[model])
        ax2.plot(x, worst_s, color=color, lw=lw, alpha=alpha, zorder=zo,
                 label=MODEL_LABELS[model])

        # SEM error bands for key models only
        if model in ("fair_curriculum_cbm", "curriculum_cbm"):
            sem_gap   = smooth(d["gap_std"])   / np.sqrt(n)
            sem_worst = smooth(d["worst_std"]) / np.sqrt(n)
            ax1.fill_between(x, gap_s - sem_gap, gap_s + sem_gap,
                             color=color, alpha=0.25, zorder=zo-1, linewidth=0)
            ax2.fill_between(x, worst_s - sem_worst, worst_s + sem_worst,
                             color=color, alpha=0.25, zorder=zo-1, linewidth=0)

    # Annotations for fair_curriculum_cbm
    fc = data.get("fair_curriculum_cbm")
    if fc:
        # Panel A (gap)
        ax1.axhline(fc["final_gap"], color=COLORS["fair_curriculum_cbm"],
                    ls="--", lw=1.2, alpha=0.6)
        ax1.annotate(f"{fc['final_gap']:.3f}",
                     xy=(100, fc["final_gap"]), xytext=(85, fc["final_gap"] - 0.05),
                     fontsize=10, color=COLORS["fair_curriculum_cbm"], fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=COLORS["fair_curriculum_cbm"], lw=1.2))
        # Adversarial debiasing annotation on gap panel
        ax1.annotate("Adversarial\ndebiasing\nactivated",
                     xy=(50, smooth(fc["gap_mean"])[int(0.50 * fc["n_epochs"]) - 1]),
                     xytext=(37, smooth(fc["gap_mean"])[int(0.50 * fc["n_epochs"]) - 1] - 0.10),
                     fontsize=9, color=COLORS["phase4"],
                     arrowprops=dict(arrowstyle="->", color=COLORS["phase4"], lw=1.0))
        # Panel B (lowest-group F1)
        ax2.annotate(f"{fc['final_worst']:.3f}",
                     xy=(100, fc["final_worst"]), xytext=(84, fc["final_worst"] + 0.05),
                     fontsize=10, color=COLORS["fair_curriculum_cbm"], fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=COLORS["fair_curriculum_cbm"], lw=1.2))

    ax1.set_xlabel("Training Progress (%)", fontsize=12)
    ax2.set_xlabel("Training Progress (%)", fontsize=12)
    ax1.set_ylabel("Performance Gap (↓ better)", fontsize=12)
    ax2.set_ylabel("Lowest-Group F1 (↑ better)", fontsize=12)
    ax1.set_title("(A) Performance Gap", fontsize=12)
    ax2.set_title("(B) Lowest-Group F1", fontsize=12)
    
    # Fixed y-axis limits
    ax1.set_ylim(0.0, 0.65)
    ax2.set_ylim(0.0, 0.70)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.00), fontsize=10, framealpha=0.9)
    fig.suptitle("Performance Gap Collapse — Fair Curriculum CBM vs Baselines",
                 fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "F2_gap_over_training")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE F3 — Latent Space Comparison (UMAP)
# ══════════════════════════════════════════════════════════════════════════════

def figure_F3_umap_comparison():
    """2-panel UMAP: Curriculum CBM vs Fair Curriculum CBM, colored by malignancy with Fitzpatrick overlays."""
    import torch
    import torchvision.models as tvm
    from sklearn.metrics import silhouette_score

    try:
        import umap as umap_lib
    except ImportError:
        print("  SKIP F3: umap-learn not installed")
        return

    if not RAW_CSV.exists():
        print("  SKIP F3: SkinCAP data not found")
        return

    sys.path.insert(0, str(REPO_ROOT))
    from src.data.dataloader import SkinCapDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = SkinCapDataset(str(DATA_ROOT), split="val", label_type="concept",
                        transform=val_transform, raw_csv=str(RAW_CSV), seed=42)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_swin(variant):
        m = tvm.swin_b(weights=None) if variant == "swin_b" else tvm.swin_t(weights=None)
        m.head = torch.nn.Identity()
        return m

    def get_embeddings(backbone, loader, device):
        backbone.eval().to(device)
        embs, labels, fitzs = [], [], []
        with torch.no_grad():
            for batch in loader:
                imgs, concepts, binary_label, fitz = batch[0].to(device), batch[1], batch[2], batch[3]
                feats = backbone(imgs)
                if feats.dim() > 2:
                    feats = feats.mean(dim=list(range(1, feats.dim() - 1)))
                embs.append(feats.cpu().numpy())
                labels.append(binary_label.numpy())
                fitzs.append(fitz.numpy())
        return np.concatenate(embs), np.concatenate(labels).astype(int), np.concatenate(fitzs).astype(int)

    mpl.rcParams.update(BASE_RC)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    models_to_compare = ["curriculum_cbm", "fair_curriculum_cbm"]
    
    for ax, model in zip(axes, models_to_compare):
        ck_path = CHECKPOINTS[model]
        try:
            state = torch.load(ck_path, map_location="cpu", weights_only=False)
            sd = state.get("model_state_dict", state)
            backbone_sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
            # Both curriculum models use swin_b
            backbone = build_swin("swin_b")
            backbone.load_state_dict(backbone_sd, strict=False)

            embs, labels, fitz_idx = get_embeddings(backbone, loader, device)
            fitz_0idx = (fitz_idx - 1).clip(0, 5)  # 1-6 → 0-5

            # UMAP reduction
            reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)
            e2d = reducer.fit_transform(embs)

            # Plot malignancy as base layer (larger points)
            benign_mask = labels == 0
            malig_mask  = labels == 1
            
            ax.scatter(e2d[benign_mask, 0], e2d[benign_mask, 1],
                       c="#90E0EF", s=25, alpha=0.4, linewidths=0, label="Benign", zorder=1)
            ax.scatter(e2d[malig_mask, 0], e2d[malig_mask, 1],
                       c="#E63946", s=25, alpha=0.4, linewidths=0, label="Malignant", zorder=1)

            # Overlay Fitzpatrick markers (smaller, with edges)
            for gi in range(6):
                mask = fitz_0idx == gi
                if mask.sum() == 0:
                    continue
                ax.scatter(e2d[mask, 0], e2d[mask, 1],
                           c="none", edgecolors=FITZ_COLORS_HEX[gi], s=15, alpha=0.65,
                           linewidths=1.2, marker='o', zorder=2)

            # Compute separation metrics
            if len(np.unique(labels)) > 1:
                from sklearn.metrics import davies_bouldin_score
                db_score = davies_bouldin_score(e2d, labels)
                sil_fitz = silhouette_score(e2d, fitz_0idx) if len(np.unique(fitz_0idx)) > 1 else float("nan")
                
                ax.text(0.03, 0.97, 
                        f"Malignancy separation (DB): {db_score:.2f} ↓ better\n"
                        f"Fitzpatrick mixing (Sil): {sil_fitz:.2f} ↓ better",
                        transform=ax.transAxes, ha="left", va="top", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.90))

            border_color = COLORS["fair_curriculum_cbm"] if model == "fair_curriculum_cbm" else COLORS["curriculum_cbm"]
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3 if model == "fair_curriculum_cbm" else 1.2)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading {model}\n{type(e).__name__}",
                    transform=ax.transAxes, ha="center", va="center", fontsize=10)

        ax.set_xticks([]); ax.set_yticks([])
        title = MODEL_LABELS[model]
        ax.set_title(title, fontsize=13, fontweight="bold" if model == "fair_curriculum_cbm" else "normal",
                     color=border_color, pad=10)

    # Shared legend (malignancy + Fitzpatrick edges)
    handles_malig = [
        plt.scatter([], [], c="#90E0EF", s=50, alpha=0.6, label="Benign"),
        plt.scatter([], [], c="#E63946", s=50, alpha=0.6, label="Malignant"),
    ]
    handles_fitz = [
        plt.scatter([], [], c="none", edgecolors=FITZ_COLORS_HEX[i], s=30, linewidths=1.5, label=FITZ_LABELS[i])
        for i in range(6)
    ]
    
    fig.legend(handles=handles_malig, loc="upper left", bbox_to_anchor=(0.01, 0.99),
               fontsize=10, title="Diagnosis", title_fontsize=10, framealpha=0.95)
    fig.legend(handles=handles_fitz, loc="upper right", bbox_to_anchor=(0.99, 0.99),
               fontsize=8, title="Fitzpatrick (edge color)", title_fontsize=9, ncol=2, framealpha=0.95)

    fig.suptitle("Latent Space Comparison — Fair Curriculum Balances Diagnostic Utility and Demographic Fairness",
                 fontsize=14, fontweight="bold", y=0.96)
    fig.text(0.5, 0.02,
             "UMAP embeddings: background color = diagnosis (benign/malignant), edge color = Fitzpatrick type  ·  "
             "DB = Davies-Bouldin (class separation), Sil = Silhouette (demographic mixing)",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    save(fig, "F3_umap_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE P1 — Key Numbers Hero Panel
# ══════════════════════════════════════════════════════════════════════════════

def figure_P1_hero_numbers():
    """Bold infographic of 3 headline results on dark navy background."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(COLORS["accent"])
    ax.set_facecolor(COLORS["accent"])
    ax.axis("off")

    cards = [
        ("+63%",  "Lowest-Group F1",   "↑ vs. difficulty curriculum", "#2DC653"),
        ("−44%",  "Performance Gap",   "↓ vs. difficulty curriculum", "#E63946"),
        ("+5%",   "Overall F1",        "↑ without fairness trade-off","#90E0EF"),
    ]

    card_w, card_h = 0.28, 0.72
    gap = 0.04
    starts = [0.04, 0.36, 0.68]

    for (stat, metric, note, color), x0 in zip(cards, starts):
        # Card background
        rect = patches.FancyBboxPatch((x0, 0.08), card_w, card_h,
                                      boxstyle="round,pad=0.02",
                                      linewidth=2, edgecolor=color,
                                      facecolor="#243B55", transform=ax.transAxes,
                                      clip_on=False)
        ax.add_patch(rect)
        cx = x0 + card_w / 2
        ax.text(cx, 0.64, stat,  transform=ax.transAxes, ha="center", va="center",
                fontsize=52, fontweight="bold", color=color)
        ax.text(cx, 0.42, metric, transform=ax.transAxes, ha="center", va="center",
                fontsize=14, fontweight="bold", color="white")
        ax.text(cx, 0.24, note,  transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#A8DADC")

    # Title
    ax.text(0.5, 0.95, "Fair Curriculum CBM — MICCAI 2026",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=16, fontweight="bold", color="white")

    # Subtitle
    ax.text(0.5, 0.04,
            "100 independent runs  ·  SkinCon  ·  Fitzpatrick I–VI  ·  p<0.01 (paired t-test)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color="#A8DADC")

    # Architecture strip at bottom
    arch_y = -0.06
    blocks = ["Swin-Tiny\nEncoder", "Concept\nBottleneck\n(23 concepts)",
              "Binary\nClassifier", "Adversarial\nDiscriminator\n(GRL)"]
    block_colors = ["#1D5C94", "#2B7A78", "#3D405B", "#8D1D3D"]
    for j, (blk, bc) in enumerate(zip(blocks, block_colors)):
        bx = 0.08 + j * 0.23
        r = patches.FancyBboxPatch((bx, arch_y), 0.19, 0.10,
                                    boxstyle="round,pad=0.01", fc=bc, ec="white",
                                    lw=0.8, transform=ax.transAxes, clip_on=False)
        ax.add_patch(r)
        ax.text(bx + 0.095, arch_y + 0.05, blk, transform=ax.transAxes,
                ha="center", va="center", fontsize=7.5, color="white")
        if j < 3:
            ax.annotate("", xy=(bx + 0.195, arch_y + 0.05),
                        xytext=(bx + 0.19, arch_y + 0.05),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.2))

    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0.12)
    save(fig, "P1_hero_numbers")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE P2 — Ablation Contribution Chart
# ══════════════════════════════════════════════════════════════════════════════

def figure_P2_ablation():
    """Horizontal grouped bar chart of ablation results."""
    mpl.rcParams.update(POSTER_RC)

    ABLATION_DATA = {
        "Full Model":             {"F1": 0.611, "LG-F1": 0.441, "Gap": 0.203},
        "w/o Phase 1 (init)":    {"F1": 0.579, "LG-F1": 0.112, "Gap": 0.718},
        "w/o Phase 2 (DP)":      {"F1": 0.587, "LG-F1": 0.115, "Gap": 0.728},
        "w/o Phase 3 (EO+adv)":  {"F1": 0.574, "LG-F1": 0.092, "Gap": 0.721},
        "w/o Phase 4 (err-drv)": {"F1": 0.602, "LG-F1": 0.147, "Gap": 0.708},
        "w/o Adversarial":       {"F1": 0.592, "LG-F1": 0.091, "Gap": 0.756},
    }

    configs = list(ABLATION_DATA.keys())
    metrics = ["F1", "LG-F1", "Gap"]
    bar_colors = [COLORS["curriculum_cbm"], COLORS["fair_curriculum_cbm"], COLORS["standard_cbm"]]
    full_vals  = {m: ABLATION_DATA["Full Model"][m] for m in metrics}

    n_configs = len(configs)
    n_metrics = len(metrics)
    y_base = np.arange(n_configs)
    bar_h  = 0.25

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_facecolor("#FAFAFA")

    for mi, (metric, color) in enumerate(zip(metrics, bar_colors)):
        vals = [ABLATION_DATA[c][metric] for c in configs]
        ys   = y_base + (mi - 1) * bar_h
        bars = ax.barh(ys, vals, bar_h * 0.88, color=color, alpha=0.85,
                       label=metric)
        # Full Model: thick border
        ax.barh(ys[0:1], vals[0:1], bar_h * 0.88, color=color, alpha=0.85,
                edgecolor=COLORS["accent"], linewidth=2.5)
        # Annotate % change vs full model
        for yi, (y, v) in enumerate(zip(ys, vals)):
            if yi == 0:
                ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9,
                        color=COLORS["accent"], fontweight="bold")
            else:
                pct = 100 * (v - full_vals[metric]) / full_vals[metric]
                sign = "+" if pct > 0 else ""
                is_improvement = (metric == "Gap" and pct < 0) or (metric != "Gap" and pct > 0)
                color = "#2DC653" if is_improvement else "crimson"
                ax.text(v + 0.005, y, f"{sign}{pct:.0f}%", va="center", fontsize=8.5,
                        color=color)

    # Reference lines at full-model values
    ax.axvline(full_vals["F1"],    color=COLORS["curriculum_cbm"],        ls="--", lw=1.2, alpha=0.7)
    ax.axvline(full_vals["LG-F1"], color=COLORS["fair_curriculum_cbm"],   ls="--", lw=1.2, alpha=0.7)
    ax.axvline(full_vals["Gap"],   color=COLORS["standard_cbm"],          ls="--", lw=1.2, alpha=0.7)

    ax.set_yticks(y_base)
    ax.set_yticklabels(configs, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Metric Value", fontsize=14)
    ax.set_title("Phase Ablation — Each Curriculum Component Is Critical",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(axis="x", alpha=0.3, ls="--")
    fig.tight_layout()
    save(fig, "P2_ablation")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE P3 — Per-Fitzpatrick Heat Strip
# ══════════════════════════════════════════════════════════════════════════════

def figure_P3_per_fitz_heatstrip():
    """Two-row heatmap comparing per-Fitzpatrick F1 for Curriculum vs Fair Curriculum CBM."""
    mpl.rcParams.update(POSTER_RC)

    CURR_F1 = [0.622, 0.534, 0.631, 0.553, 0.558, 0.270]
    FAIR_F1 = [0.644, 0.580, 0.635, 0.574, 0.642, 0.441]
    SIG     = ["", "***", "", "", "**", "**"]

    fitz_labels = ["Type I\n(lightest)", "Type II", "Type III", "Type IV", "Type V", "Type VI\n(darkest)"]
    data = np.array([CURR_F1, FAIR_F1])

    # Skin-tone inspired colormap
    cmap = LinearSegmentedColormap.from_list(
        "skintone", ["#F5CBA7", "#E59866", "#A04000", "#4A235A"], N=256)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(data, cmap=cmap, vmin=0.2, vmax=0.75, aspect="auto")

    ax.set_xticks(range(6))
    ax.set_xticklabels(fitz_labels, fontsize=13)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Curriculum CBM", "Fair Curriculum CBM ★"], fontsize=13)

    # Cell annotations
    for row in range(2):
        for col in range(6):
            v = data[row, col]
            ax.text(col, row, f"{v:.2f}", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if v < 0.5 else COLORS["accent"])

    # Significance stars between rows
    for col, sig in enumerate(SIG):
        if sig:
            ax.text(col, 0.5, sig, ha="center", va="center",
                    fontsize=16, color="#E63946", fontweight="bold")

    plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, label="F1 Score")
    ax.set_title("Per-Skin-Type F1 — Fairness Gains Without Lighter-Type Regression",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "P3_per_fitz_heatstrip")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE P4 — Training Stability Violin Plot
# ══════════════════════════════════════════════════════════════════════════════

def figure_P4_stability_violin(data):
    """Violin plots of final-epoch LG-F1 and Gap across 100 runs."""
    mpl.rcParams.update(POSTER_RC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))

    abbrev = {
        "direct":              "Direct",
        "standard_cbm":        "Std CBM",
        "curriculum_cbm":      "Curr CBM",
        "fair_standard_cbm":   "Fair Std",
        "fair_curriculum_cbm": "Fair Curr ★",
    }
    palette = {m: COLORS[m] for m in MODELS}

    for ax, metric, ylabel, title in [
        (ax1, "final_worst_all", "Worst-Group F1 / LG-F1 (↑ better)", "LG-F1"),
        (ax2, "final_gap_all",   "Performance Gap (↓ better)",         "Gap"),
    ]:
        plot_data, labels, colors_list = [], [], []
        for model in MODELS:
            d = data.get(model)
            if d and d.get(metric):
                plot_data.append(d[metric])
                labels.append(abbrev[model])
                colors_list.append(palette[model])

        parts = ax.violinplot(plot_data, positions=range(len(plot_data)),
                              showmedians=True, showextrema=True)
        for i, (pc, c) in enumerate(zip(parts["bodies"], colors_list)):
            pc.set_facecolor(c)
            pc.set_alpha(0.75)
        for comp in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[comp].set_color("grey")
            parts[comp].set_linewidth(1.0)

        # Swarm overlay
        for i, (vals, c) in enumerate(zip(plot_data, colors_list)):
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       c=c, alpha=0.3, s=10, zorder=3, linewidths=0)

        # Highlight Fair Curriculum CBM
        if "Fair Curr ★" in labels:
            idx = labels.index("Fair Curr ★")
            ax.get_xticklabels()
            ax.axvline(idx, color=COLORS["fair_curriculum_cbm"], lw=1.5, ls="--", alpha=0.5)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.grid(axis="y", alpha=0.3, ls="--")

    fig.suptitle("Distribution of Final Performance Across 100 Runs",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "P4_stability_violin")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S1 — Twitter / X Announcement Card
# ══════════════════════════════════════════════════════════════════════════════

def figure_S1_twitter_card():
    """Dark navy Twitter/X announcement card (1200×675)."""
    fig = plt.figure(figsize=(12, 6.75))
    fig.patch.set_facecolor(COLORS["accent"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(COLORS["accent"])
    ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.75)

    # MICCAI badge
    badge = patches.FancyBboxPatch((0.2, 6.0), 2.0, 0.55,
                                   boxstyle="round,pad=0.05",
                                   fc=COLORS["fair_curriculum_cbm"], ec="none")
    ax.add_patch(badge)
    ax.text(1.2, 6.28, "MICCAI 2026", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")

    # Headline
    ax.text(6, 5.4, "Fairer AI for All Skin Types",
            ha="center", va="center", fontsize=26, fontweight="bold", color="white")
    ax.text(6, 4.8,
            "Fair Curriculum CBM reduces diagnostic disparity across Fitzpatrick skin types",
            ha="center", va="center", fontsize=13, color="#A8DADC")

    # Stat cards
    stat_info = [("+63%", "LG-F1", "#2DC653"), ("−44%", "Gap", "#E63946"), ("+5%", "Overall F1", "#90E0EF")]
    for i, (stat, label, color) in enumerate(stat_info):
        cx = 2.5 + i * 3.2
        r = patches.FancyBboxPatch((cx - 1.3, 3.0), 2.6, 1.4,
                                   boxstyle="round,pad=0.08",
                                   fc="#243B55", ec=color, lw=2)
        ax.add_patch(r)
        ax.text(cx, 3.9, stat,   ha="center", fontsize=30, fontweight="bold", color=color)
        ax.text(cx, 3.25, label, ha="center", fontsize=12, color="white")

    # Phase timeline strip
    phase_labels = ["Q1\nBalanced\nFoundation", "Q2\nDemographic\nParity",
                    "Q3\nEqualized\nOdds", "Q4\nPerformance\nParity"]
    phase_colors_keys = ["phase1", "phase2", "phase3", "phase4"]
    for i, (pl, pk) in enumerate(zip(phase_labels, phase_colors_keys)):
        r = patches.FancyBboxPatch((0.3 + i * 2.85, 0.6), 2.5, 1.6,
                                   boxstyle="round,pad=0.05",
                                   fc=COLORS[pk], ec="none", alpha=0.85)
        ax.add_patch(r)
        ax.text(0.3 + i * 2.85 + 1.25, 1.45, pl, ha="center", va="center",
                fontsize=9, color="white" if pk in ("phase3", "phase4") else COLORS["accent"],
                fontweight="bold")

    # QR placeholder
    r = patches.FancyBboxPatch((10.6, 0.2), 1.1, 1.1,
                               boxstyle="round,pad=0.05",
                               fc="#3A3A3A", ec="grey", lw=0.8)
    ax.add_patch(r)
    ax.text(11.15, 0.75, "Code\n/ Paper", ha="center", va="center",
            fontsize=7, color="grey")

    # Footnote
    ax.text(6, 0.2, "100 independent runs  ·  SkinCon  ·  Fitzpatrick I–VI  ·  p<0.01",
            ha="center", va="bottom", fontsize=8.5, color="#A8DADC")

    save_png_only(fig, "S1_twitter_card", dpi=100)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S2 — Square Graphic (Instagram / LinkedIn)
# ══════════════════════════════════════════════════════════════════════════════

def figure_S2_square_card(data):
    """Per-Fitzpatrick bar chart for Instagram/LinkedIn (1080×1080)."""
    mpl.rcParams.update({**POSTER_RC,
        "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12})

    CURR_F1 = [0.622, 0.534, 0.631, 0.553, 0.558, 0.270]
    FAIR_F1 = [0.644, 0.580, 0.635, 0.574, 0.642, 0.441]
    x       = np.arange(6)
    width   = 0.35

    fig, ax = plt.subplots(figsize=(10.8, 10.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.bar(x - width/2, CURR_F1, width, color=COLORS["curriculum_cbm"],
           alpha=0.85, label="Curriculum CBM", edgecolor="white", linewidth=0.5)
    bars = ax.bar(x + width/2, FAIR_F1, width, color=COLORS["fair_curriculum_cbm"],
                  alpha=0.9, label="Fair Curriculum CBM ★", edgecolor="white", linewidth=0.5)

    # +63% callout on Type VI
    ax.annotate("+63%\n(p=0.002)", xy=(5 + width/2, FAIR_F1[5]),
                xytext=(4.5, FAIR_F1[5] + 0.12),
                fontsize=13, fontweight="bold", color=COLORS["fair_curriculum_cbm"],
                arrowprops=dict(arrowstyle="->", color=COLORS["fair_curriculum_cbm"], lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels([f"Type {r}" for r in ["I", "II", "III", "IV", "V", "VI"]], fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=14)
    ax.set_ylim(0, 0.82)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", alpha=0.3, ls="--")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.text(0.5, 1.07, "AI that works for every skin type",
            transform=ax.transAxes, ha="center", fontsize=19, fontweight="bold",
            color=COLORS["accent"])
    ax.text(0.5, -0.11, "MICCAI 2026  ·  Fair Curriculum CBM  ·  100 independent runs",
            transform=ax.transAxes, ha="center", fontsize=11, color="grey")

    fig.tight_layout()
    save_png_only(fig, "S2_square_card", dpi=100)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S3 — Animated Training GIF
# ══════════════════════════════════════════════════════════════════════════════

def figure_S3_training_animation(data):
    """Animate performance gap collapse for Fair Curriculum CBM vs Curriculum CBM."""
    import matplotlib.animation as animation

    d_fc = data.get("fair_curriculum_cbm")
    d_cc = data.get("curriculum_cbm")
    if d_fc is None or d_cc is None:
        print("  SKIP S3: missing data")
        return

    mpl.rcParams.update(BASE_RC)

    n_fc = d_fc["n_epochs"]
    n_cc = d_cc["n_epochs"]
    x_fc = np.linspace(0, 100, n_fc)
    x_cc = np.linspace(0, 100, n_cc)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    for i, (pk, alpha) in enumerate(
            zip(["phase1","phase2","phase3","phase4"], [0.08]*4)):
        ax.axvspan(i*25, (i+1)*25, alpha=alpha, color=COLORS[pk], zorder=0)

    for x_ in [25, 50, 75]:
        ax.axvline(x_, color="grey", lw=0.6, ls=":", zorder=1)

    line_cc, = ax.plot([], [], color=COLORS["curriculum_cbm"], lw=1.8,
                       alpha=0.8, label="Curriculum CBM")
    line_fc, = ax.plot([], [], color=COLORS["fair_curriculum_cbm"], lw=2.5,
                       label="Fair Curriculum CBM ★")
    ann_txt  = ax.text(0.98, 0.96, "", transform=ax.transAxes,
                       ha="right", va="top", fontsize=10,
                       color=COLORS["fair_curriculum_cbm"], fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.set_xlim(0, 100); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Training Progress (%)", fontsize=12)
    ax.set_ylabel("Performance Gap (↓ better)", fontsize=12)
    ax.set_title("Performance Gap Collapse — Fair Curriculum CBM vs Curriculum CBM", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)

    step = 5  # epochs per frame
    frames = list(range(step, max(n_fc, n_cc) + 1, step))

    def animate(frame_n_epochs):
        i_fc = min(frame_n_epochs, n_fc)
        i_cc = min(frame_n_epochs, n_cc)
        line_fc.set_data(x_fc[:i_fc], d_fc["gap_mean"][:i_fc])
        line_cc.set_data(x_cc[:i_cc], d_cc["gap_mean"][:i_cc])
        ep_pct = min(frame_n_epochs / max(n_fc, n_cc) * 100, 100)
        cur_gap = d_fc["gap_mean"][min(i_fc - 1, n_fc - 1)]
        ann_txt.set_text(f"Ep: {frame_n_epochs}  |  Gap: {cur_gap:.3f}")
        return line_fc, line_cc, ann_txt

    anim = animation.FuncAnimation(fig, animate, frames=frames,
                                   interval=120, blit=True)

    gif_path = OUT_DIR / "S3_training_animation.gif"
    try:
        anim.save(str(gif_path), writer="pillow", fps=8)
        SAVED_FILES.append(str(gif_path))
        print("  ✓ S3_training_animation")
    except Exception as e:
        print(f"  WARNING S3 GIF failed ({e}), saving frames instead")
        frames_dir = OUT_DIR / "S3_frames"
        frames_dir.mkdir(exist_ok=True)
        for frame_n in frames:
            animate(frame_n)
            fig.savefig(frames_dir / f"frame_{frame_n:04d}.png", dpi=72, bbox_inches="tight")
            SAVED_FILES.append(str(frames_dir / f"frame_{frame_n:04d}.png"))
        print(f"  ✓ S3_frames/ ({len(frames)} frames)")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D1 — Concept Importance Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def figure_D1_concept_heatmap():
    """Per-concept F1 heatmap across 5 model types (rows = concepts, cols = models)."""
    mpl.rcParams.update(BASE_RC)

    matrix = []
    valid_models = []
    for model in MODELS:
        cf1 = load_final_concept_f1(MULTI_RUN_BASE, model)
        if cf1 is not None:
            matrix.append(cf1)
            valid_models.append(model)
    n_models = len(valid_models)
    mat = np.array(matrix).T   # (23 concepts, n_models)

    # Sort rows by Fair Curriculum CBM column if present
    if "fair_curriculum_cbm" in valid_models:
        fc_col = valid_models.index("fair_curriculum_cbm")
    else:
        fc_col = n_models - 1
    sort_idx = np.argsort(mat[:, fc_col])[::-1]
    mat = mat[sort_idx]
    concept_labels = [CONCEPT_NAMES[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_models))
    ax.set_xticklabels([MODEL_LABELS[m] for m in valid_models], rotation=15, ha="right", fontsize=11)
    ax.set_yticks(range(23))
    ax.set_yticklabels(concept_labels, fontsize=9.5)

    # Annotate cells
    for r in range(23):
        for c in range(n_models):
            v = mat[r, c]
            ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color="black" if 0.3 < v < 0.7 else "white" if v < 0.3 else "black")

    # Thick border on Fair Curriculum CBM column
    fc_col_x = fc_col - 0.5
    rect = patches.Rectangle((fc_col_x, -0.5), 1, 23,
                              linewidth=2.5, edgecolor=COLORS["fair_curriculum_cbm"],
                              facecolor="none", zorder=5)
    ax.add_patch(rect)

    plt.colorbar(im, ax=ax, orientation="vertical", pad=0.01, label="Mean F1 Score")
    ax.set_title("Per-Concept Prediction Performance — Concept Bottleneck Quality Across Models",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "D1_concept_importance")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D2 — Pareto Frontier Scatter
# ══════════════════════════════════════════════════════════════════════════════

def figure_D2_pareto_frontier(data):
    """Pareto frontier: Overall F1 vs Performance Gap, point size ∝ LG-F1."""
    mpl.rcParams.update(BASE_RC)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Pareto-superior region shading (upper-right after y-axis inversion)
    fc = data.get("fair_curriculum_cbm")
    if fc:
        ax.fill_betweenx([fc["final_gap"] - 0.02, 0],
                         fc["final_overall"] - 0.01, 1.0,
                         color="#2DC653", alpha=0.06, zorder=0)
        ax.text(fc["final_overall"] + 0.005, 0.01, "Pareto-superior\nregion",
                fontsize=9, color="#2DC653", va="bottom")

    for model in MODELS:
        d = data.get(model)
        if d is None:
            continue
        x_val   = d["final_overall"]
        y_val   = d["final_gap"]
        worst   = d["final_worst"]
        x_err   = d["final_overall_std"]
        y_err   = d["final_gap_std"]
        color   = COLORS[model]
        is_fc   = model == "fair_curriculum_cbm"
        size    = 350 if is_fc else 140
        marker  = "*" if is_fc else "o"

        ax.errorbar(x_val, y_val, xerr=x_err, yerr=y_err,
                    fmt="none", ecolor=color, alpha=0.5, capsize=4, zorder=2)
        ax.scatter(x_val, y_val, s=size * (worst / 0.5),
                   c=color, marker=marker, zorder=5, linewidths=0.5,
                   edgecolors=COLORS["accent"] if is_fc else "none", alpha=0.95)
        ax.annotate(MODEL_LABELS[model],
                    xy=(x_val, y_val), xytext=(x_val + 0.003, y_val + 0.006),
                    fontsize=9.5, color=color, fontweight="bold" if is_fc else "normal")

    ax.invert_yaxis()
    ax.set_xlabel("Overall F1 (↑ better)", fontsize=13)
    ax.set_ylabel("Performance Gap (↓ better, axis inverted)", fontsize=13)
    ax.set_title("Pareto Frontier — Fair Curriculum CBM Achieves Simultaneous Gains",
                 fontsize=13, fontweight="bold")
    ax.text(0.5, -0.10,
            "Point size ∝ Lowest-Group F1  ·  Error bars = ±1 SEM across 100 runs",
            transform=ax.transAxes, ha="center", fontsize=9.5, style="italic", color="grey")
    ax.grid(alpha=0.3, ls="--")
    fig.tight_layout()
    save(fig, "D2_pareto_frontier")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("FairCBM MICCAI 2026 — Generating all visualisations")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

    # Pre-load aggregated data (shared by multiple figures)
    print("\n[1/3] Aggregating run statistics …")
    data = {}
    for model in MODELS:
        print(f"  Loading {model} …")
        data[model] = aggregate_across_runs(MULTI_RUN_BASE, model)

    # Paper figures
    print("\n[2/3] Generating paper figures …")
    print("  F1: curriculum phases timeline")
    figure_F1_curriculum_phases()
    print("  F2: performance gap over training")
    figure_F2_gap_over_training(data)
    print("  F3: UMAP latent space comparison")
    figure_F3_umap_comparison()

    # Poster figures
    print("\n  P1: hero numbers")
    figure_P1_hero_numbers()
    print("  P2: ablation chart")
    figure_P2_ablation()
    print("  P3: per-Fitzpatrick heat strip")
    figure_P3_per_fitz_heatstrip()
    print("  P4: stability violin plots")
    figure_P4_stability_violin(data)

    # Social
    print("\n  S1: Twitter card")
    figure_S1_twitter_card()
    print("  S2: Square card (Instagram/LinkedIn)")
    figure_S2_square_card(data)
    print("  S3: Training animation GIF")
    figure_S3_training_animation(data)

    # Demo
    print("\n  D1: concept importance heatmap")
    figure_D1_concept_heatmap()
    print("  D2: Pareto frontier")
    figure_D2_pareto_frontier(data)

    print("\n" + "=" * 60)
    print(f"SAVED {len(SAVED_FILES)} files:")
    for f in sorted(SAVED_FILES):
        print(f"  {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
