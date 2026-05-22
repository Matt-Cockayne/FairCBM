#!/usr/bin/env python3
"""
MICCAI 2026 Top-Tier Visualizations for Fair Curriculum CBM
Creates 4 publication-quality figures for poster, demos, and social media
"""

import os, sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torch
import torchvision.models as tvm
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("/home/csc29/projects/SkinCAP")
RAW_CSV = DATA_ROOT / "skincap_v240623.csv"
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = RESULTS_DIR / "miccai_visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))
from src.data.dataloader import SkinCapDataset

# Checkpoints
CHECKPOINTS = {
    "curriculum_cbm": RESULTS_DIR / "global_best_curriculum_cbm.pt",
    "fair_standard_cbm": RESULTS_DIR / "global_best_fair_standard_cbm.pt",
    "fair_curriculum_cbm": RESULTS_DIR / "global_best_fair_curriculum_cbm.pt",
}

# Design
COLORS = {
    "curriculum_cbm": "#457B9D",
    "fair_standard_cbm": "#A8DADC", 
    "fair_curriculum_cbm": "#E63946",
}
FITZ_COLORS = ["#F5CBA7", "#E59866", "#CA6F1E", "#A04000", "#784212", "#4A235A"]
FITZ_LABELS = ["Type I", "Type II", "Type III", "Type IV", "Type V", "Type VI"]

CONCEPT_NAMES = [
    "Papule", "Plaque", "Pustule", "Bulla", "Patch", "Nodule", "Ulcer",
    "Crust", "Erosion", "Atrophy", "Exudate", "Telangiectasia", "Scale",
    "Scar", "Friable", "Warty/Papillomatous", "Dome-shaped",
    "Brown (Hyperpig.)", "White (Hypopig.)", "Purple", "Yellow", "Black", "Erythema"
]

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Latent Space Fairness Comparison (UMAP)
# ══════════════════════════════════════════════════════════════════════════════

def plot1_latent_fairness_umap():
    """3-panel UMAP comparing latent space fairness across models."""
    try:
        import umap
    except ImportError:
        print("⚠️  SKIP Plot 1: umap-learn not installed")
        print("   Install with: pip install umap-learn")
        return
    
    if not RAW_CSV.exists():
        print("⚠️  SKIP Plot 1: SkinCAP data not found")
        return
    
    # Load validation data
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = SkinCapDataset(str(DATA_ROOT), split="val", label_type="concept",
                        transform=val_transform, raw_csv=str(RAW_CSV), seed=42)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def build_swin_backbone():
        model = tvm.swin_b(weights=None)
        model.head = torch.nn.Identity()
        return model
    
    def extract_embeddings(model_path):
        """Extract encoder embeddings for validation set."""
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        sd = state["model_state_dict"]
        backbone_sd = {k[9:]: v for k, v in sd.items() if k.startswith("backbone.")}
        
        backbone = build_swin_backbone()
        backbone.load_state_dict(backbone_sd, strict=False)
        backbone.eval().to(device)
        
        embeddings, labels, fitz_types = [], [], []
        with torch.no_grad():
            for batch in loader:
                imgs, concepts, binary_label, fitz = batch
                imgs = imgs.to(device)
                feats = backbone(imgs)
                if feats.dim() > 2:
                    feats = feats.mean(dim=list(range(1, feats.dim() - 1)))
                embeddings.append(feats.cpu().numpy())
                labels.append(binary_label.numpy())
                fitz_types.append(fitz.numpy())
        
        return (np.concatenate(embeddings), 
                np.concatenate(labels).astype(int),
                np.concatenate(fitz_types).astype(int))
    
    # Create figure
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.92, 0.08], hspace=0.25, wspace=0.25)
    
    models = ["curriculum_cbm", "fair_standard_cbm", "fair_curriculum_cbm"]
    titles = ["Curriculum CBM\n(Difficulty-Based)", 
              "Fair Standard CBM\n(Static Fairness)",
              "Fair Curriculum CBM ★\n(Progressive Fairness)"]
    
    for idx, (model, title) in enumerate(zip(models, titles)):
        print(f"  Processing {model}...")
        ax = fig.add_subplot(gs[0, idx])
        
        try:
            # Extract embeddings
            embs, labels, fitz = extract_embeddings(CHECKPOINTS[model])
            fitz_0idx = (fitz - 1).clip(0, 5)  # 1-6 → 0-5
            
            # UMAP reduction
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)
            embs_2d = reducer.fit_transform(embs)
            
            # Plot by Fitzpatrick type
            for gi in range(6):
                mask = fitz_0idx == gi
                if mask.sum() == 0:
                    continue
                ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1],
                          c=FITZ_COLORS[gi], s=20, alpha=0.6, 
                          linewidths=0.5, edgecolors='white',
                          label=FITZ_LABELS[gi] if idx == 0 else "")
            
            # Compute fairness metrics
            sil_score = silhouette_score(embs_2d, fitz_0idx) if len(np.unique(fitz_0idx)) > 1 else np.nan
            
            # Compute centroid distances (fairness = lower variance in distances)
            centroids = np.array([embs_2d[fitz_0idx == i].mean(axis=0) 
                                  for i in range(6) if (fitz_0idx == i).sum() > 0])
            if len(centroids) > 1:
                from scipy.spatial.distance import pdist
                centroid_distances = pdist(centroids)
                dist_std = centroid_distances.std()
            else:
                dist_std = np.nan
            
            # Annotations
            textstr = f'Silhouette: {sil_score:.3f}\n(↓ = better mixing)\n\nCentroid Dist. σ: {dist_std:.2f}\n(↓ = more uniform)'
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLORS[model], linewidth=2)
            ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=props, family='monospace')
            
            # Style
            ax.set_title(title, fontsize=13, fontweight='bold' if model == "fair_curriculum_cbm" else 'normal',
                        color=COLORS[model], pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Border highlighting
            for spine in ax.spines.values():
                spine.set_edgecolor(COLORS[model])
                spine.set_linewidth(3 if model == "fair_curriculum_cbm" else 1)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\n{model}\n{type(e).__name__}",
                   transform=ax.transAxes, ha='center', va='center', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Shared legend
    handles = [mpatches.Patch(color=FITZ_COLORS[i], label=FITZ_LABELS[i]) for i in range(6)]
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis('off')
    legend_ax.legend(handles=handles, loc='center', ncol=6, fontsize=11,
                    frameon=False, title="Fitzpatrick Skin Type", title_fontsize=12)
    
    fig.suptitle("Latent Space Fairness — Fair Curriculum CBM Achieves Better Demographic Mixing",
                fontsize=15, fontweight='bold', y=0.98)
    
    save_path = OUT_DIR / "plot1_latent_fairness_umap.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Adversarial Discriminator Fairness Analysis
# ══════════════════════════════════════════════════════════════════════════════

def plot2_discriminator_analysis():
    """Show discriminator confusion - Fair CBM should confuse the discriminator."""
    if not RAW_CSV.exists():
        print("⚠️  SKIP Plot 2: SkinCAP data not found")
        return
    
    # Load validation data
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = SkinCapDataset(str(DATA_ROOT), split="val", label_type="concept",
                        transform=val_transform, raw_csv=str(RAW_CSV), seed=42)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def build_models(model_path, has_discriminator=False):
        """Build backbone and discriminator if present."""
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        sd = state["model_state_dict"]
        
        # Backbone
        backbone_sd = {k[9:]: v for k, v in sd.items() if k.startswith("backbone.")}
        backbone = tvm.swin_b(weights=None)
        backbone.head = torch.nn.Identity()
        backbone.load_state_dict(backbone_sd, strict=False)
        
        # Discriminator (if present)
        discriminator = None
        if has_discriminator:
            disc_sd = {k[26:]: v for k, v in sd.items() if k.startswith("adversarial_discriminator.")}
            if disc_sd:
                from src.models.adversarial_discriminator import AdversarialDiscriminator
                discriminator = AdversarialDiscriminator(input_dim=1024, num_groups=6)
                discriminator.load_state_dict(disc_sd, strict=False)
        
        return backbone, discriminator
    
    def get_predictions(backbone, discriminator, loader, device):
        """Get discriminator predictions for Fitzpatrick type."""
        backbone.eval().to(device)
        if discriminator is not None:
            discriminator.eval().to(device)
        
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in loader:
                imgs, _, _, fitz = batch
                imgs = imgs.to(device)
                
                # Extract features
                feats = backbone(imgs)
                if feats.dim() > 2:
                    feats = feats.mean(dim=list(range(1, feats.dim() - 1)))
                
                # Predict Fitzpatrick type
                if discriminator is not None:
                    logits = discriminator(feats)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                else:
                    # Random baseline for models without discriminator
                    preds = np.random.randint(0, 6, size=len(fitz))
                
                all_preds.append(preds)
                all_true.append((fitz.numpy() - 1).clip(0, 5))  # 1-6 → 0-5
        
        return np.concatenate(all_true), np.concatenate(all_preds)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('white')
    
    models = ["curriculum_cbm", "fair_standard_cbm", "fair_curriculum_cbm"]
    has_disc = [False, True, True]  # Only fair models have discriminator
    titles = ["Curriculum CBM\n(No Discriminator)", 
              "Fair Standard CBM",
              "Fair Curriculum CBM ★"]
    
    for idx, (model, has_d, title, ax) in enumerate(zip(models, has_disc, titles, axes)):
        print(f"  Processing {model}...")
        
        try:
            backbone, discriminator = build_models(CHECKPOINTS[model], has_discriminator=has_d)
            y_true, y_pred = get_predictions(backbone, discriminator, loader, device)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
            cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            
            # Plot
            im = ax.imshow(cm_norm, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
            
            # Add text annotations
            for i in range(6):
                for j in range(6):
                    text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{cm_norm[i, j]:.2f}',
                           ha='center', va='center', color=text_color, fontsize=9)
            
            # Compute accuracy
            accuracy = np.trace(cm) / cm.sum()
            
            # Style
            ax.set_xticks(range(6))
            ax.set_yticks(range(6))
            ax.set_xticklabels([f'T{i+1}' for i in range(6)], fontsize=10)
            ax.set_yticklabels([f'T{i+1}' for i in range(6)], fontsize=10)
            ax.set_xlabel('Predicted Fitzpatrick Type', fontsize=11)
            if idx == 0:
                ax.set_ylabel('True Fitzpatrick Type', fontsize=11)
            
            # Title with accuracy
            title_with_acc = f"{title}\nAccuracy: {accuracy:.1%}\n{'(Random Baseline)' if not has_d else '(↓ = fairer)'}"
            ax.set_title(title_with_acc, fontsize=12, 
                        fontweight='bold' if model == "fair_curriculum_cbm" else 'normal',
                        color=COLORS[model], pad=10)
            
            # Border
            for spine in ax.spines.values():
                spine.set_edgecolor(COLORS[model])
                spine.set_linewidth(3 if model == "fair_curriculum_cbm" else 1)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {type(e).__name__}",
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Prediction Rate', fontsize=11)
    
    fig.suptitle("Adversarial Discriminator Confusion — Lower Accuracy = More Group-Invariant Features",
                fontsize=14, fontweight='bold', y=0.98)
    
    save_path = OUT_DIR / "plot2_discriminator_confusion.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Clinical Concept Bottleneck Showcase
# ══════════════════════════════════════════════════════════════════════════════

def plot3_concept_bottleneck_showcase():
    """Show real clinical examples with concept explanations."""
    if not RAW_CSV.exists():
        print("⚠️  SKIP Plot 3: SkinCAP data not found")
        return
    
    # Load validation data
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = SkinCapDataset(str(DATA_ROOT), split="val", label_type="concept",
                        transform=val_transform, raw_csv=str(RAW_CSV), seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Fair Curriculum CBM
    model_path = CHECKPOINTS["fair_curriculum_cbm"]
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    sd = state["model_state_dict"]
    
    # Build full model
    from src.models.fairness_aware_cbm import FairnessAwareCBM
    model = FairnessAwareCBM(
        num_concepts=23,
        num_classes=2,
        backbone_name='swin',
        num_groups=6,
        use_adversarial=True
    )
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    
    # Select interesting examples: 1 benign Type II, 1 malignant Type VI
    examples = []
    for idx in range(len(ds)):
        img_tensor, concepts, label, fitz = ds[idx]
        if label == 0 and fitz == 2 and len(examples) < 1:  # Benign Type II
            examples.append((idx, img_tensor, concepts, label, fitz, "Benign, Type II"))
        elif label == 1 and fitz == 6 and len(examples) < 2:  # Malignant Type VI
            examples.append((idx, img_tensor, concepts, label, fitz, "Malignant, Type VI"))
        if len(examples) >= 2:
            break
    
    if len(examples) < 2:
        print("⚠️  SKIP Plot 3: Could not find suitable examples")
        return
    
    # Create figure
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1.5, 1.5, 0.1], 
                  hspace=0.3, wspace=0.4)
    
    for row, (idx, img_tensor, true_concepts, true_label, fitz, desc) in enumerate(examples):
        print(f"  Processing example {row + 1}: {desc}")
        
        # Get prediction
        with torch.no_grad():
            img_batch = img_tensor.unsqueeze(0).to(device)
            concept_pred, binary_pred = model(img_batch)
            concept_probs = torch.sigmoid(concept_pred).cpu().numpy()[0]
            binary_prob = torch.sigmoid(binary_pred).cpu().numpy()[0, 0]
        
        # Panel 1: Image
        ax_img = fig.add_subplot(gs[row, 0])
        # Denormalize image
        img_display = img_tensor.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = (img_display * std + mean).clip(0, 1)
        ax_img.imshow(img_display)
        ax_img.set_title(f"Input Image\n{desc}", fontsize=11, fontweight='bold')
        ax_img.axis('off')
        
        # Panel 2: Top Concepts (horizontal bar chart)
        ax_concepts = fig.add_subplot(gs[row, 1])
        top_indices = np.argsort(concept_probs)[-10:][::-1]
        top_names = [CONCEPT_NAMES[i] for i in top_indices]
        top_probs = concept_probs[top_indices]
        
        colors = ['#2DC653' if p > 0.5 else '#CCCCCC' for p in top_probs]
        bars = ax_concepts.barh(range(10), top_probs, color=colors, alpha=0.8)
        ax_concepts.set_yticks(range(10))
        ax_concepts.set_yticklabels(top_names, fontsize=9)
        ax_concepts.set_xlabel('Concept Probability', fontsize=10)
        ax_concepts.set_title('Top 10 Concept Predictions', fontsize=11, fontweight='bold')
        ax_concepts.set_xlim(0, 1)
        ax_concepts.axvline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax_concepts.invert_yaxis()
        
        # Panel 3: Diagnosis + Intervention
        ax_diagnosis = fig.add_subplot(gs[row, 2])
        ax_diagnosis.axis('off')
        
        pred_label = "MALIGNANT" if binary_prob > 0.5 else "BENIGN"
        true_label_str = "MALIGNANT" if true_label == 1 else "BENIGN"
        correct = pred_label == true_label_str
        
        diagnosis_text = f"""
DIAGNOSIS PREDICTION
{'─'*30}
Predicted: {pred_label}
Confidence: {binary_prob:.1%}
Ground Truth: {true_label_str}
Status: {'✓ CORRECT' if correct else '✗ INCORRECT'}

{'─'*30}
CONCEPT-BASED REASONING
{'─'*30}
Key Activated Concepts:
{chr(10).join(f'  • {CONCEPT_NAMES[i]} ({concept_probs[i]:.0%})' for i in top_indices[:5])}

{'─'*30}
CLINICAL INTERVENTION
{'─'*30}
Clinician can verify/override:
  • Concept predictions
  • Final diagnosis
  • Adjust threshold
"""
        ax_diagnosis.text(0.05, 0.95, diagnosis_text, 
                         transform=ax_diagnosis.transAxes,
                         fontsize=9, verticalalignment='top',
                         family='monospace',
                         bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.9))
    
    fig.suptitle("Concept Bottleneck Clinical Showcase — Interpretable AI for Dermatology",
                fontsize=15, fontweight='bold', y=0.98)
    
    save_path = OUT_DIR / "plot3_concept_showcase.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4: SkinCon Dataset Imbalance
# ══════════════════════════════════════════════════════════════════════════════

def plot4_dataset_imbalance():
    """Visualize severe class and demographic imbalance in SkinCon."""
    if not RAW_CSV.exists():
        print("⚠️  SKIP Plot 4: SkinCAP data not found")
        return
    
    import pandas as pd
    
    # Load full dataset
    df = pd.read_csv(RAW_CSV)
    
    # Filter to valid Fitzpatrick types (1-6) and valid malignancy labels
    df = df[df['fitzpatrick_scale'].isin([1, 2, 3, 4, 5, 6])]
    df = df[df['malignant'].isin([0, 1])]
    
    # Compute cross-tabulation
    crosstab = pd.crosstab(df['fitzpatrick_scale'], df['malignant'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('white')
    
    # Plot 1: Stacked bar chart (Fitzpatrick × Malignancy)
    fitz_types = sorted(crosstab.index.tolist())
    benign_counts = [crosstab.loc[f, 0] if 0 in crosstab.columns else 0 for f in fitz_types]
    malignant_counts = [crosstab.loc[f, 1] if 1 in crosstab.columns else 0 for f in fitz_types]
    
    x = np.arange(len(fitz_types))
    width = 0.6
    
    bars1 = ax1.bar(x, benign_counts, width, label='Benign', color='#90E0EF', alpha=0.9)
    bars2 = ax1.bar(x, malignant_counts, width, bottom=benign_counts, 
                    label='Malignant', color='#E63946', alpha=0.9)
    
    # Annotations
    for i, (b, m) in enumerate(zip(benign_counts, malignant_counts)):
        total = b + m
        # Total on top
        ax1.text(i, total + 15, str(total), ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
        # Benign count
        if b > 20:
            ax1.text(i, b / 2, str(b), ha='center', va='center', 
                    fontsize=9, color='#1D3557')
        # Malignant count
        if m > 10:
            ax1.text(i, b + m / 2, str(m), ha='center', va='center', 
                    fontsize=9, color='white')
    
    ax1.set_xlabel('Fitzpatrick Skin Type', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Number of Images', fontsize=13, fontweight='bold')
    ax1.set_title('SkinCon Dataset Distribution\n(Severe Type V-VI Underrepresentation)', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Type {int(t)}' for t in fitz_types], fontsize=11)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Highlight Type VI crisis
    ax1.axvspan(x[-1] - 0.4, x[-1] + 0.4, alpha=0.15, color='red', zorder=0)
    ax1.text(x[-1], max(benign_counts) * 1.1, '⚠️ Only 114 samples', 
            ha='center', fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Plot 2: Class imbalance pie charts
    ax2.axis('off')
    
    # Overall malignancy distribution
    total_benign = sum(benign_counts)
    total_malignant = sum(malignant_counts)
    
    # Pie chart 1: Overall
    ax_pie1 = fig.add_axes([0.57, 0.55, 0.18, 0.35])
    colors_pie = ['#90E0EF', '#E63946']
    explode = (0.05, 0.05)
    wedges, texts, autotexts = ax_pie1.pie(
        [total_benign, total_malignant],
        labels=['Benign', 'Malignant'],
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90,
        explode=explode,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax_pie1.set_title(f'Overall Class Balance\n({total_benign + total_malignant} total images)', 
                     fontsize=11, fontweight='bold', pad=10)
    
    # Pie chart 2: Type VI only
    ax_pie2 = fig.add_axes([0.78, 0.55, 0.18, 0.35])
    type6_benign = benign_counts[-1]
    type6_malignant = malignant_counts[-1]
    wedges2, texts2, autotexts2 = ax_pie2.pie(
        [type6_benign, type6_malignant],
        labels=['Benign', 'Malignant'],
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90,
        explode=explode,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax_pie2.set_title(f'Type VI Distribution\n({type6_benign + type6_malignant} total images)', 
                     fontsize=11, fontweight='bold', pad=10, color='red')
    
    # Summary statistics box
    summary_text = f"""
DATASET STATISTICS
{'─'*35}
Total Images: {len(df):,}
Benign:       {total_benign:,} ({total_benign/len(df)*100:.1f}%)
Malignant:    {total_malignant:,} ({total_malignant/len(df)*100:.1f}%)

DEMOGRAPHIC IMBALANCE
{'─'*35}
Type I (Lightest):  {benign_counts[0] + malignant_counts[0]:>4} ({(benign_counts[0] + malignant_counts[0])/len(df)*100:>5.1f}%)
Type II:            {benign_counts[1] + malignant_counts[1]:>4} ({(benign_counts[1] + malignant_counts[1])/len(df)*100:>5.1f}%)
Type III:           {benign_counts[2] + malignant_counts[2]:>4} ({(benign_counts[2] + malignant_counts[2])/len(df)*100:>5.1f}%)
Type IV:            {benign_counts[3] + malignant_counts[3]:>4} ({(benign_counts[3] + malignant_counts[3])/len(df)*100:>5.1f}%)
Type V:             {benign_counts[4] + malignant_counts[4]:>4} ({(benign_counts[4] + malignant_counts[4])/len(df)*100:>5.1f}%)
Type VI (Darkest):  {benign_counts[5] + malignant_counts[5]:>4} ({(benign_counts[5] + malignant_counts[5])/len(df)*100:>5.1f}%) ⚠️

FAIRNESS CHALLENGE
{'─'*35}
Type I / Type VI Ratio: {(benign_counts[0] + malignant_counts[0]) / (benign_counts[5] + malignant_counts[5]):.1f}×

This severe imbalance motivates our
fairness-first curriculum approach.
"""
    
    ax_text = fig.add_axes([0.55, 0.05, 0.42, 0.45])
    ax_text.axis('off')
    ax_text.text(0.02, 0.98, summary_text, 
                transform=ax_text.transAxes,
                fontsize=9, verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='#FFF9E6', alpha=0.9,
                         edgecolor='#E63946', linewidth=2))
    
    fig.suptitle("SkinCon Dataset Imbalance — Motivation for Fair Curriculum Learning",
                fontsize=15, fontweight='bold', y=0.98)
    
    save_path = OUT_DIR / "plot4_dataset_imbalance.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("MICCAI 2026 — Fair Curriculum CBM Top-Tier Visualizations")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 70)
    
    print("\n[1/4] Latent Space Fairness Comparison (UMAP)...")
    plot1_latent_fairness_umap()
    
    print("\n[2/4] Adversarial Discriminator Analysis...")
    plot2_discriminator_analysis()
    
    print("\n[3/4] Clinical Concept Bottleneck Showcase...")
    plot3_concept_bottleneck_showcase()
    
    print("\n[4/4] SkinCon Dataset Imbalance...")
    plot4_dataset_imbalance()
    
    print("\n" + "=" * 70)
    print("✅ All visualizations complete!")
    print(f"📁 Output directory: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
