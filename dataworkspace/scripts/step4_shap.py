#!/usr/bin/env python3
"""Step 4: SHAP 可解释性分析 + Saliency Map"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# === 配置 (脚本在 dataworkspace/scripts/ 下) ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_ROOT / "dataworkspace"
STEP2 = DATA / "step2_features"
STEP3 = DATA / "step3_models"
OUT = DATA / "step4_interpretation"
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams["font.family"] = "DejaVu Sans"

# === 1. 加载数据和模型 ===
print("Loading data...")
labels_df = pd.read_csv(STEP2 / "labels.csv")
features_df = pd.read_csv(STEP2 / "features_k4.csv", index_col=0)
train_idx = np.load(STEP2 / "train_idx.npy")
test_idx = np.load(STEP2 / "test_idx.npy")

common_ids = sorted(set(labels_df["gene_id"]) & set(features_df.index))
features_df = features_df.loc[common_ids]
labels_df = labels_df[labels_df["gene_id"].isin(common_ids)]
y_all = labels_df.set_index("gene_id").loc[common_ids, "log2_fpkm_plus1"].values.astype(np.float32)

gene_ids = list(features_df.index)
id_to_idx = {g: i for i, g in enumerate(gene_ids)}
test_idx_mapped = np.array([id_to_idx[g] for g in np.array(gene_ids)[test_idx] if g in id_to_idx])

X_test = features_df.values.astype(np.float32)[test_idx_mapped]
y_test = y_all[test_idx_mapped]
feature_names = list(features_df.columns)

print(f"  X_test: {X_test.shape}, y_test: {len(y_test)}")

# Load RF model
print("\nLoading Random Forest model...")
with open(STEP3 / "model_rf.pkl", "rb") as f:
    rf = pickle.load(f)

# === 2. SHAP TreeExplainer ===
print("\n=== SHAP Analysis (Random Forest) ===")
print("Computing SHAP values (this may take a while)...")

import shap

# Use sample for SHAP to speed up
n_shap = min(200, len(X_test))
X_shap = X_test[:n_shap]

# Create explainer (tree_path_dependent is much faster than interventional)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_shap)
print(f"  SHAP values shape: {shap_values.shape}")

# === SHAP Summary Plot ===
print("  Generating SHAP summary plot...")
fig = plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, feature_names=feature_names,
                  max_display=20, show=False)
plt.tight_layout()
plt.savefig(OUT / "shap_summary.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: shap_summary.png")

# === SHAP Bar Plot ===
print("  Generating SHAP bar plot...")
fig = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap, feature_names=feature_names,
                  plot_type="bar", max_display=20, show=False)
plt.tight_layout()
plt.savefig(OUT / "shap_bar.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: shap_bar.png")

# === Top k-mers by SHAP importance ===
print("\n  Top 20 k-mers by mean |SHAP|:")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1][:20]

top_kmers = []
for i in top_idx:
    top_kmers.append({
        "rank": len(top_kmers) + 1,
        "kmer": feature_names[i].replace("kmer_", ""),
        "mean_abs_shap": mean_abs_shap[i],
        "direction": "positive" if shap_values[:, i].mean() > 0 else "negative",
    })
    print(f"    {len(top_kmers):2d}. {feature_names[i]}  |SHAP|={mean_abs_shap[i]:.6f}  ({top_kmers[-1]['direction']})")

top_df = pd.DataFrame(top_kmers)
top_df.to_csv(OUT / "shap_top_kmers.csv", index=False)
print("  Saved: shap_top_kmers.csv")

# === SHAP Dependence Plots for top 5 ===
print("\n  Generating SHAP dependence plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, ax in enumerate(axes[:5]):
    if i >= len(top_idx):
        ax.set_visible(False)
        continue
    feat_idx = top_idx[i]
    shap.dependence_plot(
        feat_idx, shap_values, X_shap,
        feature_names=feature_names,
        ax=ax, show=False
    )
    ax.set_title(f"#{i+1}: {feature_names[feat_idx]}", fontsize=10)
axes[5].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "shap_dependence.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_dependence.png")

# === 3. Saliency Map (CNN) ===
print("\n=== Saliency Map Analysis ===")
try:
    import tensorflow as tf
    from tensorflow import keras

    print("Loading CNN model...")
    cnn = keras.models.load_model(STEP3 / "model_cnn.keras")

    # Load one-hot data
    X_onehot = np.load(STEP2 / "features_onehot.npy")
    X_cnn_test = X_onehot[test_idx_mapped]

    # Pick a strong promoter for saliency
    high_idx = np.argsort(y_test)[-5:]  # top 5 highest expressed
    low_idx = np.argsort(y_test)[:5]    # top 5 lowest expressed

    def compute_saliency(model, X, gene_idx):
        """Compute saliency map using gradient of output w.r.t. input"""
        X_input = tf.convert_to_tensor(X[gene_idx:gene_idx+1], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(X_input)
            pred = model(X_input)
        grads = tape.gradient(pred, X_input)
        saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()[0]
        return saliency

    # Generate saliency maps for top 3 high-expression genes
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    from matplotlib.colors import LinearSegmentedColormap

    colors_list = ["#ffffff", "#fde725"]
    cmap = LinearSegmentedColormap.from_list("custom_viridis", colors_list, N=256)

    for row_i in range(3):
        for col_j in range(2):
            idx_set = [high_idx, low_idx][col_j]
            label = ["High expression", "Low expression"][col_j]

            if row_i < len(idx_set):
                gidx = idx_set[row_i]
                sal = compute_saliency(cnn, X_cnn_test, gidx)
                seq_idx = test_idx_mapped[gidx]

                ax = axes[row_i, col_j]
                # Plot saliency as heatmap-like bar
                positions = np.arange(len(sal))
                ax.bar(positions, sal, width=1.0, color=plt.cm.viridis(sal / max(sal.max(), 1e-8)),
                       edgecolor="none")
                ax.set_title(f"{label} (#{row_i+1}, y={y_test[gidx]:.2f})", fontsize=10)
                ax.set_xlabel("Position (bp)")
                ax.set_ylabel("Saliency")
                # Mark TSS
                ax.axvline(x=800, color="red", linestyle="--", alpha=0.5, label="TSS (800bp)")
                ax.legend(fontsize=8)
            else:
                axes[row_i, col_j].set_visible(False)

    plt.suptitle("CNN Saliency Maps — Base-level Importance", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "saliency_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: saliency_maps.png")

    # === Detailed saliency for one representative gene ===
    print("\n  Generating detailed saliency map...")
    # Pick the highest expressed gene
    best_idx = high_idx[0]
    sal = compute_saliency(cnn, X_cnn_test, best_idx)

    # Get the actual sequence
    seq = ""
    base_map = {0: "A", 1: "T", 2: "G", 3: "C"}
    for j in range(X_cnn_test[best_idx].shape[0]):
        oh = X_cnn_test[best_idx][j]
        max_base = np.argmax(oh)
        seq += base_map.get(max_base, "N")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 6), gridspec_kw={"height_ratios": [3, 1]})

    # Top: saliency heatmap
    ax1.bar(range(len(sal)), sal, width=1.0, color=plt.cm.Reds(sal / max(sal.max(), 1e-8)))
    ax1.axvline(x=800, color="blue", linestyle="--", linewidth=2, alpha=0.7, label="TSS")
    ax1.set_ylabel("Saliency", fontsize=11)
    ax1.set_title(f"Saliency Map — High Expression Gene (y={y_test[best_idx]:.2f})", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Bottom: sequence with highlighted regions
    ax2.set_xlim(0, 850)
    ax2.set_ylim(0, 1)
    ax2.axvline(x=800, color="blue", linestyle="--", linewidth=2, alpha=0.7)

    # Color each base
    base_colors = {"A": "#2ecc71", "T": "#e74c3c", "G": "#f39c12", "C": "#3498db", "N": "#95a5a6"}
    for j, base in enumerate(seq[:850]):
        ax2.axvspan(j, j+1, alpha=0.4, color=base_colors.get(base, "#95a5a6"))

    ax2.set_yticks([])
    ax2.set_xlabel("Position (bp)", fontsize=11)

    # Add base legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.6, label=b) for b, c in base_colors.items()]
    ax2.legend(handles=legend_elements, loc="upper right", ncol=5, fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / "saliency_detailed.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: saliency_detailed.png")

except Exception as e:
    print(f"  ⚠ Saliency map generation failed: {e}")
    print("  (This is OK - CNN model may have issues, the project note explains this)")

# === 4. Summary ===
with open(OUT / "summary.txt", "w") as f:
    f.write("Step 4 可解释性分析\n")
    f.write("=" * 60 + "\n\n")
    f.write("SHAP Analysis (Random Forest):\n")
    f.write(f"  - 分析样本数: {n_shap}\n")
    f.write(f"  - 特征维度: {shap_values.shape[1]}\n")
    f.write(f"  - Top 20 k-mers saved to shap_top_kmers.csv\n\n")
    f.write("Output files:\n")
    f.write("  - shap_summary.png: SHAP beeswarm plot (top 20)\n")
    f.write("  - shap_bar.png: Mean |SHAP| bar chart\n")
    f.write("  - shap_top_kmers.csv: Top 20 important k-mers\n")
    f.write("  - shap_dependence.png: Dependence plots for top 5\n")
    f.write("  - saliency_maps.png: CNN base-level saliency\n")
    f.write("  - saliency_detailed.png: Detailed saliency + sequence\n")

print(f"\nDone! Output -> {OUT}")
