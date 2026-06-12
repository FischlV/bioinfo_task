#!/usr/bin/env python3
"""Step 3: SHAP Analysis for XGBoost Multiclass Protein Subcellular Localization Model.

Uses shap.TreeExplainer (tree_path_dependent) on the k=2 DPC XGBoost model.
Requires: xgboost==2.0.3, shap==0.49.1.
"""

import os, sys, json, pickle, time, traceback
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb

# ============================================================
# Paths
# ============================================================
DATA_DIR = '/data2/hyh/yeast_promoter_project/dataworkspace/step1_features'
MODEL_DIR = '/data2/hyh/yeast_promoter_project/dataworkspace/step2_models'
OUT_DIR = '/data2/hyh/yeast_promoter_project/dataworkspace/step3_shap'
SHAP_DEP_DIR = os.path.join(OUT_DIR, 'shap_dependence')

CLASS_NAMES = ['Cytoplasm', 'Membrane_Secretory', 'Mitochondria', 'Nucleus', 'Other']
N_CLASSES = 5

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SHAP_DEP_DIR, exist_ok=True)

# ============================================================
# Logging
# ============================================================
class Tee:
    def __init__(self, path):
        self.file = open(path, 'w')
        self.stdout = sys.stdout
    def write(self, data):
        self.file.write(data); self.stdout.write(data)
    def flush(self):
        self.file.flush(); self.stdout.flush()

sys.stdout = sys.stderr = Tee(os.path.join(OUT_DIR, 'nohup.out'))

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ============================================================
# Main
# ============================================================
def main():
    t_start = time.time()
    log(f"Python: {sys.version.split()[0]}, xgb: {xgb.__version__}, shap: {shap.__version__}, np: {np.__version__}")

    # ---- 1. Load data ----
    log("Loading labels...")
    labels_df = pd.read_csv(os.path.join(DATA_DIR, 'labels.csv'))
    log(f"  shape={labels_df.shape}, cols={list(labels_df.columns)}")
    log(f"  distribution:\n{labels_df['location'].value_counts().to_string()}")

    log("Loading features (k=2 DPC, 400 features)...")
    features_df = pd.read_csv(os.path.join(DATA_DIR, 'features_fusion.csv'), index_col=0)
    features_df = features_df.iloc[:, 20:]  # skip first 20 AAC, keep 400 DPC
    feature_names = list(features_df.columns)
    log(f"  shape={features_df.shape}, first={feature_names[0]}, last={feature_names[-1]}")

    log("Loading train/test indices...")
    train_idx = np.load(os.path.join(DATA_DIR, 'train_idx.npy'))
    test_idx = np.load(os.path.join(DATA_DIR, 'test_idx.npy'))
    log(f"  train={len(train_idx)}, test={len(test_idx)}")

    log("Loading XGBoost model...")
    with open(os.path.join(MODEL_DIR, 'model_xgb.pkl'), 'rb') as f:
        model = pickle.load(f)
    log(f"  type={type(model).__name__}, objective={getattr(model,'objective','?')}, n_estimators={model.n_estimators}")
    log(f"  n_features_in_={model.n_features_in_}, n_classes_={model.n_classes_}")

    # ---- 2. Prepare test data ----
    X = features_df.values
    y = labels_df['location_encoded'].values.astype(int)
    X_test = X[test_idx]
    y_test = y[test_idx]
    log(f"  X_test={X_test.shape}")
    dist = {CLASS_NAMES[k]: v for k, v in sorted(dict(zip(*np.unique(y_test, return_counts=True))).items())}
    log(f"  y_test dist: {dist}")

    # ---- 3. SHAP TreeExplainer ----
    log("Creating TreeExplainer (tree_path_dependent)...")
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    ev = np.array(explainer.expected_value)
    log(f"  expected_value={ev}")

    log(f"Computing SHAP on {X_test.shape[0]} test samples...")
    t_shap = time.time()
    shap_values = explainer.shap_values(X_test)
    t_shap_done = time.time()
    log(f"  Done in {t_shap_done-t_shap:.1f}s")

    # Handle both list (booster) and ndarray (sklearn wrapper) formats
    if isinstance(shap_values, list):
        log(f"  shap_values: list of {len(shap_values)} arrays")
        for i, sv in enumerate(shap_values):
            log(f"    [{CLASS_NAMES[i]}] shape={sv.shape}, mean|SHAP|={np.mean(np.abs(sv)):.5f}")
        shap_array = np.stack(shap_values, axis=-1)
    else:
        log(f"  shap_values: ndarray shape={shap_values.shape}")
        shap_array = shap_values
        for ci in range(min(shap_array.shape[2], N_CLASSES)):
            log(f"    [{CLASS_NAMES[ci]}] mean|SHAP|={np.mean(np.abs(shap_array[:,:,ci])):.5f}")
    log(f"  final shape={shap_array.shape}")

    # ---- 4. Save full SHAP array ----
    shap_npy = os.path.join(OUT_DIR, 'shap_values_full.npy')
    np.save(shap_npy, shap_array)
    log(f"  Saved {shap_npy} ({os.path.getsize(shap_npy)/1024/1024:.1f} MB)")

    # ---- 5. Feature importance per class ----
    # mean_abs shape: (n_classes, n_features) = (5, 400)
    mean_abs = np.mean(np.abs(shap_array), axis=0).T

    top_json = {}
    for ci, cname in enumerate(CLASS_NAMES):
        scores = mean_abs[ci]
        top_idx = np.argsort(scores)[::-1][:10]
        top_json[cname] = [[feature_names[i], float(scores[i])] for i in top_idx]
        top5 = [(feature_names[i], f'{scores[i]:.4f}') for i in top_idx[:5]]
        log(f"  {cname} top5: {top5}")

    with open(os.path.join(OUT_DIR, 'shap_top_features_per_class.json'), 'w') as f:
        json.dump(top_json, f, indent=2)
    log("  Saved shap_top_features_per_class.json")

    # ---- 6. Per-class bar chart ----
    log("Plotting shap_per_class.png...")
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()
    for ci, cname in enumerate(CLASS_NAMES):
        ax = axes[ci]
        scores = mean_abs[ci]
        top_idx = np.argsort(scores)[::-1][:10]
        top_names = [feature_names[i] for i in top_idx]
        top_scores = scores[top_idx]
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, 10))
        ax.barh(range(9, -1, -1), top_scores[::-1], color=colors[::-1], edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(9, -1, -1))
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel('Mean |SHAP|', fontsize=11)
        ax.set_title(cname, fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    axes[-1].set_visible(False)
    fig.suptitle('Top 10 Features by Mean |SHAP| per Class', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'shap_per_class.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    log("  Saved shap_per_class.png")

    # ---- 7. Beeswarm ----
    log("Plotting shap_beeswarm.png...")
    overall_importance = np.sum(mean_abs, axis=0)
    best_class_idx = int(np.argmax(np.sum(mean_abs, axis=1)))
    log(f"  Best class: {CLASS_NAMES[best_class_idx]}")

    # Get SHAP for best class
    if isinstance(shap_values, list):
        sv_for_plot = shap_values[best_class_idx]
    else:
        sv_for_plot = shap_array[:, :, best_class_idx]

    shap.summary_plot(sv_for_plot, X_test, feature_names=feature_names,
                      max_display=20, show=False)
    fig = plt.gcf()
    plt.gca().set_title(f'SHAP Beeswarm — {CLASS_NAMES[best_class_idx]} (Top 20)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'shap_beeswarm.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    log("  Saved shap_beeswarm.png")

    # ---- 8. Feature type comparison ----
    log("Plotting shap_by_feature_type.png...")
    g1 = np.sum(mean_abs[:, :200], axis=1)
    g2 = np.sum(mean_abs[:, 200:], axis=1)

    fig, ax = plt.subplots(figsize=(10, 7))
    xp = np.arange(N_CLASSES); w = 0.35
    b1 = ax.bar(xp-w/2, g1, w, label='DPC group 1 (first 200)', color='#3498db', edgecolor='black', linewidth=0.5)
    b2 = ax.bar(xp+w/2, g2, w, label='DPC group 2 (last 200)', color='#e74c3c', edgecolor='black', linewidth=0.5)
    for bar, val in zip(b1, g1):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(b2, g2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Total |SHAP|', fontsize=12)
    ax.set_title('DPC Feature Contribution per Class', fontsize=14, fontweight='bold')
    ax.set_xticks(xp); ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
    for i, cn in enumerate(CLASS_NAMES):
        r = g2[i]/g1[i] if g1[i]>0 else float('inf')
        ax.annotate(f'G2/G1:{r:.2f}', xy=(i, max(g1[i],g2[i])*0.85), fontsize=9, ha='center', color='darkgreen', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'shap_by_feature_type.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    log("  Saved shap_by_feature_type.png")

    # ---- 9. Dependence plots ----
    log("Plotting shap_dependence/...")
    top5_idx = np.argsort(overall_importance)[::-1][:5]
    top5_names = [feature_names[i] for i in top5_idx]
    log(f"  Top 5: {top5_names}")

    for rank, fi in enumerate(top5_idx):
        fn = feature_names[fi]
        log(f"    [{rank+1}/5] {fn}")
        fig, ax = plt.subplots(figsize=(10, 7))
        try:
            shap.dependence_plot(fi, sv_for_plot, X_test, feature_names=feature_names, show=False, ax=ax)
            ax.set_title(f'SHAP Dependence — {fn} ({CLASS_NAMES[best_class_idx]})', fontsize=13, fontweight='bold')
            plt.tight_layout()
            fig.savefig(os.path.join(SHAP_DEP_DIR, f'dep_{rank+1:02d}_{fn}.png'), dpi=150, bbox_inches='tight')
        except Exception as e:
            log(f"    WARNING: {e}")
        plt.close(fig)
    log("  Done dependence plots")

    # ---- 10. Summary ----
    total_time = time.time() - t_start
    top3_pc = []
    for ci, cname in enumerate(CLASS_NAMES):
        scores = mean_abs[ci]
        top_idx = np.argsort(scores)[::-1][:3]
        top3_pc.append(f"  {cname}: " + ", ".join(f"{feature_names[i]}={scores[i]:.4f}" for i in top_idx))

    lines = [
        "="*70, "SHAP Analysis Summary", "="*70,
        f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Runtime: {total_time:.1f}s ({total_time/60:.1f} min)",
        "",
        f"Model: XGBoost multiclass ({N_CLASSES} classes), k=2 DPC (400 features)",
        f"Test samples: {X_test.shape[0]}, SHAP shape: {shap_array.shape}",
        f"Best class (total SHAP): {CLASS_NAMES[best_class_idx]}",
        "",
        "--- Top 5 Overall Features ---",
    ]
    for i, (idx, name) in enumerate(zip(top5_idx, top5_names)):
        lines.append(f"  {i+1}. {name}: {overall_importance[idx]:.4f}")

    lines += ["", "--- Per-Class Top 3 ---"] + top3_pc

    lines += ["", "--- DPC Group Analysis (G1=first 200, G2=last 200) ---"]
    for ci, cname in enumerate(CLASS_NAMES):
        r = g2[ci]/g1[ci] if g1[ci]>0 else float('inf')
        lines.append(f"  {cname}: G1={g1[ci]:.4f}, G2={g2[ci]:.4f}, G2/G1={r:.2f}")

    lines += ["", "--- Output Files ---"]
    for root, dirs, files in os.walk(OUT_DIR):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, OUT_DIR)
            sz = os.path.getsize(fp)
            lines.append(f"  {rel} ({sz/1024:.1f} KB)" if sz>=1024 else f"  {rel} ({sz} B)")

    lines += ["", "="*70, "END"]
    summary = "\n".join(lines)
    with open(os.path.join(OUT_DIR, 'summary.txt'), 'w') as f:
        f.write(summary)
    print(summary)

    log(f"ALL DONE! Total: {total_time:.1f}s")
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nFATAL: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
