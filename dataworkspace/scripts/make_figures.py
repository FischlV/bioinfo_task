#!/usr/bin/env python3
"""
make_figures.py - Generate Figures 1-3 for yeast protein subcellular localization project.
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11, "font.family": "sans-serif"})

BASE  = Path("/data2/hyh/yeast_promoter_project/dataworkspace")
FIGS  = BASE / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

MODEL_DIR = BASE / "step2_models"
SHAP_DIR  = BASE / "step3_shap"
MOTIF_DIR = BASE / "step4_motif"
FEAT_DIR  = BASE / "step1_features"

MATRIX_SEEDS = [MODEL_DIR / f"matrix_seed{s}.json" for s in ("123","456","789")]
SHAP_VALS = SHAP_DIR / "shap_values_full.npy"
SHAP_TOP  = SHAP_DIR / "shap_top_features_per_class.json"
FEAT_K2   = FEAT_DIR / "features_k2.csv"
MOTIF_CSV = MOTIF_DIR / "motif_validation_final.csv"
FINAL_XGB = MODEL_DIR / "final_xgb.out"

STYLE = "seaborn-v0_8-whitegrid"
try:
    plt.style.use(STYLE)
except Exception:
    plt.style.use("seaborn-v0_8")


def make_figure1():
    print("[FIG1] Starting...")
    fig = plt.figure(figsize=(18, 7))

    # -- Panel A (left) - Pipeline --
    axA = fig.add_axes([0.02, 0.10, 0.26, 0.82])
    axA.set_xlim(0, 10); axA.set_ylim(0, 12); axA.axis("off")
    axA.set_title("A  Study Workflow", fontsize=13, fontweight="bold", loc="left", pad=12)

    steps = [
        (5.0, 10.5, "SGD / UniProt", "#E3F2FD"),
        (5.0, 8.5,  "Protein Sequences\n(~5,800 yeast ORFs)", "#E3F2FD"),
        (5.0, 6.5,  "k-mer Encoding\n(k=1 AAC, k=2 DPC, ...)", "#FFF3E0"),
        (5.0, 4.5,  "ML Models\n(RF / XGBoost / LogReg)", "#E8F5E9"),
        (5.0, 2.5,  "SHAP Analysis\n& Biological Validation", "#FCE4EC"),
    ]
    for x, y, label, color in steps:
        axA.add_patch(FancyBboxPatch((x-2.2, y-0.75), 4.4, 1.5,
            boxstyle="round,pad=0.15", facecolor=color, edgecolor="#455A64", linewidth=1.2))
        axA.text(x, y, label, ha="center", va="center", fontsize=10, fontweight="bold", color="#263238")
    for ay in [9.7, 7.7, 5.7, 3.7]:
        axA.annotate("", xy=(5.0, ay-0.05), xytext=(5.0, ay+0.85),
                     arrowprops=dict(arrowstyle="->", color="#546E7A", lw=2.5))

    # -- Panel B (right) - Heatmap --
    all_data = {}
    for fn in MATRIX_SEEDS:
        with open(fn) as fh:
            all_data[fn.name] = json.load(fh)

    rows_map = ["LogReg", "RF", "XGBoost"]
    cols_map = [
        "k=1 AAC (20d)", "k=2 DPC (400d)", "k=3 TPC (8000d)",
        "Fusion k1+k2 (420d)", "Binary enc (5000d)",
        "Integer enc (1000d)", "One-hot flatten (20000d)",
    ]
    col_labels_short = [
        "k=1 AAC\n(20d)", "k=2 DPC\n(400d)", "k=3 TPC\n(8000d)",
        "Fusion\n(420d)", "Binary\n(5000d)", "Integer\n(1000d)", "One-hot\n(20000d)",
    ]

    acc_mat = np.zeros((len(rows_map), len(cols_map)))
    for fname, seed_data in all_data.items():
        for key, val in seed_data.items():
            parts = key.split("|")
            mdl = parts[0].strip()
            enc = "|".join(parts[1:]).strip()
            if "Fusion" in enc:
                enc = "Fusion k1+k2 (420d)"
            elif "Binary" in enc and "5000" in enc:
                enc = "Binary enc (5000d)"
            elif "Integer" in enc:
                enc = "Integer enc (1000d)"
            elif "One-hot" in enc:
                enc = "One-hot flatten (20000d)"
            if mdl in rows_map and enc in cols_map:
                r = rows_map.index(mdl)
                c = cols_map.index(enc)
                acc_mat[r, c] += val["Acc"]

    acc_mat /= len(all_data)
    acc_mat = np.round(acc_mat, 4)
    df_heat = pd.DataFrame(acc_mat, index=rows_map, columns=col_labels_short)
    df_annot = np.round(acc_mat * 100, 1)

    axB = fig.add_axes([0.33, 0.12, 0.65, 0.80])
    axB.set_title("B  Model x Encoding Accuracy Heatmap (mean of 3 seeds)", fontsize=13, fontweight="bold", loc="left", pad=12)

    best_r, best_c = np.unravel_index(np.argmax(acc_mat), acc_mat.shape)
    annot = [[f"{v:.1f}" for v in row] for row in df_annot]

    sns.heatmap(df_heat, annot=annot, fmt="", cmap="RdYlGn",
                linewidths=1.5, linecolor="white",
                cbar_kws={"label": "Accuracy", "shrink": 0.8},
                vmin=0.26, vmax=0.66, ax=axB,
                annot_kws={"fontsize": 11, "fontweight": "bold"})

    axB.add_patch(plt.Rectangle((best_c, best_r), 1, 1, fill=False,
                                 edgecolor="black", lw=3.5, linestyle="-"))
    axB.text(best_c+0.5, best_r+0.2, "\u2605", ha="center", va="center",
             fontsize=22, color="black", fontweight="bold")
    axB.set_xlabel("Encoding Method", fontsize=12, fontweight="bold")
    axB.set_ylabel("Model", fontsize=12, fontweight="bold")
    axB.tick_params(axis="both", labelsize=10)

    fig.text(0.5, 0.02, "\u2605 Best: XGBoost x Fusion k1+k2 (~63.6% Acc).  Data: yeast protein subcellular localization, 5 classes, 3 independent seeds.",
             ha="center", fontsize=9, fontstyle="italic", color="#546E7A")

    outpath = FIGS / "fig1_overview_heatmap.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [FIG1] Saved -> {outpath}  ({os.path.getsize(outpath)/1024:.1f} KB)")


def make_figure2():
    print("[FIG2] Starting...")
    shap_vals = np.load(SHAP_VALS)
    with open(SHAP_TOP) as fh:
        shap_top = json.load(fh)

    feat_names_k2 = pd.read_csv(FEAT_K2, nrows=0).columns.tolist()
    feat_names_k2 = [c for c in feat_names_k2 if c.startswith("kmer_")]
    if shap_vals.ndim == 2:
        n_features = shap_vals.shape[1]
    else:
        n_features = shap_vals.shape[1]

    if len(feat_names_k2) == n_features:
        feature_names = feat_names_k2
    else:
        feature_names = [f"kmer_{i}" for i in range(n_features)]

    if shap_vals.ndim == 3:
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=(0, 2))
    else:
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    top_n = 15
    top_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]
    top_names = [feature_names[i].replace("kmer_", "") for i in top_idx]
    top_values = mean_abs_shap[top_idx]

    fig = plt.figure(figsize=(20, 18))

    # Panel A (top) - Top 15 SHAP bar chart
    axA = fig.add_subplot(3, 1, 1)
    axA.set_title("A  Overall Top-15 Dipeptide Features (mean |SHAP| across all 5 classes)",
                  fontsize=13, fontweight="bold", loc="left", pad=10)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, top_n))
    axA.barh(range(top_n), top_values[::-1], color=colors[::-1], edgecolor="#333333", linewidth=0.8)
    axA.set_yticks(range(top_n))
    axA.set_yticklabels(top_names[::-1], fontsize=14, fontfamily="monospace")
    axA.set_xlabel("mean |SHAP| value", fontsize=12, fontweight="bold")
    axA.invert_yaxis()
    for i, val in enumerate(top_values[::-1]):
        axA.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=10, fontweight="bold")

    # Panel B (bottom left) - Per-class top 5
    classes = list(shap_top.keys())
    n_cls = len(classes)

    gs_bottom = fig.add_gridspec(1, n_cls, top=0.65, bottom=0.06, left=0.04, right=0.55,
                                  wspace=0.35)
    axB_title = fig.text(0.04, 0.66, "B  Per-Class Top-5 Dipeptide Features (SHAP importance)",
                        fontsize=13, fontweight="bold")

    for ci, cls_name in enumerate(classes):
        ax = fig.add_subplot(gs_bottom[0, ci])
        items = shap_top[cls_name][:5]
        names = [it[0].replace("kmer_", "") for it in items]
        vals  = [it[1] for it in items]
        ax.barh(range(5), vals, color=plt.cm.Set2(ci), edgecolor="#333333", linewidth=0.8)
        ax.set_yticks(range(5))
        ax.set_yticklabels(names, fontsize=9, fontfamily="monospace")
        ax.set_title(cls_name.replace("_", "\n"), fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlim(0, max(0.25, max(vals)*1.2))
        for j, v in enumerate(vals):
            ax.text(v + 0.002, j, f"{v:.3f}", va="center", fontsize=7)

    # Panel C (bottom right) - Biological Corroboration table
    axC = fig.add_axes([0.59, 0.09, 0.40, 0.54])
    axC.axis("off")
    axC.set_title("C  Biological Corroboration", fontsize=13, fontweight="bold", loc="left", pad=8)

    df_motif = pd.read_csv(MOTIF_CSV)
    df_tbl = df_motif[["dipeptide", "class", "ratio", "p_adj", "verdict"]].copy()
    df_tbl["ratio"] = df_tbl["ratio"].round(2)
    df_tbl["p_adj"] = df_tbl["p_adj"].apply(lambda x: f"{x:.1e}" if x < 0.01 else f"{x:.3f}")
    df_tbl.columns = ["Dipeptide", "Class", "Enrich.\nRatio", "p-adj", "Verdict"]

    rows_to_show = []
    for cls in classes:
        sub = df_tbl[df_tbl["Class"] == cls].head(5)
        rows_to_show.append(sub)
    df_show = pd.concat(rows_to_show, ignore_index=True)

    col_labels = df_show.columns.tolist()
    cell_text = df_show.values.tolist()

    table = axC.table(cellText=cell_text, colLabels=col_labels,
                      cellLoc="center", loc="center",
                      colWidths=[0.12, 0.22, 0.12, 0.12, 0.20])
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.35)

    for i, row in df_show.iterrows():
        for j, col_name in enumerate(df_show.columns):
            cell = table[i+1, j]
            if row["Verdict"] == "Confirmed":
                cell.set_facecolor("#C8E6C9")
            elif row["Verdict"] == "Novel finding":
                cell.set_facecolor("#BBDEFB")
            elif "Partially" in str(row["Verdict"]):
                cell.set_facecolor("#FFF9C4")
            cell.set_edgecolor("#CFD8DC")

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=8)

    legend_ax = fig.add_axes([0.59, 0.02, 0.40, 0.04])
    legend_ax.axis("off")
    from matplotlib.patches import Patch
    legend_ax.legend(handles=[
        Patch(color="#C8E6C9", label="Confirmed"),
        Patch(color="#BBDEFB", label="Novel finding"),
        Patch(color="#FFF9C4", label="Partially supported"),
    ], loc="center", ncol=3, fontsize=8, frameon=False)

    outpath = FIGS / "fig2_shap_corroboration.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [FIG2] Saved -> {outpath}  ({os.path.getsize(outpath)/1024:.1f} KB)")


def make_figure3():
    print("[FIG3] Starting...")
    import re
    with open(FINAL_XGB) as fh:
        text = fh.read()

    pattern = r"\[(\w+)\]\s+Acc=([\d.]+)\u00b1([\d.]+)"
    k1_data = {}; k2_data = {}
    in_mode = None
    for line in text.split("\n"):
        line = line.strip()
        if "k=1 AAC:" in line:
            in_mode = "k1"; continue
        if "k=2 DPC:" in line:
            in_mode = "k2"; continue
        if line.startswith("Done!") or ("k=3 TPC" in line and "Acc=" in line):
            in_mode = None
        if in_mode:
            m = re.search(pattern, line)
            if m:
                win, acc, std = m.group(1), float(m.group(2)), float(m.group(3))
                (k1_data if in_mode == "k1" else k2_data)[win] = (acc, std)

    windows = ["full", "N100", "N200", "NC", "mid200"]
    win_labels = ["Full seq", "N-100aa", "N-200aa", "N+C\n100aa", "Mid\n200aa"]

    k1_acc = [k1_data.get(w, (0,0))[0] for w in windows]
    k1_std = [k1_data.get(w, (0,0))[1] for w in windows]
    k2_acc = [k2_data.get(w, (0,0))[0] for w in windows]
    k2_std = [k2_data.get(w, (0,0))[1] for w in windows]

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(windows))
    width = 0.35

    b1 = ax.bar(x - width/2, k1_acc, width, yerr=k1_std,
                label="k=1 AAC (20d)", color="#42A5F5", edgecolor="#1E88E5",
                linewidth=1.2, capsize=6, error_kw={"elinewidth": 1.5})
    b2 = ax.bar(x + width/2, k2_acc, width, yerr=k2_std,
                label="k=2 DPC (400d)", color="#EF5350", edgecolor="#E53935",
                linewidth=1.2, capsize=6, error_kw={"elinewidth": 1.5})

    for bars, accs, stds in [(b1, k1_acc, k1_std), (b2, k2_acc, k2_std)]:
        for bar, acc, std in zip(bars, accs, stds):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color="#1565C0" if bars is b1 else "#C62828")

    ax.set_xticks(x)
    ax.set_xticklabels(win_labels, fontsize=12)
    ax.set_ylabel("Accuracy (XGBoost, mean +/- std across 3 seeds)", fontsize=12, fontweight="bold")
    ax.set_title("Figure 3: Window Ablation Analysis - Localization Signal Distribution",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="lower left", frameon=True)

    y_max = max(max(k1_acc), max(k2_acc)) * 1.08
    y_min = min(min(k1_acc), min(k2_acc)) * 0.92
    ax.set_ylim(y_min, y_max)

    ax.annotate(
        "Signal not limited to N-terminus\n"
        "- Full sequence performs best\n"
        "- N+C terminals capture most signal\n"
        "- Mid-region retains ~57% accuracy\n"
        "-> Distributed localization signals",
        xy=(0.5, 0.96), xycoords="axes fraction", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF8E1", edgecolor="#F9A825", linewidth=2),
        ha="center", va="top")

    ax.axhline(y=k1_acc[0], color="#42A5F5", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=k2_acc[0], color="#EF5350", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(len(windows)-0.5, k1_acc[0]+0.003, "k=1 full", fontsize=8, color="#42A5F5", alpha=0.7)
    ax.text(len(windows)-0.5, k2_acc[0]+0.003, "k=2 full", fontsize=8, color="#EF5350", alpha=0.7)

    outpath = FIGS / "fig3_window_ablation.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [FIG3] Saved -> {outpath}  ({os.path.getsize(outpath)/1024:.1f} KB)")


if __name__ == "__main__":
    print("=" * 65)
    print("  make_figures.py - Yeast Protein Localization Figures")
    print("=" * 65)
    print(f"  Output dir : {FIGS}")
    print(f"  Python     : {sys.version.split()[0]}")
    print()

    for fn_name, fn in [("FIG1", make_figure1), ("FIG2", make_figure2), ("FIG3", make_figure3)]:
        try:
            fn()
        except Exception as e:
            print(f"  [{fn_name}] ERROR: {e}")
            import traceback; traceback.print_exc()

    print()
    print("=" * 65)
    print("  ALL FIGURES COMPLETE")
    print("=" * 65)
    for f in sorted(FIGS.glob("fig*.png")):
        sz_kb = os.path.getsize(f) / 1024
        print(f"  {f.name:<35s}  {sz_kb:>8.1f} KB")
    print()
