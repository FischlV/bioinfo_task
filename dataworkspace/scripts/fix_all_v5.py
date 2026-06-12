import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np, pandas as pd, json, os

FIGS = '/data2/hyh/yeast_promoter_project/dataworkspace/figures'
os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})

BASE = '/data2/hyh/yeast_promoter_project/dataworkspace/step2_models'

all_acc = {}
for seed in ['123','456','789']:
    with open(f'{BASE}/matrix_seed{seed}.json') as f:
        data = json.load(f)
    for key, val in data.items():
        model, enc = key.split('|', 1)
        all_acc.setdefault(enc, {}).setdefault(model, []).append(val['Acc'])

enc_order = ['k=1 AAC (20d)','k=2 DPC (400d)','k=3 TPC (8000d)',
             'Fusion k1+k2 (420d)','Binary enc (5000d)','Integer enc (1000d)','One-hot flatten (20000d)']
enc_short = ['k=1\nAAC','k=2\nDPC','k=3\nTPC','Fusion','Binary','Integer','One-hot']
model_order = ['LogReg','RF','XGBoost']
matrix = np.zeros((3, 7))
annot = [['']*7 for _ in range(3)]
for i, m in enumerate(model_order):
    for j, e in enumerate(enc_order):
        mean_val = np.mean(all_acc[e][m]) * 100
        matrix[i, j] = mean_val
        annot[i][j] = f'{mean_val:.1f}'

# ═══ FIGURE 1 ═══
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={'width_ratios': [0.30, 0.70]})
axA.set_xlim(0, 10); axA.set_ylim(-0.5, 12.5); axA.axis('off')
steps = ['SGD / UniProt', 'Protein Sequences\n(4,860 yeast ORFs)',
         'k-mer Encoding\n(k=1, 2, 3, fusion)', 'ML Models\n(RF / XGBoost / LogReg)',
         'SHAP Analysis &\nBiological Corroboration']
colors = ['#a8d8ea','#a8d8ea','#f9d89c','#b5e7a0','#f4b6c2']
for i, (s, c) in enumerate(zip(steps, colors)):
    y = 11.5 - i * 2.4
    rect = mpatches.FancyBboxPatch((2, y-0.75), 6, 1.5, boxstyle='round,pad=0.15',
                                     facecolor=c, edgecolor='gray', linewidth=1)
    axA.add_patch(rect)
    axA.text(5, y, s, ha='center', va='center', fontsize=10, fontweight='bold')
    if i < 4:
        axA.annotate('', xy=(5, y - 1.55), xytext=(5, y - 0.85),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2, shrinkA=0, shrinkB=0))
axA.set_title('A  Study Workflow', fontweight='bold', fontsize=12, loc='left')
axB.set_title('B  Model x Encoding Accuracy (%, mean of 3 seeds)', fontweight='bold', fontsize=12, loc='left')
sns.heatmap(matrix, annot=np.array(annot), fmt='', cmap='RdYlGn',
            xticklabels=enc_short, yticklabels=model_order,
            vmin=25, vmax=66, linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Accuracy (%)'}, ax=axB,
            annot_kws={'fontsize': 11, 'fontweight': 'bold'})
best_i, best_j = np.unravel_index(np.argmax(matrix), matrix.shape)
axB.add_patch(plt.Rectangle((best_j, best_i), 1, 1, fill=False, edgecolor='black', lw=3))
axB.tick_params(axis='x', rotation=25)
plt.tight_layout()
fig.savefig(f'{FIGS}/fig1_overview_heatmap.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{FIGS}/fig1_overview_heatmap.pdf', bbox_inches='tight')
fig.savefig(f'{FIGS}/fig1_overview_heatmap.svg', bbox_inches='tight')
plt.close()
print('Fig 1 saved')

# ═══ FIGURE 2 — fix overlap ═══
shap_dir = '/data2/hyh/yeast_promoter_project/dataworkspace/step3_shap'
with open(f'{shap_dir}/shap_top_features_per_class.json') as f:
    top_feats = json.load(f)

fig = plt.figure(figsize=(17, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[0.38, 0.62], width_ratios=[0.50, 0.50],
                      hspace=0.55, wspace=0.25)

# Panel A: Top 15 overall
axA = fig.add_subplot(gs[0, :])
global_imp = {}
for cls, feats in top_feats.items():
    for name, val in feats:
        dp = name.replace('kmer_', '')
        global_imp[dp] = global_imp.get(dp, 0) + val
top15 = sorted(global_imp.items(), key=lambda x: x[1], reverse=True)[:15]
names = [x[0] for x in top15][::-1]
values = [x[1] for x in top15][::-1]
colors_a = plt.cm.viridis(np.linspace(0.15, 0.85, 15))
bars = axA.barh(range(15), values, color=colors_a, height=0.7)
for bar, val in zip(bars, values):
    axA.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
axA.set_yticks(range(15))
axA.set_yticklabels(names, fontsize=9, fontfamily='monospace')
axA.set_xlabel('Mean |SHAP| value', fontsize=11)
axA.set_title('A  Overall Top-15 Dipeptide Features (SHAP Importance)', fontweight='bold', fontsize=13, loc='left')
axA.set_xlim(0, max(values)*1.18)

# Panel B: Per-class top 5 — use title inside the axes area instead of fig.text
class_names = list(top_feats.keys())
class_colors = ['#2ecc71','#e74c3c','#3498db','#e91e63','#1abc9c']
gs_b = gs[1, 0].subgridspec(1, 5, wspace=0.40)

for idx, (cls, color) in enumerate(zip(class_names, class_colors)):
    ax = fig.add_subplot(gs_b[0, idx])
    feats = top_feats[cls][:5]
    names_cls = [f[0].replace('kmer_', '') for f in feats][::-1]
    vals_cls = [f[1] for f in feats][::-1]
    bars = ax.barh(range(5), vals_cls, color=color, height=0.6)
    for bar, val in zip(bars, vals_cls):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=7, fontweight='bold')
    ax.set_yticks(range(5))
    ax.set_yticklabels(names_cls, fontsize=8, fontfamily='monospace')
    ax.set_title(cls.replace('_',' '), fontsize=9, fontweight='bold')
    ax.set_xlim(0, max(vals_cls)*1.45)
    if idx > 0:
        ax.set_yticklabels([])
    ax.set_xticks([0, 0.1, 0.2])
    ax.tick_params(axis='x', labelsize=7)

# Use fig.text with more spacing below Panel A
fig.text(0.03, 0.42, 'B  Per-Class Top-5 Dipeptide Features (SHAP Importance)',
         fontweight='bold', fontsize=12, ha='left', va='bottom')

# Panel C: Corroboration table — legend right below
axC = fig.add_subplot(gs[1, 1])
axC.axis('off')

table_data = [
    ['FF', 'Membrane', '2.69', '<0.001', 'Confirmed (signal peptide)'],
    ['LF', 'Membrane', '1.86', '<0.001', 'Confirmed (signal peptide)'],
    ['FI', 'Membrane', '1.76', '<0.001', 'Novel finding'],
    ['KR', 'Nucleus', '1.41', '<0.001', 'Confirmed (NLS)'],
    ['RK', 'Nucleus', '1.27', '0.002', 'Partly supported'],
    ['DD', 'Nucleus', '1.60', '<0.001', 'Enriched (acidic patch)'],
    ['DD', 'Mitochondria', '0.62', '<0.001', 'Depleted (matches MTP)'],
    ['KG', 'Mitochondria', '1.45', '<0.001', 'Novel finding'],
    ['EE', 'Mitochondria', '0.75', '<0.001', 'Depleted (matches MTP)'],
    ['HG', 'Cytoplasm', '1.12', '0.04', 'Novel finding'],
]

col_labels = ['Dipeptide', 'Class', 'Ratio', 'p-adj', 'Verdict']
cell_colors = []
for row in table_data:
    v = row[-1]
    if 'Confirmed' in v:
        cell_colors.append(['#c8e6c9']*5)
    elif 'Novel' in v or 'Enriched' in v:
        cell_colors.append(['#bbdefb']*5)
    elif 'Depleted' in v or 'Partly' in v:
        cell_colors.append(['#fff9c4']*5)
    else:
        cell_colors.append(['white']*5)

table = axC.table(cellText=table_data, colLabels=col_labels, cellColours=cell_colors,
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.4, 1.5)
for j in range(5):
    table[0, j].set_facecolor('#455a64')
    table[0, j].set_text_props(color='white', fontweight='bold', fontsize=8)

axC.set_title('C  Biological Corroboration', fontweight='bold', fontsize=12, loc='left')

legend_elements = [
    mpatches.Patch(facecolor='#c8e6c9', label='Confirmed (lit.)'),
    mpatches.Patch(facecolor='#bbdefb', label='Novel / Enriched'),
    mpatches.Patch(facecolor='#fff9c4', label='Depleted / Partial'),
]
axC.legend(handles=legend_elements, loc='upper center', fontsize=8, ncol=3,
           bbox_to_anchor=(0.5, -0.02))

fig.savefig(f'{FIGS}/fig2_shap_corroboration.png', dpi=250, bbox_inches='tight')
fig.savefig(f'{FIGS}/fig2_shap_corroboration.pdf', bbox_inches='tight')
fig.savefig(f'{FIGS}/fig2_shap_corroboration.svg', bbox_inches='tight')
plt.close()
print('Fig 2 saved')

# ═══ FIGURE 3 ═══
windows = ['Full seq', 'N-100aa', 'N-200aa', 'N+C 100aa', 'Mid 200aa']
k1_mean = np.array([0.624, 0.541, 0.578, 0.590, 0.564])
k1_std  = np.array([0.005, 0.005, 0.004, 0.003, 0.011])
k2_mean = np.array([0.631, 0.573, 0.574, 0.592, 0.571])
k2_std  = np.array([0.003, 0.007, 0.007, 0.002, 0.002])

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(windows)); w = 0.35
b1 = ax.bar(x - w/2, k1_mean, w, yerr=k1_std, capsize=4, color='#3498db', label='k=1 AAC', edgecolor='white')
b2 = ax.bar(x + w/2, k2_mean, w, yerr=k2_std, capsize=4, color='#e74c3c', label='k=2 DPC', edgecolor='white')
for bar, val, std in zip(b1, k1_mean, k1_std):
    ax.text(bar.get_x() + bar.get_width()/2, val + std + 0.008, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2980b9')
for bar, val, std in zip(b2, k2_mean, k2_std):
    ax.text(bar.get_x() + bar.get_width()/2, val + std + 0.008, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#c0392b')
ax.axhline(y=k2_mean[0], color='#e74c3c', linestyle='--', alpha=0.4, linewidth=1)
ax.text(4.2, k2_mean[0], f'k=2 full ({k2_mean[0]:.3f})', fontsize=8, color='#e74c3c', va='center')
ax.axhline(y=k1_mean[0], color='#3498db', linestyle='--', alpha=0.4, linewidth=1)
ax.text(4.2, k1_mean[0], f'k=1 full ({k1_mean[0]:.3f})', fontsize=8, color='#3498db', va='center')
ax.set_xticks(x); ax.set_xticklabels(windows, fontsize=10)
ax.set_ylabel('Accuracy (XGBoost, mean +/- std, 3 seeds)', fontsize=11)
ax.set_ylim(0.50, 0.68); ax.legend(loc='lower left', fontsize=9); ax.grid(axis='y', alpha=0.3)
ax.set_title('Window Ablation: Localization Signal Distribution', fontweight='bold', fontsize=13)
plt.tight_layout()
fig.savefig(f'{FIGS}/fig3_window_ablation.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{FIGS}/fig3_window_ablation.pdf', bbox_inches='tight')
fig.savefig(f'{FIGS}/fig3_window_ablation.svg', bbox_inches='tight')
plt.close()
print('Fig 3 saved')
print('All done!')
