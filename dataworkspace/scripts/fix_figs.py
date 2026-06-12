"""Fix the two figures that have issues."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np, pandas as pd, json, os

FIGS = '/data2/hyh/yeast_promoter_project/dataworkspace/figures'
os.makedirs(FIGS, exist_ok=True)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'  # use basic font, no special chars

# ===== FIG 1: Heatmap fix - remove broken characters =====
BASE = '/data2/hyh/yeast_promoter_project/dataworkspace/step2_models'

# Load all 3 seeds
all_acc = {}
for seed in ['123','456','789']:
    with open(f'{BASE}/matrix_seed{seed}.json') as f:
        data = json.load(f)
    for key, val in data.items():
        model, enc = key.split('|', 1)
        if enc not in all_acc: all_acc[enc] = {}
        if model not in all_acc[enc]: all_acc[enc][model] = []
        all_acc[enc][model].append(val['Acc'])

enc_order = ['k=1 AAC (20d)','k=2 DPC (400d)','k=3 TPC (8000d)',
             'Fusion k1+k2 (420d)','Binary enc (5000d)','Integer enc (1000d)','One-hot flatten (20000d)']
enc_short = ['k=1 AAC','k=2 DPC','k=3 TPC','Fusion','Binary','Integer','One-hot']
model_order = ['LogReg','RF','XGBoost']

matrix = np.zeros((3, 7))
annot = [['']*7 for _ in range(3)]
for i, m in enumerate(model_order):
    for j, e in enumerate(enc_order):
        vals = all_acc[e][m]
        matrix[i,j] = np.mean(vals)
        # Simple format: no special chars
        annot[i][j] = f'{matrix[i,j]:.1f}'

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [0.28, 0.72]})

# Panel A: Flowchart
axA.set_xlim(0, 10)
axA.set_ylim(0, 12)
axA.axis('off')
steps = ['SGD / UniProt', 'Protein Sequences\n(~4,860 yeast ORFs)',
         'k-mer Encoding\n(k=1,2,3, fusion)', 'ML Models\n(RF / XGBoost / LogReg)',
         'SHAP Analysis &\nBiological Corroboration']
colors = ['#a8d8ea','#a8d8ea','#f9d89c','#b5e7a0','#f4b6c2']
for i, (s, c) in enumerate(zip(steps, colors)):
    y = 11.5 - i * 2.2
    rect = mpatches.FancyBboxPatch((2, y-0.7), 6, 1.4, boxstyle='round,pad=0.15',
                                     facecolor=c, edgecolor='gray', linewidth=1)
    axA.add_patch(rect)
    axA.text(5, y, s, ha='center', va='center', fontsize=10, fontweight='bold')
    if i < 4:
        axA.annotate('', xy=(5, y-0.8), xytext=(5, y+0.8-2.2),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
axA.set_title('A  Study Workflow', fontweight='bold', fontsize=12, loc='left')

# Panel B: Heatmap
axB.set_title('B  Model × Encoding Accuracy (mean of 3 seeds)', fontweight='bold', fontsize=12, loc='left')
sns.heatmap(matrix, annot=np.array(annot), fmt='', cmap='RdYlGn',
            xticklabels=enc_short, yticklabels=model_order,
            vmin=25, vmax=66, linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Accuracy'}, ax=axB, annot_kws={'fontsize': 11, 'fontweight': 'bold'})
# Highlight best cell
best_i, best_j = np.unravel_index(np.argmax(matrix), matrix.shape)
axB.add_patch(plt.Rectangle((best_j, best_i), 1, 1, fill=False, edgecolor='black', lw=3))
axB.set_xlabel('')
axB.tick_params(axis='x', rotation=30)
fig.suptitle('', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{FIGS}/fig1_overview_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print('Fig 1 saved')

# ===== FIG 2: Fix layout overlap =====
shap_dir = '/data2/hyh/yeast_promoter_project/dataworkspace/step3_shap'
with open(f'{shap_dir}/shap_top_features_per_class.json') as f:
    top_feats = json.load(f)

# Create figure with proper spacing
fig = plt.figure(figsize=(16, 10))

# Panel A: Top 15 overall - use gridspec
gs = fig.add_gridspec(2, 2, height_ratios=[0.45, 0.55], width_ratios=[0.55, 0.45],
                      hspace=0.35, wspace=0.25)

axA = fig.add_subplot(gs[0, :])  # Full width for panel A

# Combine all class features, compute global importance
global_imp = {}
for cls, feats in top_feats.items():
    for name, val in feats:
        dp = name.replace('kmer_', '')
        if dp not in global_imp: global_imp[dp] = 0
        global_imp[dp] += val  # sum across classes

top15 = sorted(global_imp.items(), key=lambda x: x[1], reverse=True)[:15]
names = [x[0] for x in top15][::-1]
values = [x[1] for x in top15][::-1]
colors_a = plt.cm.viridis(np.linspace(0.15, 0.85, 15))

bars = axA.barh(range(15), values, color=colors_a)
for bar, val in zip(bars, values):
    axA.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
axA.set_yticks(range(15))
axA.set_yticklabels(names, fontsize=9, fontfamily='monospace')
axA.set_xlabel('Mean |SHAP| value', fontsize=11)
axA.set_title('A  Overall Top-15 Dipeptide Features (SHAP Importance)', fontweight='bold', fontsize=13, loc='left')
axA.set_xlim(0, max(values)*1.15)

# Panel B: Per-class top 5
class_names = list(top_feats.keys())
class_colors = ['#2ecc71','#e74c3c','#3498db','#e91e63','#1abc9c']
for idx, (cls, color) in enumerate(zip(class_names, class_colors)):
    ax = fig.add_subplot(gs[1, 0])  # All 5 in grid
    # Actually create 5 sub-axes within the bottom-left area
    pass

# Create 5 small subplots in a row for class-specific features
gs_b = gs[1, 0].subgridspec(1, 5, wspace=0.3)

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
    ax.set_xlim(0, max(vals_cls)*1.2)
    if idx > 0:
        ax.set_yticklabels([])
    ax.tick_params(axis='x', labelsize=7)

gs_b_title = gs[1, 0]
fig.text(0.05, 0.48, 'B  Per-Class Top-5 Dipeptide Features (SHAP Importance)',
         fontweight='bold', fontsize=12, ha='left', va='bottom')

# Panel C: Corroboration table
axC = fig.add_subplot(gs[1, 1])
axC.axis('off')

# Build table data
table_data = [
    ['FF', 'Membrane/Secretory', '2.69', '<0.001', 'Confirmed (signal peptide)'],
    ['LF', 'Membrane/Secretory', '1.86', '<0.001', 'Confirmed (signal peptide)'],
    ['FI', 'Membrane/Secretory', '1.76', '<0.001', 'Novel finding'],
    ['KR', 'Nucleus', '1.41', '<0.001', 'Confirmed (NLS)'],
    ['RK', 'Nucleus', '1.27', '0.002', 'Partly supported'],
    ['DD', 'Nucleus', '1.60', '<0.001', 'Novel (acidic patch?)'],
    ['DD', 'Mitochondria', '0.62', '<0.001', 'Depleted (matches MTP)'],
    ['KG', 'Mitochondria', '1.45', '<0.001', 'Novel finding'],
    ['EE', 'Mitochondria', '0.75', '<0.001', 'Depleted (matches MTP)'],
    ['HG', 'Cytoplasm', '1.12', '0.04', 'Novel finding'],
]

col_labels = ['Dipeptide', 'Class', 'Enrich.\nRatio', 'p-adj', 'Verdict']
cell_colors = []
for row in table_data:
    v = row[-1]
    if 'Confirmed' in v:
        cell_colors.append(['#c8e6c9']*5)
    elif 'Novel' in v:
        cell_colors.append(['#bbdefb']*5)
    elif 'Depleted' in v or 'Partly' in v:
        cell_colors.append(['#fff9c4']*5)
    else:
        cell_colors.append(['white']*5)

table = axC.table(cellText=table_data, colLabels=col_labels, cellColours=cell_colors,
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1.0, 1.2)

# Style header
for j in range(5):
    table[0, j].set_facecolor('#455a64')
    table[0, j].set_text_props(color='white', fontweight='bold', fontsize=7)

axC.set_title('C  Biological Corroboration', fontweight='bold', fontsize=12, loc='left')

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#c8e6c9', label='Confirmed (lit. support)'),
    mpatches.Patch(facecolor='#bbdefb', label='Novel finding'),
    mpatches.Patch(facecolor='#fff9c4', label='Depleted / Partial'),
]
axC.legend(handles=legend_elements, loc='lower center', fontsize=7, ncol=3,
           bbox_to_anchor=(0.5, -0.08))

plt.tight_layout(pad=1.5, h_pad=2.0, w_pad=1.5)
fig.savefig(f'{FIGS}/fig2_shap_corroboration.png', dpi=250, bbox_inches='tight')
plt.close()
print('Fig 2 saved')
print('Done!')
