import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np, json, os
FIGS = '/data2/hyh/yeast_promoter_project/dataworkspace/figures'
os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})
all_acc = {}
for seed in ['123','456','789']:
    with open(f'/data2/hyh/yeast_promoter_project/dataworkspace/step2_models/matrix_seed{seed}.json') as f:
        data = json.load(f)
    for key, val in data.items():
        model, enc = key.split('|', 1)
        all_acc.setdefault(enc, {}).setdefault(model, []).append(val['Acc'])
enc_order = ['k=1 AAC (20d)','k=2 DPC (400d)','k=3 TPC (8000d)','Fusion k1+k2 (420d)','Binary enc','Integer enc','One-hot flatten']
enc_short = ['k=1\nAAC','k=2\nDPC','k=3\nTPC','Fusion','Binary\n24550d','Integer\n4910d','One-hot\n98200d']
model_order = ['LogReg','RF','XGBoost']
matrix = np.zeros((3, 7))
annot = [['']*7 for _ in range(3)]
for i, m in enumerate(model_order):
    for j, e in enumerate(enc_order):
        mean_val = np.mean(all_acc[e][m]) * 100
        matrix[i, j] = mean_val
        annot[i][j] = f'{mean_val:.1f}'
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={'width_ratios': [0.30, 0.70]})
axA.set_xlim(0, 10); axA.set_ylim(-0.5, 12.5); axA.axis('off')
steps = ['SGD / UniProt', 'Protein Sequences\n(4,860 yeast ORFs)', 'k-mer Encoding\n(k=1, 2, 3, fusion)', 'ML Models\n(RF / XGBoost / LogReg)', 'SHAP Analysis &\nBiological Corroboration']
colors = ['#a8d8ea','#a8d8ea','#f9d89c','#b5e7a0','#f4b6c2']
for i, (s, clr) in enumerate(zip(steps, colors)):
    y = 11.5 - i * 2.4
    rect = mpatches.FancyBboxPatch((2, y-0.75), 6, 1.5, boxstyle='round,pad=0.15', facecolor=clr, edgecolor='gray', linewidth=1)
    axA.add_patch(rect)
    axA.text(5, y, s, ha='center', va='center', fontsize=10, fontweight='bold')
    if i < 4:
        axA.annotate('', xy=(5, y - 1.55), xytext=(5, y - 0.85), arrowprops=dict(arrowstyle='->', color='gray', lw=2, shrinkA=0, shrinkB=0))
axA.set_title('A  Study Workflow', fontweight='bold', fontsize=12, loc='left')
axB.set_title('B  Model x Encoding Accuracy (%, mean of 3 seeds)', fontweight='bold', fontsize=12, loc='left')
sns.heatmap(matrix, annot=np.array(annot), fmt='', cmap='RdYlGn', xticklabels=enc_short, yticklabels=model_order, vmin=30, vmax=66, linewidths=0.5, linecolor='white', cbar_kws={'label': 'Accuracy (%)'}, ax=axB, annot_kws={'fontsize': 11, 'fontweight': 'bold'})
best_i, best_j = np.unravel_index(np.argmax(matrix), matrix.shape)
axB.add_patch(plt.Rectangle((best_j, best_i), 1, 1, fill=False, edgecolor='black', lw=3))
axB.tick_params(axis='x', rotation=25)
plt.tight_layout()
fig.savefig(f'{FIGS}/fig1_overview_heatmap.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{FIGS}/fig1_overview_heatmap.pdf', bbox_inches='tight')
fig.savefig(f'{FIGS}/fig1_overview_heatmap.svg', bbox_inches='tight')
plt.close()
print('Fig 1 saved')
