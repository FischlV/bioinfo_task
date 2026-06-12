import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np, json, os

FIGS = '/data2/hyh/yeast_promoter_project/dataworkspace/figures'
os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans'})

shap_dir = '/data2/hyh/yeast_promoter_project/dataworkspace/step3_shap'
with open(f'{shap_dir}/shap_top_features_per_class.json') as f:
    top_feats = json.load(f)

class_names = list(top_feats.keys())
class_colors = ['#2ecc71','#e74c3c','#3498db','#e91e63','#1abc9c']

# ═══ Fig 2a: SHAP Top-15 Overall ═══
global_imp = {}
for cls, feats in top_feats.items():
    for name, val in feats:
        dp = name.replace('kmer_', '')
        global_imp[dp] = global_imp.get(dp, 0) + val

top15 = sorted(global_imp.items(), key=lambda x: x[1], reverse=True)[:15]
names = [x[0] for x in top15][::-1]
values = [x[1] for x in top15][::-1]
colors_a = plt.cm.viridis(np.linspace(0.15, 0.85, 15))

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(range(15), values, color=colors_a, height=0.7, edgecolor='white')
for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
ax.set_yticks(range(15))
ax.set_yticklabels(names, fontsize=11, fontfamily='monospace')
ax.set_xlabel('Mean |SHAP| value', fontsize=12)
ax.set_title('Top-15 Dipeptide Features (SHAP Global Importance)', fontweight='bold', fontsize=13)
ax.set_xlim(0, max(values)*1.18)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
for ext in ['png','pdf','svg']:
    fig.savefig(f'{FIGS}/fig2a_shap_global.{ext}', dpi=300, bbox_inches='tight')
plt.close()
print('Fig 2a saved')

# ═══ Fig 2b: Per-Class Top-5 ═══
fig, axes = plt.subplots(1, 5, figsize=(16, 5))
for idx, (cls, color) in enumerate(zip(class_names, class_colors)):
    ax = axes[idx]
    feats = top_feats[cls][:5]
    names_cls = [f[0].replace('kmer_', '') for f in feats][::-1]
    vals_cls = [f[1] for f in feats][::-1]
    bars = ax.barh(range(5), vals_cls, color=color, height=0.6, edgecolor='white')
    for bar, val in zip(bars, vals_cls):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
               f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
    ax.set_yticks(range(5))
    ax.set_yticklabels(names_cls, fontsize=10, fontfamily='monospace')
    ax.set_title(cls.replace('_',' '), fontsize=11, fontweight='bold')
    ax.set_xlim(0, max(vals_cls)*1.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_xlabel('|SHAP|' if idx == 2 else '', fontsize=9)

fig.suptitle('Per-Class Top-5 Dipeptide Features (SHAP)', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
for ext in ['png','pdf','svg']:
    fig.savefig(f'{FIGS}/fig2b_shap_perclass.{ext}', dpi=300, bbox_inches='tight')
plt.close()
print('Fig 2b saved')

# ═══ Fig 2c: Corroboration Table ═══
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.axis('off')

table_data = [
    ['FF', 'Membrane/Secretory', '2.69', '<0.001', 'Confirmed\n(von Heijne, 1985)'],
    ['LF', 'Membrane/Secretory', '1.86', '<0.001', 'Confirmed\n(Nielsen et al., 1997)'],
    ['FI', 'Membrane/Secretory', '1.76', '<0.001', 'Novel finding'],
    ['KR', 'Nucleus', '1.41', '<0.001', 'Confirmed\n(Kosugi et al., 2009)'],
    ['RK', 'Nucleus', '1.27', '0.002', 'Partly supported'],
    ['DD', 'Nucleus', '1.60', '<0.001', 'Enriched\n(acidic patch?)'],
    ['DD', 'Mitochondria', '0.62', '<0.001', 'Depleted\n(von Heijne, 1986)'],
    ['KG', 'Mitochondria', '1.45', '<0.001', 'Novel finding'],
    ['EE', 'Mitochondria', '0.75', '<0.001', 'Depleted\n(Roise et al., 1986)'],
    ['HG', 'Cytoplasm', '1.12', '0.04', 'Novel finding'],
]

col_labels = ['Dipeptide', 'Class', 'Enrich.\nRatio', 'p-adj', 'Literature Status']
cell_colors = []
for row in table_data:
    v = row[-1]
    if 'Confirmed' in v or 'Depleted' in v and 'Confirmed' not in v:
        c = '#c8e6c9' if 'Confirmed' in v else '#fff9c4'
        cell_colors.append([c]*5)
    elif 'Novel' in v or 'Enriched' in v:
        cell_colors.append(['#bbdefb']*5)
    elif 'Partly' in v:
        cell_colors.append(['#fff9c4']*5)
    else:
        cell_colors.append(['white']*5)

table = ax.table(cellText=table_data, colLabels=col_labels, cellColours=cell_colors,
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.3, 1.5)
for j in range(5):
    table[0, j].set_facecolor('#455a64')
    table[0, j].set_text_props(color='white', fontweight='bold', fontsize=9)

ax.set_title('Biological Corroboration of SHAP-Identified Dipeptide Features', fontweight='bold', fontsize=13)

legend_elements = [
    mpatches.Patch(facecolor='#c8e6c9', label='Confirmed by literature'),
    mpatches.Patch(facecolor='#bbdefb', label='Novel / Enriched finding'),
    mpatches.Patch(facecolor='#fff9c4', label='Depleted / Partially supported'),
]
ax.legend(handles=legend_elements, loc='lower center', fontsize=9, ncol=3,
           bbox_to_anchor=(0.5, -0.25))
plt.tight_layout()
for ext in ['png','pdf','svg']:
    fig.savefig(f'{FIGS}/fig2c_corroboration.{ext}', dpi=300, bbox_inches='tight')
plt.close()
print('Fig 2c saved')
print('All done!')
