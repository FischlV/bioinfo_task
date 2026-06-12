import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
FIGS = '/data2/hyh/yeast_promoter_project/dataworkspace/figures'
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})

fig, ax = plt.subplots(figsize=(4.5, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis('off')

steps = [
    'SGD / UniProt\n数据库获取',
    '蛋白质序列\n(~4,860 酵母 ORFs)',
    'k-mer 编码\n(k=1, 2, 3, 融合)',
    '机器学习模型\n(RF / XGBoost / LogReg)',
    'SHAP 可解释性分析\n& 生物学佐证',
]
colors = ['#a8d8ea','#a8d8ea','#f9d89c','#b5e7a0','#f4b6c2']

for i, (s, clr) in enumerate(zip(steps, colors)):
    y = 10.8 - i * 2.3
    rect = mpatches.FancyBboxPatch((2.5, y-0.7), 5, 1.4, boxstyle='round,pad=0.1',
                                     facecolor=clr, edgecolor='gray', linewidth=1)
    ax.add_patch(rect)
    ax.text(5, y, s, ha='center', va='center', fontsize=9, fontweight='bold')
    if i < 4:
        ax.annotate('', xy=(5, y - 0.8), xytext=(5, y - 1.45),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, shrinkA=0, shrinkB=0))

ax.set_title('研究流程图', fontweight='bold', fontsize=13, pad=10)
plt.tight_layout()
for ext in ['png','pdf','svg']:
    fig.savefig(f'{FIGS}/fig1a_workflow.{ext}', dpi=250, bbox_inches='tight')
plt.close()
print('Workflow redrawn in Chinese')
