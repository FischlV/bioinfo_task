# 酵母启动子强度预测项目

## 项目结构

```
origin_datas/              ← 原始数据（参考基因组 + RNA-seq）
  ├── GCF_000146045.2_R64_genomic.fna
  ├── genomic.gff
  ├── cds_from_genomic.fna
  └── GSE301747_RNAseq.csv

dataworkspace/
  ├── step1_preprocess/     ← 数据预处理
  │   ├── promoters.fa      5742条启动子 (850bp)
  │   ├── expression.csv    表达量 + 元数据
  │   └── summary.txt
  ├── step2_features/       ← 特征工程
  │   ├── features_k3.csv   64维 k=3
  │   ├── features_k4.csv   256维 k=4 (主特征)
  │   ├── features_k5.csv   1024维 k=5
  │   ├── features_fusion.csv 1344维 融合
  │   ├── features_onehot.npy (5742, 850, 4) CNN输入
  │   ├── features_aux.csv  GC/TATA辅助特征
  │   ├── labels.csv        标签 + 数据划分
  │   └── train_idx.npy / test_idx.npy
  ├── step3_models/         ← 模型训练
  │   ├── model_rf.pkl
  │   ├── model_xgb.pkl
  │   ├── model_cnn.keras
  │   ├── model_comparison.csv/png
  │   ├── pred_vs_actual.png
  │   └── predictions.csv
  └── step4_interpretation/ ← SHAP解释
```

## 运行方式

### 环境
```
pip install pandas numpy biopython scikit-learn xgboost tensorflow matplotlib seaborn shap --break-system-packages
```

### 逐步运行
```bash
cd /path/to/生物信息学/dataworkspace/
python3 ../scripts/step1_preprocess.py
python3 ../scripts/step2_features.py
python3 ../scripts/step3_models.py        # ⚠️ CPU密集，建议服务器
python3 ../scripts/step4_shap.py
```

### 一键运行
```bash
bash ../scripts/run_all.sh
```

## 数据来源
- 基因组: S. cerevisiae S288c R64 (GCF_000146045.2)
- 表达量: GSE301747, BY4741 strain, glucose (SC medium), 3 replicates
- 启动子: TSS上游800bp + 下游50bp

## Step 3 结果 (当前)
| 模型 | R² | RMSE | Pearson r |
|------|-----|------|-----------|
| Random Forest | 0.030 | 2.212 | 0.177 |
| XGBoost | -0.005 | 2.252 | 0.141 |
| CNN | -0.893 | 3.091 | 0.366 |

R² 较低说明纯启动子序列信息不足以精确预测表达强度——这是该领域的已知挑战。
讨论部分可以从以下角度展开：
- 染色质结构/核小体占位对启动子活性的影响
- 转录因子可用性的细胞状态依赖性
- mRNA稳定性的转录后调控
- 增强子/沉默子等远端调控元件
- 与Kotopka & Smolke (2020)等工作的对比
