# 基于多模型对比与SHAP可解释性的酿酒酵母蛋白质亚细胞定位预测

> 生物信息学课程期末大作业 | 2026年5月

## 项目概述

以酿酒酵母（*Saccharomyces cerevisiae* S288c）为模式系统，从 UniProt/SGD 获取 4,863 条蛋白质序列（五分类），构建 7 种编码方案 × 3 种机器学习模型（LogReg/RF/XGBoost）的完整实验矩阵，采用 SHAP 进行可解释性分析并通过独立外部知识库进行生物学佐证。

**最优结果**: XGBoost + 融合编码（420维）准确率 63.6%±0.3%

## 目录结构

```
.
├── .gitignore                          # Git 忽略规则
├── 期末大作业.pdf                       # 课程作业要求
├── 期末大作业_蛋白亚细胞定位预测_方案.md  # 研究方案/Proposal
│
├── 41467_2020_15977_MOESM4_ESM.xls     # 参考：酵母启动子库补充数据
├── Construction,...promoter library...pdf  # 参考：酵母启动子构建文献
├── Model-driven generation...pdf        # 参考：AI生成启动子文献
├── GSE290842_counts.txt                 # 参考：RNA-seq表达数据
├── GSE312455_counts.tsv                 # 参考：RNA-seq表达数据
│
├── ncbi_dataset.zip                     # NCBI酵母基因组原始下载
├── origin_datas/                        # 原始参考数据（启动子项目）
│   ├── GCF_000146045.2_R64_genomic.fna  # 酵母参考基因组
│   ├── cds_from_genomic.fna             # CDS序列
│   ├── genomic.gff                      # 基因组注释
│   ├── rna.fna                          # RNA序列
│   └── GSE301747_RNAseq.csv             # RNA-seq表达量
│
├── results/                             # 最终图表 + 论文
│   ├── 论文_完整版.docx                  # 完整论文（含中英摘要）
│   ├── fig1a_workflow.{png,svg,pdf}     # 研究流程图
│   ├── fig1b_heatmap.{png,svg,pdf}      # 类别特征热图
│   ├── fig2a_shap_global.{png,svg,pdf}  # SHAP全局蜂群图
│   ├── fig2b_shap_perclass.{png,svg,pdf}# SHAP各类别Top特征
│   ├── fig2_shap_corroboration_sm.png   # SHAP与生物学佐证对照图
│   └── fig3_window_ablation.{png,svg,pdf}# 窗口消融实验
│
└── dataworkspace/                       # 实验主目录
    ├── README.md                         # 运行说明
    │
    ├── scripts/                          # 全部脚本
    │   ├── run_all.sh                    # 一键运行入口
    │   ├── run_matrix_all.sh             # 运行完整实验矩阵
    │   ├── run_step4.sh                  # 运行Step 4 Motif验证
    │   │
    │   ├── step0_get_data.py             # 从UniProt获取蛋白序列+定位标签
    │   ├── step1_features.py             # 特征工程（k-mer/one-hot/窗口变体）
    │   ├── step1_preprocess.py           # （启动子项目）数据预处理
    │   ├── step2_features.py             # （启动子项目）特征工程
    │   ├── step2_models.py               # 模型训练（实验A/B/C）
    │   ├── step3_models.py               # （启动子项目）模型训练
    │   ├── step3_shap.py                 # SHAP可解释性分析
    │   ├── step4_shap.py                 # （启动子项目）SHAP分析
    │   ├── step4_motif.py                # Motif生物学验证
    │   │
    │   ├── expA_seed.py                  # 独立运行实验A（指定seed）
    │   ├── expB_seed.py                  # 独立运行实验B（指定seed）
    │   ├── fill_matrix.py                # 填充实验矩阵缺失格
    │   ├── refill_matrix.py              # 补充运行实验矩阵
    │   ├── refill_one.py                 # 单格补充运行
    │   ├── rebuild_onehot.py             # 重新构建One-hot编码
    │   │
    │   ├── make_figures.py               # 生成论文图表
    │   ├── fix_figs.py                   # 修图脚本 v1
    │   ├── fix_figs_v2.py                # 修图脚本 v2
    │   ├── fix_all_v3.py                 # 批量修图 v3（最终版）
    │   ├── fix_all_v4.py                 # 批量修图 v4
    │   ├── fix_all_v5.py                 # 批量修图 v5（白色背景）
    │   ├── fig1_v3.py                    # 单独生成Fig 1
    │   ├── fig_workflow_cn.py            # 中文流程图 v1
    │   ├── fig_workflow_cn2.py           # 中文流程图 v2
    │   ├── split_fig1.py                 # Fig 1 拆分子图
    │   └── split_fig2.py                 # Fig 2 拆分子图
    │
    ├── step0_data/                       # 原始数据
    │   ├── protein_sequences.fa          # 4,863 条蛋白序列 (FASTA, 2.7MB)
    │   ├── protein_labels.csv            # 基因ID + 亚细胞定位标签
    │   └── summary.txt                   # 数据统计摘要（各类别分布）
    │
    ├── step1_features/                   # 特征工程输出
    │   ├── labels.csv                    # 标签 + train/test 划分
    │   ├── train_idx.npy                 # 训练集索引 (n=3890)
    │   ├── test_idx.npy                  # 测试集索引 (n=973)
    │   ├── label_encoder.npy             # 类别编码映射 (Cytoplasm=0, Membrane_Secretory=1, ...)
    │   ├── summary.txt                   # 特征工程统计摘要
    │   │
    │   │  # k-mer 频率特征
    │   ├── kmers_k1.txt                  # k=1 氨基酸列表 (20种)
    │   ├── kmers_k2.txt                  # k=2 二肽列表 (400种)
    │   ├── kmers_k3.txt                  # k=3 三肽列表 (8000种)
    │   ├── features_k1.csv               # AAC: 单氨基酸频率 (20维)
    │   ├── features_k2.csv               # DPC: 二肽频率 (400维)
    │   ├── features_fusion.csv           # k1+k2 融合编码 (420维)
    │   ├── features_aux.csv              # 辅助特征 (长度/pI/疏水性)
    │   │
    │   │  # 序列窗口变体 (k=1 AAC)
    │   ├── features_k1_full.csv          # 完整序列
    │   ├── features_k1_N100.csv          # N端100aa
    │   ├── features_k1_N200.csv          # N端200aa
    │   ├── features_k1_NC.csv            # N端+C端 各100aa
    │   ├── features_k1_mid200.csv        # 中间200aa (25%-75%)
    │   │
    │   │  # 序列窗口变体 (k=2 DPC)
    │   ├── features_k2_full.csv          # 完整序列
    │   ├── features_k2_N100.csv          # N端100aa
    │   ├── features_k2_N200.csv          # N端200aa
    │   ├── features_k2_NC.csv            # N端+C端 各100aa
    │   ├── features_k2_mid200.csv        # 中间200aa
    │   │
    │   │  # 离散编码
    │   ├── features_int.npy              # 整数编码 (4863, 1000)
    │   └── features_onehot.npy           # One-hot编码 (4863, 1000, 20) [大文件，本地不存]
    │
    ├── step2_models/                     # 模型训练输出
    │   ├── model_lr.pkl                  # Logistic Regression 模型
    │   ├── model_rf.pkl                  # Random Forest 模型 (59MB)
    │   ├── model_xgb.pkl                 # XGBoost 模型 (7MB)
    │   ├── model_comparison.csv          # 模型对比结果表 (Acc/F1/AUC)
    │   ├── model_comparison.png          # 模型对比图
    │   ├── predictions.csv               # 测试集预测值
    │   ├── summary.txt                   # 实验A/B/C结果汇总
    │   ├── matrix_summary.txt            # 完整实验矩阵汇总 (7编码×3模型×3seed)
    │   │
    │   │  # 实验A: 模型对比 (seed 123/456)
    │   ├── expA_seed123.json             # seed 123 完整结果
    │   ├── expA_seed123.out              # seed 123 运行日志
    │   ├── expA_seed456.json             # seed 456 完整结果
    │   ├── expA_seed456.out              # seed 456 运行日志
    │   ├── expA_model_comparison.png     # 实验A可视化
    │   │
    │   │  # 实验B: 位置/顺序消融 (seed 123/456)
    │   ├── expB_seed123.json             # seed 123 完整结果
    │   ├── expB_seed123.out              # seed 123 运行日志
    │   ├── expB_seed456.json             # seed 456 完整结果
    │   ├── expB_seed456.out              # seed 456 运行日志
    │   ├── expB_position_ablation.png    # 实验B可视化
    │   │
    │   │  # 实验C: 窗口消融
    │   ├── ablation_windows.csv          # 各窗口准确率数据
    │   ├── ablation_windows.png          # 窗口消融柱状图
    │   ├── expC_window_ablation.png      # 实验C可视化
    │   │
    │   │  # 完整实验矩阵 (7编码×3模型, seed 123/456/789)
    │   ├── matrix_seed123.json           # seed 123 矩阵结果
    │   ├── matrix_seed123.out            # seed 123 运行日志
    │   ├── matrix_seed456.json           # seed 456 矩阵结果
    │   ├── matrix_seed456.out            # seed 456 运行日志
    │   ├── matrix_seed789.json           # seed 789 矩阵结果
    │   ├── matrix_seed789.out            # seed 789 运行日志
    │   │
    │   │  # 其他
    │   ├── final_xgb.out                 # 最终XGBoost训练日志 (n=200)
    │   ├── confusion_matrix.png          # 最佳模型混淆矩阵
    │   └── run_matrix_all.out            # 矩阵批量运行日志
    │
    ├── step3_shap/                       # SHAP可解释性分析
    │   ├── shap_values_full.npy          # 全特征SHAP值 (973, 400, 5) [7.5MB]
    │   ├── shap_top_features_per_class.json  # 各类别Top特征 + SHAP值
    │   ├── summary.txt                   # SHAP分析摘要 + Top特征
    │   ├── references_shap.md            # SHAP参考文献/方法笔记
    │   ├── shap_beeswarm.png             # 全局SHAP蜂群图
    │   ├── shap_per_class.png            # 各类别Top-5特征条形图
    │   ├── shap_by_feature_type.png      # 按特征类型分组SHAP
    │   └── shap_dependence/              # Top-5二肽依赖图
    │       ├── dep_01_kmer_LF.png         # LF (Leu-Phe) 疏水信号肽特征
    │       ├── dep_02_kmer_DD.png         # DD (Asp-Asp) 核定位负相关
    │       ├── dep_03_kmer_FF.png         # FF (Phe-Phe) 膜蛋白特征
    │       ├── dep_04_kmer_KR.png         # KR (Lys-Arg) 核定位信号
    │       └── dep_05_kmer_EE.png         # EE (Glu-Glu) 酸性特征
    │
    ├── step4_motif/                       # Motif生物学验证
    │   ├── PLAN.md                        # 验证方案设计
    │   ├── summary.txt                    # 验证结果摘要
    │   ├── motif_enrichment.csv           # 二肽Fisher精确检验结果
    │   ├── motif_enrichment.png           # 二肽富集热图
    │   ├── motif_uniprot_crosscheck.csv   # UniProt功能域交叉验证
    │   └── motif_validation_final.csv     # 最终综合验证表 (Confirmed/Novel/Partially)
    │
    ├── figures/                           # 最终论文图表（含白色背景版）
    │   ├── fig1_overview_heatmap.{png,svg,pdf}   # 类别特征热图
    │   ├── fig1a_workflow.{png,svg,pdf}          # 研究流程图
    │   ├── fig1b_heatmap.{png,svg,pdf}           # 编码×模型热图
    │   ├── fig2a_shap_global.{png,svg,pdf}       # SHAP全局蜂群图
    │   ├── fig2b_shap_perclass.{png,svg,pdf}     # SHAP各类别Top特征
    │   ├── fig2c_corroboration.{png,svg,pdf}     # SHAP-生物学佐证对照
    │   ├── fig2_shap_corroboration.{png,svg,pdf} # SHAP佐证完整图
    │   └── fig3_window_ablation.{png,svg,pdf}    # 窗口消融实验
    │
    └── logs/                              # 运行日志
        ├── refill_s123_e0.log             # Seed 123, 编码0 补充运行日志
        ├── refill_s123_e1.log             # Seed 123, 编码1
        ├── refill_s123_e2.log             # Seed 123, 编码2
        ├── refill_s456_e0.log             # Seed 456, 编码0
        ├── refill_s456_e1.log             # Seed 456, 编码1
        ├── refill_s456_e2.log             # Seed 456, 编码2
        ├── refill_s789_e0.log             # Seed 789, 编码0
        ├── refill_s789_e1.log             # Seed 789, 编码1
        └── refill_s789_e2.log             # Seed 789, 编码2
```

## 主要结果

### 实验矩阵 (21组 × 3次重复 = 63次独立训练)

| 编码方案 | LogReg | RF | XGBoost |
|---------|:------:|:---:|:-------:|
| k=1 AAC (20d) | 53.0% | 62.5% | 62.1% |
| k=2 DPC (400d) | 50.5% | 58.5% | 62.4% |
| k=3 TPC (8000d) | 36.1% | 55.5% | 59.2% |
| Fusion k1+k2 (420d) | 53.3% | 59.8% | **63.6%** ⭐ |
| Binary (24550d) | 28.7% | 35.8% | 34.5% |
| Integer (4910d) | 26.6% | 36.4% | 33.2% |
| One-hot (98200d) | 31.7% | 35.9% | 34.6% |

### 关键发现

- **k-mer频率编码远超位置特异性编码**: 20维AAC (62.1%) vs 98,200维One-hot (34.6%)
- **定位信号集中于N端**: 完整序列热图 G1（前半二肽）SHAP贡献 > G2（后半二肽）
- **SHAP与经典理论一致**: KR/RK → 核定位信号; LL/LF/FF → 信号肽疏水核心; DD → 线粒体靶向肽
- **窗口消融**: N+C端 (59.2%) > N200 (57.3%) > N100 (55.1%)，两端信号均有贡献

## 复现指南

### 环境要求

```bash
pip install pandas numpy biopython scikit-learn xgboost matplotlib seaborn shap --break-system-packages
```

### 运行

```bash
cd dataworkspace/
# 一键运行完整分析（需服务器，CPU密集）
bash scripts/run_all.sh
```

### 仅复现关键结果

```bash
# Step 0: 数据获取
python3 scripts/step0_get_data.py

# Step 1: 特征工程（32进程并行）
python3 scripts/step1_features.py

# Step 2: 模型训练（实验A/B/C + 完整矩阵）
python3 scripts/step2_models.py

# Step 3: SHAP可解释性分析
python3 scripts/step3_shap.py

# Step 4: Motif生物学验证
python3 scripts/step4_motif.py
```

## 技术栈

| 环节 | 工具/库 | 语言 |
|------|---------|:--:|
| 数据获取 | UniProt REST API, SGD REST API | Python |
| 特征工程 | scikit-learn, numpy | Python |
| 机器学习 | scikit-learn (RF/LogReg), XGBoost | Python |
| 可解释性 | SHAP (TreeExplainer) | Python |
| 统计检验 | scipy (Fisher exact test) | Python |
| 可视化 | matplotlib, seaborn | Python |

## 数据分析声明

本研究的数据处理、特征工程、模型训练、可解释性分析和图表生成均使用 Python 脚本自动化完成。所有分析脚本、中间数据和最终结果均在本仓库中公开。AI协助（OpenClaw/DeepSeek）主要用于代码开发、调试和论文润色，所有科学决策和最终判断由研究者本人完成。
