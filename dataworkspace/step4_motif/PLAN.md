# Step 4: Motif 验证 — 实验计划

## 目标
将 SHAP 发现的关键二肽特征与以下来源进行系统性比对验证：
1. 各类别蛋白序列中的实际频率差异（统计检验）
2. SGD/UniProt 已知功能注释（信号肽、NLS、线粒体靶向肽等）
3. PROSITE/Pfam 已知 motif 数据库

## 输入
- 蛋白序列: step0_data/protein_sequences.fa
- 类别标签: step1_features/labels.csv
- SHAP Top 特征: step3_shap/shap_top_features_per_class.json
- SGD REST API: https://www.yeastgenome.org/backend

## 分析 1: 二肽频率富集分析
- 取每类 SHAP Top 10 二肽
- 统计该类 vs 其他类的频率差异
- Mann-Whitney U 检验 + Benjamini-Hochberg 多重检验校正
- 输出: motif_enrichment.csv + motif_barplot.png

## 分析 2: UniProt/SGD 注释对照
- 通过 SGD API 拉取信号肽/NLS/跨膜等注释
- 统计 SHAP 高重要性的二肽在有无该注释的蛋白中的频率
- 输出: motif_uniprot_crosscheck.csv

## 分析 3: 综合 Motif 验证表
- 将 SHAP 发现 + 频率富集 p 值 + UniProt 注释 + 已知文献整合为一张表
- 输出: motif_validation_final.csv + summary.txt

## 时间估算
- 频率计算: ~1 min
- SGD API 调用: ~10-20 min（~100 个查询）
- 绘图: ~2 min
- 总计: ~30 min

## 输出目录
step4_motif/
├── motif_enrichment.csv
├── motif_enrichment.png
├── motif_uniprot_crosscheck.csv
├── motif_validation_final.csv
├── nohup.out
└── summary.txt
