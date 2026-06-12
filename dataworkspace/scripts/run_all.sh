#!/bin/bash
# shopt:set bombastic astic
# 酵母启动子强度预测 — 一键运行
# Usage: bash run_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================"
echo "  酵母启动子强度预测 — 自动化流程"
echo "============================================"
echo "  项目目录: $PROJECT_DIR"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# Step 1: 数据预处理
echo ""
echo ">>> Step 1: 数据预处理"
python3 "$SCRIPT_DIR/step1_preprocess.py"

# Step 2: 特征工程
echo ""
echo ">>> Step 2: 特征工程"
python3 "$SCRIPT_DIR/step2_features.py"

# Step 3: 模型训练 (CPU heavy)
echo ""
echo ">>> Step 3: 模型训练 (⚠ CPU 密集)"
python3 "$SCRIPT_DIR/step3_models.py"

# Step 4: SHAP 解释
echo ""
echo ">>> Step 4: SHAP 可解释性分析"
python3 "$SCRIPT_DIR/step4_shap.py"

echo ""
echo "============================================"
echo "  全部完成!"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
