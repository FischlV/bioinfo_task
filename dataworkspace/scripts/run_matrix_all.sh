#!/bin/bash
export PYTHONUNBUFFERED=1
cd /data2/hyh/yeast_promoter_project/dataworkspace
for seed in 123 456 789; do
    echo "=== seed $seed ==="
    python3 scripts/fill_matrix.py $seed
    echo ""
done
echo "ALL DONE"
