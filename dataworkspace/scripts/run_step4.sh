#!/bin/bash
cd /data2/hyh/yeast_promoter_project/dataworkspace
PYTHONUNBUFFERED=1 python3 scripts/step4_motif.py > step4_motif/nohup.out 2>&1
echo "DONE" >> step4_motif/nohup.out
