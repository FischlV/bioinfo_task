#!/usr/bin/env python3
"""Experiment A: Model comparison with given seed. Usage: python3 expA_seed.py <seed>"""
import numpy as np, pandas as pd, pickle, sys
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

seed = int(sys.argv[1])
DATA = Path("/data2/hyh/yeast_promoter_project/dataworkspace")
STEP1 = DATA / "step1_features"
N_JOBS = 96

labels_df = pd.read_csv(STEP1/"labels.csv").set_index("gene_id")
feat_k2 = pd.read_csv(STEP1/"features_k2.csv", index_col=0)
t_idx = np.load(STEP1/"train_idx.npy"); e_idx = np.load(STEP1/"test_idx.npy")

cids = sorted(set(labels_df.index) & set(feat_k2.index))
id2i = {g:i for i,g in enumerate(cids)}
garr = np.array(cids)
tm = np.array([id2i[g] for g in garr[t_idx] if g in id2i])
em = np.array([id2i[g] for g in garr[e_idx] if g in id2i])
y_tr = labels_df.loc[cids,"location_encoded"].values[tm].astype(np.int32)
y_te = labels_df.loc[cids,"location_encoded"].values[em].astype(np.int32)
Xtr = feat_k2.loc[cids].values.astype(np.float32)[tm]
Xte = feat_k2.loc[cids].values.astype(np.float32)[em]

results = {}
# LogReg
lr = LogisticRegression(multi_class="multinomial", max_iter=1000, C=1.0, n_jobs=N_JOBS, random_state=seed)
lr.fit(Xtr, y_tr); yp = lr.predict(Xte)
results["LogReg"] = {"Acc": accuracy_score(y_te,yp), "F1": f1_score(y_te,yp,average="macro")}

# RF
rf = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=5, n_jobs=N_JOBS, random_state=seed)
rf.fit(Xtr, y_tr); yp = rf.predict(Xte)
results["RF"] = {"Acc": accuracy_score(y_te,yp), "F1": f1_score(y_te,yp,average="macro")}

# XGBoost
xp = dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
          colsample_bytree=0.8, n_jobs=N_JOBS, random_state=seed, verbosity=0)
xm = xgb.XGBClassifier(**xp)
xm.fit(Xtr, y_tr); yp = xm.predict(Xte)
results["XGBoost"] = {"Acc": accuracy_score(y_te,yp), "F1": f1_score(y_te,yp,average="macro")}

print(f"EXP_A seed={seed}", flush=True)
for m, v in results.items():
    print(f"  {m}: Acc={v['Acc']:.4f}, F1={v['F1']:.4f}", flush=True)

# Save as JSON for aggregation
import json
with open(DATA/"step2_models"/f"expA_seed{seed}.json","w") as f:
    json.dump(results, f)
