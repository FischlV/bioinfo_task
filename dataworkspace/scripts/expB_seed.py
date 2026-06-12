#!/usr/bin/env python3
"""Experiment B: Encoding comparison with given seed. Usage: python3 expB_seed.py <seed>"""
import numpy as np, pandas as pd, json, sys
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

seed = int(sys.argv[1])
DATA = Path("/data2/hyh/yeast_promoter_project/dataworkspace")
STEP1 = DATA / "step1_features"
N_JOBS = 96

labels_df = pd.read_csv(STEP1/"labels.csv").set_index("gene_id")
feat_k1 = pd.read_csv(STEP1/"features_k1.csv", index_col=0)
feat_k2 = pd.read_csv(STEP1/"features_k2.csv", index_col=0)
feat_k3 = pd.read_csv(STEP1/"features_k3.csv", index_col=0)
Xoh = np.load(STEP1/"features_onehot.npy")
t_idx = np.load(STEP1/"train_idx.npy"); e_idx = np.load(STEP1/"test_idx.npy")

cids = sorted(set(labels_df.index) & set(feat_k1.index))
id2i = {g:i for i,g in enumerate(cids)}
garr = np.array(cids)
tm = np.array([id2i[g] for g in garr[t_idx] if g in id2i])
em = np.array([id2i[g] for g in garr[e_idx] if g in id2i])
y_tr = labels_df.loc[cids,"location_encoded"].values[tm].astype(np.int32)
y_te = labels_df.loc[cids,"location_encoded"].values[em].astype(np.int32)
idx_list = [id2i[g] for g in cids]

Xk1 = feat_k1.loc[cids].values.astype(np.float32)
Xk2 = feat_k2.loc[cids].values.astype(np.float32)
Xk3 = feat_k3.loc[cids].values.astype(np.float32)
Xoh_flat = Xoh[idx_list].reshape(len(cids), -1).astype(np.float32)
Xint = np.argmax(Xoh, axis=-1).astype(np.float32)[idx_list]
aa_idx = np.argmax(Xoh, axis=-1).astype(np.uint8)[idx_list]
n, sl = aa_idx.shape
Xbin = np.zeros((n, sl*5), dtype=np.float32)
for b in range(5): Xbin[:,b::5] = (aa_idx>>b)&1

xp = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
          subsample=0.8, colsample_bytree=0.8, n_jobs=N_JOBS, random_state=seed, verbosity=0)

results = {}
for name, Xraw in [
    ("k=1 AAC (20d)", Xk1), ("k=2 DPC (400d)", Xk2), ("k=3 TPC (8000d)", Xk3),
    ("Binary enc (5000d)", Xbin), ("Integer enc (1000d)", Xint),
    ("One-hot flatten (20000d)", Xoh_flat),
]:
    m = xgb.XGBClassifier(**xp)
    m.fit(Xraw[tm], y_tr); yp = m.predict(Xraw[em])
    results[name] = {"Acc": accuracy_score(y_te,yp), "F1": f1_score(y_te,yp,average="macro")}

print(f"EXP_B seed={seed}", flush=True)
for name, v in results.items():
    print(f"  {name}: Acc={v['Acc']:.4f}, F1={v['F1']:.4f}", flush=True)

with open(DATA/"step2_models"/f"expB_seed{seed}.json","w") as f:
    json.dump(results, f)
