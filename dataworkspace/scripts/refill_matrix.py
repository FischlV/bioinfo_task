#!/usr/bin/env python3
"""Redo matrix for discrete encodings with corrected onehot (4910 max_len)."""
import numpy as np, pandas as pd, json, sys, time, warnings
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
warnings.filterwarnings("ignore")

seed = int(sys.argv[1])
DATA = Path("/data2/hyh/yeast_promoter_project/dataworkspace")
STEP1 = DATA / "step1_features"
OUT = DATA / "step2_models"
N_JOBS = 16

labels_df = pd.read_csv(STEP1/"labels.csv").set_index("gene_id")
Xoh = np.load(STEP1/"features_onehot.npy")
t_idx = np.load(STEP1/"train_idx.npy")
e_idx = np.load(STEP1/"test_idx.npy")

cids = sorted(set(labels_df.index))
id2i = {g: i for i, g in enumerate(cids)}
garr = np.array(cids)
tm = np.array([id2i[g] for g in garr[t_idx] if g in id2i])
em = np.array([id2i[g] for g in garr[e_idx] if g in id2i])
y_tr = labels_df.loc[cids, "location_encoded"].values[tm].astype(np.int32)
y_te = labels_df.loc[cids, "location_encoded"].values[em].astype(np.int32)
idx_list = [id2i[g] for g in cids]

# Build discrete encodings with NEW onehot
Xoh_flat = Xoh[idx_list].reshape(len(cids), -1).astype(np.float32)
Xint = np.argmax(Xoh, axis=-1).astype(np.float32)[idx_list]
aa_idx = np.argmax(Xoh, axis=-1).astype(np.uint8)[idx_list]
n, sl = aa_idx.shape
Xbin = np.zeros((n, sl*5), dtype=np.float32)
for b in range(5):
    Xbin[:, b::5] = (aa_idx >> b) & 1

print(f"Onehot: {Xoh_flat.shape[1]}d, Binary: {Xbin.shape[1]}d, Integer: {Xint.shape[1]}d")

encodings_local = [
    ("Binary enc", Xbin),
    ("Integer enc", Xint),
    ("One-hot flatten", Xoh_flat),
]

def run_logreg(Xtr, Xte, yt, ye):
    C = 0.1 if Xtr.shape[1] > 4000 else 1.0
    m = LogisticRegression(multi_class="multinomial", max_iter=2000, C=C, solver="saga", n_jobs=N_JOBS, random_state=seed)
    m.fit(Xtr, yt)
    pred = m.predict(Xte)
    return float(accuracy_score(ye, pred))

def run_rf(Xtr, Xte, yt, ye):
    m = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=N_JOBS, random_state=seed)
    m.fit(Xtr, yt)
    return float(accuracy_score(ye, m.predict(Xte)))

def run_xgb(Xtr, Xte, yt, ye):
    m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="multi:softprob", eval_metric="mlogloss", n_jobs=N_JOBS, random_state=seed, verbosity=0)
    m.fit(Xtr, yt)
    return float(accuracy_score(ye, m.predict(Xte)))

results = {}
for enc_name, Xf in encodings_local:
    Xtr, Xte = Xf[tm], Xf[em]
    print(f"\n=== {enc_name} ({Xf.shape[1]}d), seed={seed} ===")
    
    t0 = time.time()
    acc_lr = run_logreg(Xtr, Xte, y_tr, y_te)
    print(f"  LogReg: {acc_lr:.4f} ({time.time()-t0:.0f}s)")
    
    t0 = time.time()
    acc_rf = run_rf(Xtr, Xte, y_tr, y_te)
    print(f"  RF: {acc_rf:.4f} ({time.time()-t0:.0f}s)")
    
    t0 = time.time()
    acc_xgb = run_xgb(Xtr, Xte, y_tr, y_te)
    print(f"  XGBoost: {acc_xgb:.4f} ({time.time()-t0:.0f}s)")
    
    results[f"LogReg|{enc_name}"] = {"Acc": acc_lr}
    results[f"RF|{enc_name}"] = {"Acc": acc_rf}
    results[f"XGBoost|{enc_name}"] = {"Acc": acc_xgb}

# Load existing matrix and update
existing_file = OUT / f"matrix_seed{seed}.json"
if existing_file.exists():
    with open(existing_file) as f:
        existing = json.load(f)
else:
    existing = {}

# Update with new results (overwrite old discrete encoding entries)
for k, v in results.items():
    existing[k] = v

with open(existing_file, 'w') as f:
    json.dump(existing, f, indent=2)

print(f"\nUpdated {existing_file}")
print(f"Total entries: {len(existing)}")
