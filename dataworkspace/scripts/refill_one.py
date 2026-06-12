#!/usr/bin/env python3
"""Run one encoding+seed combo. Usage: refill_one.py <seed> <encoding_index>"""
import numpy as np, pandas as pd, json, sys, time, warnings
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
warnings.filterwarnings("ignore")

seed = int(sys.argv[1])
enc_idx = int(sys.argv[2])  # 0=Binary, 1=Integer, 2=Onehot
DATA = Path("/data2/hyh/yeast_promoter_project/dataworkspace")
STEP1 = DATA / "step1_features"
OUT = DATA / "step2_models"
N_JOBS = 4  # lower per-job parallelism

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

# Build discrete encodings
Xoh_flat = Xoh[idx_list].reshape(len(cids), -1).astype(np.float32)
aa_idx = np.argmax(Xoh, axis=-1).astype(np.uint8)[idx_list]
n, sl = aa_idx.shape
Xbin = np.zeros((n, sl*5), dtype=np.float32)
for b in range(5):
    Xbin[:, b::5] = (aa_idx >> b) & 1
Xint = np.argmax(Xoh, axis=-1).astype(np.float32)[idx_list]

encodings = [
    ("Binary enc", Xbin),
    ("Integer enc", Xint),
    ("One-hot flatten", Xoh_flat),
]

enc_name, Xf = encodings[enc_idx]
d = Xf.shape[1]
Xtr, Xte = Xf[tm], Xf[em]
print(f"Seed={seed} Enc={enc_name} Dim={d} | Train={Xtr.shape} Test={Xte.shape}")

def run_logreg(Xtr, Xte, yt, ye):
    C = 0.1 if Xtr.shape[1] > 4000 else 1.0
    m = LogisticRegression(multi_class="multinomial", max_iter=2000, C=C, solver="saga", n_jobs=N_JOBS, random_state=seed)
    m.fit(Xtr, yt)
    return float(accuracy_score(ye, m.predict(Xte)))

def run_rf(Xtr, Xte, yt, ye):
    m = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=N_JOBS, random_state=seed)
    m.fit(Xtr, yt)
    return float(accuracy_score(ye, m.predict(Xte)))

def run_xgb(Xtr, Xte, yt, ye):
    m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="multi:softprob", eval_metric="mlogloss", n_jobs=N_JOBS, random_state=seed, verbosity=0)
    m.fit(Xtr, yt)
    return float(accuracy_score(ye, m.predict(Xte)))

results = {}
t0 = time.time()
results[f"LogReg|{enc_name}"] = {"Acc": run_logreg(Xtr, Xte, y_tr, y_te)}
print(f"  LogReg: {results[f'LogReg|{enc_name}']['Acc']:.4f} ({time.time()-t0:.0f}s)")

t0 = time.time()
results[f"RF|{enc_name}"] = {"Acc": run_rf(Xtr, Xte, y_tr, y_te)}
print(f"  RF: {results[f'RF|{enc_name}']['Acc']:.4f} ({time.time()-t0:.0f}s)")

t0 = time.time()
results[f"XGBoost|{enc_name}"] = {"Acc": run_xgb(Xtr, Xte, y_tr, y_te)}
print(f"  XGBoost: {results[f'XGBoost|{enc_name}']['Acc']:.4f} ({time.time()-t0:.0f}s)")

# Read existing matrix and update
with open(OUT / f"matrix_seed{seed}.json") as f:
    existing = json.load(f)
existing.update(results)
with open(OUT / f"matrix_seed{seed}.json", 'w') as f:
    json.dump(existing, f, indent=2)
print(f"DONE seed={seed} enc={enc_name} | Updated matrix_seed{seed}.json")
