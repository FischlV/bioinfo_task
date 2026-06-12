#!/usr/bin/env python3 -u
#!/usr/bin/env python3
"""补全矩阵: 所有 encoding × 所有 model, 3 seeds. Usage: python3 fill_matrix.py <seed>"""
import numpy as np, pandas as pd, json, sys, time, warnings
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

warnings.filterwarnings("ignore")

seed = int(sys.argv[1])
DATA = Path("/data2/hyh/yeast_promoter_project/dataworkspace")
STEP1 = DATA / "step1_features"
OUT = DATA / "step2_models"
N_JOBS = 16  # 保守点, 96核全开可能OOM

# ====== 加载数据 ======
labels_df = pd.read_csv(STEP1/"labels.csv").set_index("gene_id")
feat_k1 = pd.read_csv(STEP1/"features_k1.csv", index_col=0)
feat_k2 = pd.read_csv(STEP1/"features_k2.csv", index_col=0)
feat_k3 = pd.read_csv(STEP1/"features_k3.csv", index_col=0)
feat_fusion = pd.read_csv(STEP1/"features_fusion.csv", index_col=0)
Xoh = np.load(STEP1/"features_onehot.npy")
t_idx = np.load(STEP1/"train_idx.npy")
e_idx = np.load(STEP1/"test_idx.npy")

# 对齐
cids = sorted(set(labels_df.index) & set(feat_k1.index))
id2i = {g:i for i,g in enumerate(cids)}
garr = np.array(cids)
tm = np.array([id2i[g] for g in garr[t_idx] if g in id2i])
em = np.array([id2i[g] for g in garr[e_idx] if g in id2i])
y_tr = labels_df.loc[cids,"location_encoded"].values[tm].astype(np.int32)
y_te = labels_df.loc[cids,"location_encoded"].values[em].astype(np.int32)
idx_list = [id2i[g] for g in cids]

# ====== 构建所有 encoding ======
Xk1 = feat_k1.loc[cids].values.astype(np.float32)
Xk2 = feat_k2.loc[cids].values.astype(np.float32)
Xk3 = feat_k3.loc[cids].values.astype(np.float32)
Xfusion = feat_fusion.loc[cids].values.astype(np.float32)

# 全位置编码 (拍平的三种)
Xoh_flat = Xoh[idx_list].reshape(len(cids), -1).astype(np.float32)
Xint = np.argmax(Xoh, axis=-1).astype(np.float32)[idx_list]
aa_idx = np.argmax(Xoh, axis=-1).astype(np.uint8)[idx_list]
n, sl = aa_idx.shape
Xbin = np.zeros((n, sl*5), dtype=np.float32)
for b in range(5): Xbin[:,b::5] = (aa_idx>>b)&1

encodings = [
    ("k=1 AAC (20d)", Xk1),
    ("k=2 DPC (400d)", Xk2),
    ("k=3 TPC (8000d)", Xk3),
    ("Fusion k1+k2 (420d)", Xfusion),
    ("Binary enc (5000d)", Xbin),
    ("Integer enc (1000d)", Xint),
    ("One-hot flatten (20000d)", Xoh_flat),
]

# ====== 模型定义 ======
def run_logreg(Xtr, Xte, y_tr, y_te):
    """L2 LogisticRegression, saga solver for high-dim"""
    t0 = time.time()
    C = 0.1 if Xtr.shape[1] > 4000 else 1.0
    m = LogisticRegression(multi_class="multinomial", max_iter=2000, C=C,
                           solver="saga", n_jobs=N_JOBS, random_state=seed)
    m.fit(Xtr, y_tr)
    yp = m.predict(Xte)
    dt = time.time() - t0
    return {"Acc": accuracy_score(y_te, yp), "F1": f1_score(y_te, yp, average="macro"), "time_s": dt}

def run_rf(Xtr, Xte, y_tr, y_te):
    t0 = time.time()
    m = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=5,
                                n_jobs=N_JOBS, random_state=seed)
    m.fit(Xtr, y_tr)
    yp = m.predict(Xte)
    dt = time.time() - t0
    return {"Acc": accuracy_score(y_te, yp), "F1": f1_score(y_te, yp, average="macro"), "time_s": dt}

def run_xgb(Xtr, Xte, y_tr, y_te):
    t0 = time.time()
    xp = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, n_jobs=N_JOBS,
              tree_method="hist", random_state=seed, verbosity=0)
    m = xgb.XGBClassifier(**xp)
    m.fit(Xtr, y_tr)
    yp = m.predict(Xte)
    dt = time.time() - t0
    return {"Acc": accuracy_score(y_te, yp), "F1": f1_score(y_te, yp, average="macro"), "time_s": dt}

models = [
    ("LogReg", run_logreg),
    ("RF", run_rf),
    ("XGBoost", run_xgb),
]

# ====== 跑矩阵 ======
print(f"seed={seed} | {len(encodings)} encodings × {len(models)} models", flush=True)
print(f"Train: {len(tm)}, Test: {len(em)}", flush=True)

results = {}
for enc_name, Xraw in encodings:
    d = Xraw.shape[1]
    print(f"\n[{enc_name}]", flush=True)
    for model_name, model_fn in models:
        try:
            r = model_fn(Xraw[tm], Xraw[em], y_tr, y_te)
            results[f"{model_name}|{enc_name}"] = r
            print(f"  {model_name}: Acc={r['Acc']:.4f}, F1={r['F1']:.4f} ({r['time_s']:.0f}s)", flush=True)
        except Exception as e:
            print(f"  {model_name}: FAILED - {e}", flush=True)
            results[f"{model_name}|{enc_name}"] = {"Acc": None, "F1": None, "error": str(e)}

# 保存
outfile = OUT / f"matrix_seed{seed}.json"
with open(outfile, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {outfile}", flush=True)
