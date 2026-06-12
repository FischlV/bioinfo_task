#!/usr/bin/env python3
"""Step 2: 模型训练 — A/B/CNN 并行, C 等待"""

import pandas as pd, numpy as np, pickle, warnings, os, sys
from pathlib import Path
from multiprocessing import Pool, Process
warnings.filterwarnings("ignore")

N_JOBS = 96
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_ROOT / "dataworkspace"
STEP1 = DATA / "step1_features"
OUT = DATA / "step2_models"
OUT.mkdir(parents=True, exist_ok=True)

from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 数据加载 (主进程做，传给子进程)
# ============================================================
print("Loading data...")
labels_df = pd.read_csv(STEP1/"labels.csv").set_index("gene_id")
t_idx = np.load(STEP1/"train_idx.npy"); e_idx = np.load(STEP1/"test_idx.npy")
classes = np.load(STEP1/"label_encoder.npy", allow_pickle=True)

feat_k2 = pd.read_csv(STEP1/"features_k2.csv", index_col=0)
feat_k1 = pd.read_csv(STEP1/"features_k1.csv", index_col=0)
Xoh_raw = np.load(STEP1/"features_onehot.npy")

cids = sorted(set(labels_df.index) & set(feat_k2.index))
id2i = {g:i for i,g in enumerate(cids)}
garr = np.array(cids)
tm = np.array([id2i[g] for g in garr[t_idx] if g in id2i])
em = np.array([id2i[g] for g in garr[e_idx] if g in id2i])

feat_k2 = feat_k2.loc[cids]; feat_k1 = feat_k1.loc[cids]
y_all = labels_df.loc[cids,"location_encoded"].values.astype(np.int32)
idx_list = [id2i[g] for g in cids]

# 所有数据包好
DATA_PKG = {
    "Xk2_tr": feat_k2.values.astype(np.float32)[tm],
    "Xk2_te": feat_k2.values.astype(np.float32)[em],
    "Xk1_tr": feat_k1.values.astype(np.float32)[tm],
    "Xk1_te": feat_k1.values.astype(np.float32)[em],
    "Xoh_tr": Xoh_raw[idx_list].reshape(len(cids),-1)[tm].astype(np.float32),
    "Xoh_te": Xoh_raw[idx_list].reshape(len(cids),-1)[em].astype(np.float32),
    "Xoh3_tr": Xoh_raw[idx_list][tm],
    "Xoh3_te": Xoh_raw[idx_list][em],
    "y_tr": y_all[tm], "y_te": y_all[em],
    "classes": classes, "STEP1": str(STEP1), "OUT": str(OUT),
    "N_JOBS": N_JOBS, "cids": cids, "tm": tm, "em": em,
}
print(f"Train: {len(tm)}, Test: {len(em)}, Classes: {len(classes)}")

# ============================================================
# 实验 A + B + CNN (三个独立函数, 并行跑)
# ============================================================

def run_experiment_A(pkg):
    out = Path(pkg["OUT"])
    Xtr, Xte = pkg["Xk2_tr"], pkg["Xk2_te"]
    y_tr, y_te = pkg["y_tr"], pkg["y_te"]
    nj = pkg["N_JOBS"]

    print("[实验A] 模型对比 (k=2 DPC)")
    lr = LogisticRegression(multi_class="multinomial",max_iter=1000,C=1.0,n_jobs=nj,random_state=42)
    lr.fit(Xtr,y_tr); yp=lr.predict(Xte)
    rA = {"LogReg":{"Acc":accuracy_score(y_te,yp),"F1":f1_score(y_te,yp,average="macro")}}
    print(f"  LogReg: Acc={rA['LogReg']['Acc']:.4f}, F1={rA['LogReg']['F1']:.4f}")

    rf = RandomForestClassifier(n_estimators=500,max_depth=20,min_samples_split=5,n_jobs=nj,random_state=42)
    rf.fit(Xtr,y_tr); yp=rf.predict(Xte)
    rA["RF"] = {"Acc":accuracy_score(y_te,yp),"F1":f1_score(y_te,yp,average="macro")}
    print(f"  RF: Acc={rA['RF']['Acc']:.4f}, F1={rA['RF']['F1']:.4f}")

    xm = xgb.XGBClassifier(n_estimators=500,max_depth=6,learning_rate=0.05,subsample=0.8,
                            colsample_bytree=0.8,n_jobs=nj,random_state=42)
    xm.fit(Xtr,y_tr); yp=xm.predict(Xte)
    rA["XGBoost"] = {"Acc":accuracy_score(y_te,yp),"F1":f1_score(y_te,yp,average="macro")}
    print(f"  XGBoost: Acc={rA['XGBoost']['Acc']:.4f}, F1={rA['XGBoost']['F1']:.4f}")

    with open(out/"model_lr.pkl","wb") as f: pickle.dump(lr,f)
    with open(out/"model_rf.pkl","wb") as f: pickle.dump(rf,f)
    with open(out/"model_xgb.pkl","wb") as f: pickle.dump(xm,f)

    df = pd.DataFrame(rA).T
    fig,ax=plt.subplots(figsize=(6,4)); x=np.arange(len(df)); w=0.3
    ax.bar(x-w/2,df["Acc"],w,label="Accuracy",color="#3498db")
    ax.bar(x+w/2,df["F1"],w,label="F1 Macro",color="#e74c3c")
    ax.set_xticks(x); ax.set_xticklabels(df.index); ax.set_ylabel("Score")
    ax.set_title("实验A: 模型对比 (k=2 DPC)",fontweight="bold")
    ax.legend(); ax.set_ylim(0,1.05); ax.grid(axis="y",alpha=0.3)
    for i,(a,f) in enumerate(zip(df["Acc"],df["F1"])):
        ax.text(i-w/2,a,f"{a:.3f}",ha="center",va="bottom",fontsize=9)
        ax.text(i+w/2,f,f"{f:.3f}",ha="center",va="bottom",fontsize=9)
    plt.tight_layout(); plt.savefig(out/"expA_model_comparison.png",dpi=150); plt.close()
    print("[实验A] 完成")
    return rA

def run_experiment_B(pkg):
    out = Path(pkg["OUT"])
    y_tr, y_te = pkg["y_tr"], pkg["y_te"]
    nj = pkg["N_JOBS"]

    print("[实验B] 位置/顺序消融 (RF)")
    rB = {}
    for name, Xtr, Xte in [("k=1 AAC (纯组成)", pkg["Xk1_tr"], pkg["Xk1_te"]),
                             ("One-hot flatten (完整位置)", pkg["Xoh_tr"], pkg["Xoh_te"])]:
        rf = RandomForestClassifier(n_estimators=300,max_depth=15,n_jobs=nj,random_state=42)
        rf.fit(Xtr,y_tr); yp=rf.predict(Xte)
        rB[name] = {"Acc":accuracy_score(y_te,yp),"F1":f1_score(y_te,yp,average="macro")}
        print(f"  {name}: Acc={rB[name]['Acc']:.4f}, F1={rB[name]['F1']:.4f}")

    df = pd.DataFrame(rB).T
    fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(len(df)); w=0.3
    ax.bar(x-w/2,df["Acc"],w,label="Accuracy",color="#3498db")
    ax.bar(x+w/2,df["F1"],w,label="F1 Macro",color="#e74c3c")
    ax.set_xticks(x); ax.set_xticklabels(df.index,fontsize=9)
    ax.set_ylabel("Score"); ax.set_title("实验B: 位置/顺序重要吗? (RF)",fontweight="bold")
    ax.legend(); ax.set_ylim(0,1.05); ax.grid(axis="y",alpha=0.3)
    for i,(a,f) in enumerate(zip(df["Acc"],df["F1"])):
        ax.text(i-w/2,a,f"{a:.3f}",ha="center",va="bottom",fontsize=9)
        ax.text(i+w/2,f,f"{f:.3f}",ha="center",va="bottom",fontsize=9)
    plt.tight_layout(); plt.savefig(out/"expB_position_ablation.png",dpi=150); plt.close()
    print("[实验B] 完成")
    return rB

def run_CNN(pkg):
    out = Path(pkg["OUT"])
    print("[CNN] One-hot 3D")

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    Xtr, Xte = pkg["Xoh3_tr"], pkg["Xoh3_te"]
    y_tr, y_te = pkg["y_tr"], pkg["y_te"]
    n_cls = len(pkg["classes"])

    yc_tr = keras.utils.to_categorical(y_tr, n_cls)
    yc_te = keras.utils.to_categorical(y_te, n_cls)

    m = keras.Sequential([
        layers.Conv1D(64,5,padding="same"),layers.BatchNormalization(),layers.Activation("relu"),
        layers.MaxPooling1D(4),layers.Dropout(0.2),
        layers.Conv1D(128,3,padding="same"),layers.BatchNormalization(),layers.Activation("relu"),
        layers.MaxPooling1D(4),layers.Dropout(0.2),
        layers.Conv1D(256,3,padding="same"),layers.BatchNormalization(),layers.Activation("relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(128,activation="relu"),layers.Dropout(0.3),
        layers.Dense(n_cls,activation="softmax"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3),loss="categorical_crossentropy",metrics=["accuracy"])

    es = keras.callbacks.EarlyStopping(monitor="val_loss",patience=15,restore_best_weights=True)
    rl = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=5,min_lr=1e-6)
    h = m.fit(Xtr,yc_tr,validation_data=(Xte,yc_te),batch_size=64,epochs=100,
              callbacks=[es,rl],verbose=1)

    yp = np.argmax(m.predict(Xte),axis=1)
    acc = accuracy_score(y_te,yp); f = f1_score(y_te,yp,average="macro")
    print(f"  CNN: Acc={acc:.4f}, F1={f:.4f}")
    m.save(out/"model_cnn.keras")

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,k in zip(axes,["loss","accuracy"]):
        ax.plot(h.history[k],label="Train"); ax.plot(h.history[f"val_{k}"],label="Val")
        ax.set_xlabel("Epoch"); ax.set_ylabel(k); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title(f"CNN {k}")
    plt.tight_layout(); plt.savefig(out/"cnn_training_curve.png",dpi=150); plt.close()
    print("[CNN] 完成")
    return {"Acc": acc, "F1": f}

# ============================================================
# Main: A/B/CNN 并行 → C 继续
# ============================================================
if __name__ == "__main__":
    pA = Process(target=lambda: run_experiment_A(DATA_PKG))
    pB = Process(target=lambda: run_experiment_B(DATA_PKG))
    pD = Process(target=lambda: run_CNN(DATA_PKG))

    print("\n>>> 并行启动 实验A | 实验B | CNN...")
    pA.start(); pB.start(); pD.start()
    pA.join(); pB.join(); pD.join()
    print(">>> A/B/CNN 全部完成\n")

    # 实验 C: 序列窗口消融 (等A/B/CNN跑完后)
    print("="*60+"\n实验C: 序列窗口消融 (RF + k=2 DPC, Pool并行)\n"+"="*60)
    y_tr, y_te = DATA_PKG["y_tr"], DATA_PKG["y_te"]
    STEP1_path = Path(DATA_PKG["STEP1"])

    wfiles = {"full":"features_k2_full.csv","N100":"features_k2_N100.csv",
              "N200":"features_k2_N200.csv","NC":"features_k2_NC.csv",
              "mid200":"features_k2_mid200.csv"}

    def train_window(args):
        wn, ff = args
        wf = pd.read_csv(STEP1_path/ff,index_col=0).loc[DATA_PKG["cids"]]
        Xtr = wf.values[DATA_PKG["tm"]].astype(np.float32)
        Xte = wf.values[DATA_PKG["em"]].astype(np.float32)
        rw = RandomForestClassifier(n_estimators=300,max_depth=15,n_jobs=16,random_state=42)
        rw.fit(Xtr,y_tr); yp=rw.predict(Xte)
        a=accuracy_score(y_te,yp); f=f1_score(y_te,yp,average="macro")
        print(f"  [{wn}] Acc={a:.4f}, F1={f:.4f}")
        return {"window":wn,"Accuracy":a,"F1_macro":f}

    with Pool(len(wfiles)) as pool:
        rowsC = pool.map(train_window, list(wfiles.items()))
    dfC = pd.DataFrame(rowsC)

    fig,ax=plt.subplots(figsize=(6,4)); x=np.arange(len(dfC)); w=0.3
    ax.bar(x-w/2,dfC["Accuracy"],w,label="Accuracy",color="#3498db")
    ax.bar(x+w/2,dfC["F1_macro"],w,label="F1 Macro",color="#e74c3c")
    ax.set_xticks(x); ax.set_xticklabels(dfC["window"]); ax.set_ylabel("Score")
    ax.set_title("实验C: 定位信号集中在哪里?",fontweight="bold")
    ax.legend(); ax.set_ylim(0,1.05); ax.grid(axis="y",alpha=0.3)
    for i,(a,f) in enumerate(zip(dfC["Accuracy"],dfC["F1_macro"])):
        ax.text(i-w/2,a,f"{a:.3f}",ha="center",va="bottom",fontsize=9)
        ax.text(i+w/2,f,f"{f:.3f}",ha="center",va="bottom",fontsize=9)
    plt.tight_layout(); plt.savefig(OUT/"expC_window_ablation.png",dpi=150); plt.close()

    # ============================================================
    # Summary
    # ============================================================
    # 从磁盘读回模型评估 (A/B 在子进程里跑的，结果没传回来)
    Xk2_tr, Xk2_te = DATA_PKG["Xk2_tr"], DATA_PKG["Xk2_te"]
    Xk1_tr, Xk1_te = DATA_PKG["Xk1_tr"], DATA_PKG["Xk1_te"]
    Xoh_tr, Xoh_te = DATA_PKG["Xoh_tr"], DATA_PKG["Xoh_te"]

    with open(OUT/"model_lr.pkl","rb") as f: lr=pickle.load(f)
    with open(OUT/"model_rf.pkl","rb") as f: rf=pickle.load(f)
    with open(OUT/"model_xgb.pkl","rb") as f: xm=pickle.load(f)

    expA = {
        "LogReg":{"Acc":accuracy_score(y_te,lr.predict(Xk2_te)),"F1":f1_score(y_te,lr.predict(Xk2_te),average="macro")},
        "RF":{"Acc":accuracy_score(y_te,rf.predict(Xk2_te)),"F1":f1_score(y_te,rf.predict(Xk2_te),average="macro")},
        "XGBoost":{"Acc":accuracy_score(y_te,xm.predict(Xk2_te)),"F1":f1_score(y_te,xm.predict(Xk2_te),average="macro")},
    }

    expB = {}
    for name, Xtr, Xte in [("k=1 AAC", Xk1_tr, Xk1_te), ("One-hot flatten", Xoh_tr, Xoh_te)]:
        rfb = RandomForestClassifier(n_estimators=300,max_depth=15,n_jobs=N_JOBS,random_state=42)
        rfb.fit(Xtr,y_tr); yp=rfb.predict(Xte)
        expB[name] = {"Acc":accuracy_score(y_te,yp),"F1":f1_score(y_te,yp,average="macro")}

    # CNN from disk
    import tensorflow as tf
    cnn_m = tf.keras.models.load_model(OUT/"model_cnn.keras")
    yp_cnn = np.argmax(cnn_m.predict(DATA_PKG["Xoh3_te"]),axis=1)
    cnn_acc = accuracy_score(y_te,yp_cnn); cnn_f1 = f1_score(y_te,yp_cnn,average="macro")

    print("\n"+"="*60+"\n汇总\n"+"="*60)
    with open(OUT/"summary.txt","w") as f:
        f.write("Step 2 实验汇总\n"+"="*60+"\n\n")
        f.write("实验A: 模型对比 (k=2 DPC)\n")
        for m,v in expA.items(): f.write(f"  {m}: Acc={v['Acc']:.4f}, F1={v['F1']:.4f}\n")
        f.write("\n实验B: 位置/顺序消融 (RF)\n")
        for m,v in expB.items(): f.write(f"  {m}: Acc={v['Acc']:.4f}, F1={v['F1']:.4f}\n")
        f.write("\n实验C: 序列窗口消融 (RF + k=2 DPC)\n")
        for _,r in dfC.iterrows(): f.write(f"  {r['window']}: Acc={r['Accuracy']:.4f}, F1={r['F1_macro']:.4f}\n")
        f.write(f"\nCNN (one-hot 3D, 独立): Acc={cnn_acc:.4f}, F1={cnn_f1:.4f}\n")
        f.write(f"\n类别: {list(classes)}\n训练集: {len(tm)}, 测试集: {len(em)}\n")

    print(open(OUT/"summary.txt").read())
    print("Done!")
