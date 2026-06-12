#!/usr/bin/env python3
"""Step 3: 模型训练 — RF / XGBoost / CNN"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
import os
import sys

warnings.filterwarnings("ignore")

# 限制线程数，防止把服务器跑满
N_JOBS = 32  # 留余量，不占满服务器
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)

# === 配置 (脚本在 dataworkspace/scripts/ 下) ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_ROOT / "dataworkspace"
STEP2 = DATA / "step2_features"
OUT = DATA / "step3_models"
OUT.mkdir(parents=True, exist_ok=True)

# === 1. 加载数据 ===
print("Loading features and labels...")
labels_df = pd.read_csv(STEP2 / "labels.csv")
features_df = pd.read_csv(STEP2 / "features_k4.csv", index_col=0)  # k=4 主特征
train_idx = np.load(STEP2 / "train_idx.npy")
test_idx = np.load(STEP2 / "test_idx.npy")

# 对齐 gene_id
common_ids = sorted(set(labels_df["gene_id"]) & set(features_df.index))
features_df = features_df.loc[common_ids]
labels_df = labels_df[labels_df["gene_id"].isin(common_ids)].set_index("gene_id").loc[common_ids]

X = features_df.values.astype(np.float32)
y = labels_df["log2_fpkm_plus1"].values.astype(np.float32)
gene_ids = list(features_df.index)

# 重新映射索引
id_to_idx = {g: i for i, g in enumerate(gene_ids)}
train_idx_mapped = np.array([id_to_idx[g] for g in np.array(gene_ids)[train_idx] if g in id_to_idx])
test_idx_mapped = np.array([id_to_idx[g] for g in np.array(gene_ids)[test_idx] if g in id_to_idx])

X_train, X_test = X[train_idx_mapped], X[test_idx_mapped]
y_train, y_test = y[train_idx_mapped], y[test_idx_mapped]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Features: {X.shape[1]}, y range: [{y.min():.2f}, {y.max():.2f}]")

# === 2. 评估函数 ===
def evaluate(y_true, y_pred, name=""):
    from scipy.stats import pearsonr, spearmanr
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r, p_r = pearsonr(y_true, y_pred)
    rho, p_rho = spearmanr(y_true, y_pred)
    print(f"  {name}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, r={r:.4f}, ρ={rho:.4f}")
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "Pearson_r": r, "Pearson_p": p_r, "Spearman_rho": rho, "Spearman_p": p_rho}

results = {}

# === 3. Random Forest ===
print("\n=== Random Forest ===")
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    n_jobs=N_JOBS,
    random_state=42,
    verbose=1,
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results["Random Forest"] = evaluate(y_test, y_pred_rf, "RF")

# Save
with open(OUT / "model_rf.pkl", "wb") as f:
    pickle.dump(rf, f)
print("  Saved: model_rf.pkl")

# === 4. XGBoost ===
print("\n=== XGBoost ===")
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=N_JOBS,
    random_state=42,
    verbosity=1,
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
results["XGBoost"] = evaluate(y_test, y_pred_xgb, "XGB")

# Save
with open(OUT / "model_xgb.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
print("  Saved: model_xgb.pkl")

# === 5. CNN ===
print("\n=== CNN ===")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 设置 GPU 内存增长 (如果有 GPU)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"  GPU(s): {len(gpus)}")
else:
    print("  No GPU found, using CPU")

# Load one-hot data
X_onehot = np.load(STEP2 / "features_onehot.npy")
print(f"  One-hot: {X_onehot.shape}")

X_cnn_train = X_onehot[train_idx_mapped]
X_cnn_test = X_onehot[test_idx_mapped]

# Build model
def build_cnn(input_shape=(850, 4)):
    model = keras.Sequential([
        layers.Conv1D(64, kernel_size=8, padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling1D(pool_size=4),
        layers.Dropout(0.2),

        layers.Conv1D(128, kernel_size=4, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling1D(pool_size=4),
        layers.Dropout(0.2),

        layers.Conv1D(256, kernel_size=4, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.GlobalMaxPooling1D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1),
    ])
    return model

cnn = build_cnn()
cnn.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
cnn.summary()

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

# Train
history = cnn.fit(
    X_cnn_train, y_train,
    validation_data=(X_cnn_test, y_test),
    batch_size=64,
    epochs=100,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

y_pred_cnn = cnn.predict(X_cnn_test, batch_size=128).flatten()
results["CNN"] = evaluate(y_test, y_pred_cnn, "CNN")

# Save
cnn.save(OUT / "model_cnn.keras")
print("  Saved: model_cnn.keras")

# Training curve plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.plot(history.history["loss"], label="Train Loss")
ax.plot(history.history["val_loss"], label="Val Loss")
ax.set_xlabel("Epoch"), ax.set_ylabel("MSE Loss")
ax.set_title("CNN Training Curve")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(history.history["mae"], label="Train MAE")
ax.plot(history.history["val_mae"], label="Val MAE")
ax.set_xlabel("Epoch"), ax.set_ylabel("MAE")
ax.set_title("CNN MAE")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "cnn_training_curve.png", dpi=150)
plt.close()
print("  Saved: cnn_training_curve.png")

# === 6. 预测结果汇总 ===
print("\n=== Saving predictions... ===")
test_gene_ids = np.array(gene_ids)[test_idx_mapped]
pred_df = pd.DataFrame({
    "gene_id": test_gene_ids,
    "y_true": y_test,
    "y_pred_rf": y_pred_rf,
    "y_pred_xgb": y_pred_xgb,
    "y_pred_cnn": y_pred_cnn,
})
pred_df.to_csv(OUT / "predictions.csv", index=False)

# === 7. 模型对比 ===
print("\n=== Model Comparison ===")
comp_rows = []
for name, metrics in results.items():
    comp_rows.append({
        "model": name,
        "R2": metrics["R2"],
        "RMSE": metrics["RMSE"],
        "Pearson_r": metrics["Pearson_r"],
        "Pearson_p": metrics["Pearson_p"],
        "Spearman_rho": metrics["Spearman_rho"],
        "Spearman_p": metrics["Spearman_p"],
    })
comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(OUT / "model_comparison.csv", index=False)
print(comp_df)

# Comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
models = [r["model"] for r in comp_rows]

for ax, metric, title in zip(
    axes,
    ["R2", "RMSE", "Pearson_r"],
    ["R² (higher is better)", "RMSE (lower is better)", "Pearson r (higher is better)"]
):
    values = [r[metric] for r in comp_rows]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    bars = ax.bar(models, values, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "model_comparison.png", dpi=150)
plt.close()
print("  Saved: model_comparison.png")

# === 8. Predicted vs Actual scatter ===
print("\n=== Prediction vs Actual plots ===")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
pred_pairs = [
    (y_pred_rf, "Random Forest", axes[0]),
    (y_pred_xgb, "XGBoost", axes[1]),
    (y_pred_cnn, "CNN", axes[2]),
]
for y_pred, model_name, ax in pred_pairs:
    r2 = results[model_name]["R2"]
    r = results[model_name]["Pearson_r"]
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="#3498db", edgecolors="none")
    # y=x line
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Actual log2(FPKM+1)")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name}\nR²={r2:.3f}, r={r:.3f}", fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "pred_vs_actual.png", dpi=150)
plt.close()
print("  Saved: pred_vs_actual.png")

# === 9. Summary ===
with open(OUT / "summary.txt", "w") as f:
    f.write("Step 3 模型训练与评估\n")
    f.write("=" * 60 + "\n")
    f.write(f"训练集: {len(train_idx_mapped)}, 测试集: {len(test_idx_mapped)}\n")
    f.write(f"主特征: k=4 k-mer ({X.shape[1]} 维)\n")
    f.write(f"标签: log2(mean_FPKM+1), D(glucose) 3 replicates\n\n")
    f.write("模型性能 (测试集):\n")
    for _, row in comp_df.iterrows():
        f.write(f"  {row['model']}: R²={row['R2']:.4f}, RMSE={row['RMSE']:.4f}, r={row['Pearson_r']:.4f}, ρ={row['Spearman_rho']:.4f}\n")
    best = comp_df.loc[comp_df["R2"].idxmax()]
    f.write(f"\n最佳模型: {best['model']} (R²={best['R2']:.4f})\n")

print(f"\nDone! Output -> {OUT}")
