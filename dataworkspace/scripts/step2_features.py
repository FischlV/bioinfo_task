#!/usr/bin/env python3
"""Step 2: 特征工程 — k-mer 频率 + One-hot + 辅助特征 (多进程加速, FPKM标签)"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
import os, sys

# === 配置 (脚本在 dataworkspace/scripts/ 下) ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STEP1 = PROJECT_ROOT / "dataworkspace" / "step1_preprocess"
OUT = PROJECT_ROOT / "dataworkspace" / "step2_features"
OUT.mkdir(parents=True, exist_ok=True)

N_WORKERS = 32
print(f"CPU cores: {cpu_count()}, using {N_WORKERS} workers")

# === 1. 加载数据 ===
print("Loading expression data...")
expr_df = pd.read_csv(STEP1 / "expression.csv")
print(f"Gene count: {len(expr_df)}")

sequences = dict(zip(expr_df["gene_id"], expr_df["promoter_seq"]))

# 用 FPKM (D=glucose 条件下的三个重复)
fpkm_cols = ["D_1_fpkm", "D_2_fpkm", "D_3_fpkm"]
expr_df["mean_fpkm"] = expr_df[fpkm_cols].mean(axis=1)
# log2 变换
expr_df["log2_fpkm"] = np.log2(expr_df["mean_fpkm"] + 1)

mean_fpkm = dict(zip(expr_df["gene_id"], expr_df["mean_fpkm"]))
log2_fpkm = dict(zip(expr_df["gene_id"], expr_df["log2_fpkm"]))
gene_ids = expr_df["gene_id"].tolist()

print(f"FPKM range: [{expr_df['mean_fpkm'].min():.2f}, {expr_df['mean_fpkm'].max():.2f}]")
print(f"log2(FPKM+1) range: [{expr_df['log2_fpkm'].min():.2f}, {expr_df['log2_fpkm'].max():.2f}]")

# === 2. k-mer 频率计算 (多进程) ===
BASES = ["A", "T", "G", "C"]

def generate_kmers(k):
    return ["".join(p) for p in product(BASES, repeat=k)]

def count_kmer_freq_single(args):
    seq, k, kmer_to_idx, n_kmers = args
    freq = np.zeros(n_kmers)
    seq_len = len(seq)
    for i in range(seq_len - k + 1):
        kmer = seq[i:i+k]
        idx = kmer_to_idx.get(kmer, -1)
        if idx >= 0:
            freq[idx] += 1
    total = seq_len - k + 1
    if total > 0:
        freq /= total
    return freq

def build_feature_matrix_parallel(ids, seqs, k):
    kmers = generate_kmers(k)
    kmer_to_idx = {km: i for i, km in enumerate(kmers)}
    n_kmers = len(kmers)
    print(f"  k={k}, {n_kmers} features, {len(ids)} sequences, {N_WORKERS} workers...")
    args_list = [(seqs.get(gid, ""), k, kmer_to_idx, n_kmers) for gid in ids]
    with Pool(N_WORKERS) as pool:
        results = pool.map(count_kmer_freq_single, args_list, chunksize=max(1, len(ids) // (N_WORKERS * 4)))
    X = np.array(results)
    columns = [f"kmer_{km}" for km in kmers]
    df = pd.DataFrame(X, index=ids, columns=columns)
    print(f"  k={k} done: {df.shape}")
    return df

for k in [3, 4, 5]:
    print(f"\nBuilding k={k} features...")
    df_k = build_feature_matrix_parallel(gene_ids, sequences, k)
    df_k.to_csv(OUT / f"features_k{k}.csv")
    with open(OUT / f"kmers_k{k}.txt", "w") as f:
        f.write("\n".join(df_k.columns))

# === 3. Fusion ===
print("\nBuilding fusion features...")
df_k3 = pd.read_csv(OUT / "features_k3.csv", index_col=0)
df_k4 = pd.read_csv(OUT / "features_k4.csv", index_col=0)
df_k5 = pd.read_csv(OUT / "features_k5.csv", index_col=0)
common_ids = sorted(set(df_k3.index) & set(df_k4.index) & set(df_k5.index))
df_fusion = pd.concat([df_k3.loc[common_ids], df_k4.loc[common_ids], df_k5.loc[common_ids]], axis=1)
df_fusion.to_csv(OUT / "features_fusion.csv")
print(f"  Fusion: {df_fusion.shape}")

# === 4. One-hot ===
print("\nBuilding one-hot encoding...")
BASE_TO_IDX = {"A": 0, "T": 1, "G": 2, "C": 3}
seq_len = 850
X_onehot = np.zeros((len(gene_ids), seq_len, 4), dtype=np.float32)
for i, gid in enumerate(gene_ids):
    seq = sequences.get(gid, "")
    if len(seq) < seq_len:
        continue
    for j, base in enumerate(seq[:seq_len]):
        if base in BASE_TO_IDX:
            X_onehot[i, j, BASE_TO_IDX[base]] = 1.0
np.save(OUT / "features_onehot.npy", X_onehot)
print(f"  One-hot shape: {X_onehot.shape}")

# === 5. 辅助特征 ===
print("\nBuilding auxiliary features...")
def gc_content(seq):
    seq = seq.upper()
    return sum(1 for b in seq if b in "GC") / max(len(seq), 1)

def has_tata(seq, window=600):
    for p in ["TATAAAA", "TATATAA", "TATATAT", "TATAATA"]:
        if p in seq[:window]:
            return 1
    return 0

aux_data = [{"gene_id": gid, "gc_content": gc_content(sequences.get(gid, "")),
             "tata_present": has_tata(sequences.get(gid, ""))} for gid in gene_ids]
aux_df = pd.DataFrame(aux_data).set_index("gene_id")
aux_df.to_csv(OUT / "features_aux.csv")
print(f"  Aux features: {aux_df.shape}")

# === 6. 标签 (FPKM) + 数据划分 ===
print("\nBuilding labels (FPKM-based) and train/test split...")
labels_data = [{
    "gene_id": gid,
    "mean_fpkm": mean_fpkm.get(gid, 0),
    "log2_fpkm_plus1": log2_fpkm.get(gid, 0),
    "gc_content": aux_df.loc[gid, "gc_content"] if gid in aux_df.index else 0,
    "tata_present": aux_df.loc[gid, "tata_present"] if gid in aux_df.index else 0,
} for gid in gene_ids]

labels_df = pd.DataFrame(labels_data)
all_ids = labels_df["gene_id"].tolist()
train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
labels_df["split"] = labels_df["gene_id"].apply(lambda x: "train" if x in train_ids else "test")
labels_df.to_csv(OUT / "labels.csv", index=False)

train_indices = np.array([all_ids.index(g) for g in train_ids])
test_indices = np.array([all_ids.index(g) for g in test_ids])
np.save(OUT / "train_idx.npy", train_indices)
np.save(OUT / "test_idx.npy", test_indices)
print(f"  Train: {len(train_indices)}, Test: {len(test_indices)}")

# === 7. Summary ===
with open(OUT / "summary.txt", "w") as f:
    f.write("Step 2 特征工程\n" + "=" * 50 + "\n")
    f.write(f"总基因数: {len(gene_ids)}\n")
    f.write(f"标签: log2(mean_FPKM + 1), 来源: GSE301747 D(glucose) 3 replicates\n")
    f.write(f"FPKM range: [{expr_df['mean_fpkm'].min():.2f}, {expr_df['mean_fpkm'].max():.2f}]\n")
    f.write(f"k=3: {df_k3.shape[1]} dim, k=4: {df_k4.shape[1]} dim, k=5: {df_k5.shape[1]} dim\n")
    f.write(f"Fusion: {df_fusion.shape[1]} dim, One-hot: {X_onehot.shape}\n")
    f.write(f"训练集: {len(train_indices)}, 测试集: {len(test_indices)}\n")
    f.write(f"多进程: {N_WORKERS} workers\n")

print(f"\nDone! Output -> {OUT}")
