#!/usr/bin/env python3
"""Step 1: 蛋白序列特征工程 — 氨基酸 k-mer + One-hot + 序列窗口"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
import os, sys

# === 配置 ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STEP0 = PROJECT_ROOT / "dataworkspace" / "step0_data"
OUT = PROJECT_ROOT / "dataworkspace" / "step1_features"
OUT.mkdir(parents=True, exist_ok=True)

N_WORKERS = 32
print(f"CPU cores: {cpu_count()}, using {N_WORKERS} workers")

# 20 种标准氨基酸
AAS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AAS)}

# === 1. 加载数据 ===
print("Loading protein data...")
labels_df = pd.read_csv(STEP0 / "protein_labels.csv")
print(f"Proteins: {len(labels_df)}")
print(f"Location distribution:\n{labels_df['location'].value_counts().to_string()}")

# Load sequences
sequences = {}
with open(STEP0 / "protein_sequences.fa") as f:
    current_id = None
    current_seq = []
    for line in f:
        if line.startswith(">"):
            if current_id:
                sequences[current_id] = "".join(current_seq)
            current_id = line[1:].strip().split("|")[0]
            current_seq = []
        else:
            current_seq.append(line.strip())
    if current_id:
        sequences[current_id] = "".join(current_seq)

gene_ids = labels_df["gene_id"].tolist()
print(f"Loaded {len(sequences)} sequences")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels_df["location"])
print(f"Classes: {list(le.classes_)}")
np.save(OUT / "label_encoder.npy", le.classes_)

# === 2. k-mer 频率 (k=1,2,3) ===
def generate_aa_kmers(k):
    return ["".join(p) for p in product(AAS, repeat=k)]

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

def build_kmer_features(ids, seqs, k, prefix=""):
    kmers = generate_aa_kmers(k)
    kmer_to_idx = {km: i for i, km in enumerate(kmers)}
    n_kmers = len(kmers)
    name = f"k{k}"
    print(f"  {name}: {n_kmers} features, {N_WORKERS} workers...")
    args_list = [(seqs.get(gid, ""), k, kmer_to_idx, n_kmers) for gid in ids]
    with Pool(N_WORKERS) as pool:
        results = pool.map(count_kmer_freq_single, args_list, chunksize=max(1, len(ids)//(N_WORKERS*4)))
    X = np.array(results)
    columns = [f"{prefix}kmer_{km}" for km in kmers]
    df = pd.DataFrame(X, index=ids, columns=columns)
    print(f"  {name} done: {df.shape}")
    return df

# k=1 (AAC: 20 dim) + k=2 (DPC: 400 dim)
for k in [1, 2]:
    print(f"\nBuilding k={k} features...")
    df_k = build_kmer_features(gene_ids, sequences, k)
    df_k.to_csv(OUT / f"features_k{k}.csv")
    with open(OUT / f"kmers_k{k}.txt", "w") as f:
        f.write("\n".join(df_k.columns))

# Fusion: k=1 + k=2 (420 dim, practical for 4863 samples)
df_k1 = pd.read_csv(OUT / "features_k1.csv", index_col=0)
df_k2 = pd.read_csv(OUT / "features_k2.csv", index_col=0)
df_fusion = pd.concat([df_k1, df_k2], axis=1)
df_fusion.to_csv(OUT / "features_fusion.csv")
print(f"\nFusion (k1+k2): {df_fusion.shape}")

# === 3. One-hot encoding (CNN input) ===
print("\nBuilding one-hot encoding...")
# Use max length 1000 for full sequence
MAX_LEN = 1000

def one_hot_encode(seq, max_len):
    X = np.zeros((max_len, 20), dtype=np.float32)
    for i, aa in enumerate(seq[:max_len]):
        if aa in AA_TO_IDX:
            X[i, AA_TO_IDX[aa]] = 1.0
    return X

X_onehot = np.zeros((len(gene_ids), MAX_LEN, 20), dtype=np.float32)
valid_mask = np.ones(len(gene_ids), dtype=bool)
for i, gid in enumerate(gene_ids):
    seq = sequences.get(gid, "")
    if len(seq) < 10:
        valid_mask[i] = False
        continue
    X_onehot[i] = one_hot_encode(seq, MAX_LEN)

np.save(OUT / "features_onehot.npy", X_onehot)
print(f"  One-hot shape: {X_onehot.shape} (L={MAX_LEN})")

# === 4. 序列窗口变体 (消融实验关键) ===
print("\nBuilding sequence window variants...")

def extract_window(seq, mode):
    """Extract sequence window: full, N100, N200, NC"""
    if mode == "full":
        return seq
    elif mode == "N100":
        return seq[:100]
    elif mode == "N200":
        return seq[:200]
    elif mode == "NC":
        return seq[:100] + seq[-100:] if len(seq) >= 200 else seq[:len(seq)//2] + seq[-len(seq)//2:]
    elif mode == "mid200":
        # Middle 200aa, skip N100 and C100
        if len(seq) >= 400:
            start = (len(seq) - 200) // 2
            return seq[start:start+200]
        elif len(seq) >= 200:
            start = (len(seq) - 200) // 2
            return seq[max(0,start):start+200]
        else:
            return seq
    return seq

window_modes = ["full", "N100", "N200", "NC", "mid200"]

for mode in window_modes:
    print(f"\n  Window: {mode}")
    # Extract windowed sequences
    window_seqs = {}
    for gid in gene_ids:
        window_seqs[gid] = extract_window(sequences.get(gid, ""), mode)
    
    # k=2 features for this window
    df_win = build_kmer_features(gene_ids, window_seqs, k=2, prefix=f"{mode}_")
    df_win.to_csv(OUT / f"features_k2_{mode}.csv")
    
    # One-hot for this window
    max_wlen = {"full": MAX_LEN, "N100": 100, "N200": 200, "NC": 200, "mid200": 200}[mode]
    X_win = np.zeros((len(gene_ids), max_wlen, 20), dtype=np.float32)
    for i, gid in enumerate(gene_ids):
        seq = window_seqs.get(gid, "")
        X_win[i] = one_hot_encode(seq, max_wlen)
    np.save(OUT / f"features_onehot_{mode}.npy", X_win)

# === 5. 辅助特征 ===
print("\nBuilding auxiliary features...")
aux_data = []
for gid in gene_ids:
    seq = sequences.get(gid, "")
    length = len(seq)
    # Simple amino acid composition features
    aa_counts = {aa: seq.count(aa) for aa in AAS}
    n_term_20 = [seq[:20].count(aa) for aa in AAS] if len(seq) >= 20 else [0]*20
    c_term_20 = [seq[-20:].count(aa) for aa in AAS] if len(seq) >= 20 else [0]*20
    
    aux_data.append({
        "gene_id": gid,
        "length": length,
        "log_length": np.log10(max(length, 1)),
        "n_term_hydrophobic": sum(seq[:50].count(aa) for aa in "AILMFWVP")/max(len(seq[:50]),1) if seq else 0,
        "n_term_positive": sum(seq[:50].count(aa) for aa in "KRH")/max(len(seq[:50]),1) if seq else 0,
        "n_term_aliphatic": sum(seq[:50].count(aa) for aa in "AILV")/max(len(seq[:50]),1) if seq else 0,
    })

aux_df = pd.DataFrame(aux_data).set_index("gene_id")
aux_df.to_csv(OUT / "features_aux.csv")
print(f"  Aux features: {aux_df.shape}")

# === 6. 标签 + 数据划分 ===
print("\nBuilding labels and train/test split...")
labels_out = pd.DataFrame({
    "gene_id": gene_ids,
    "location": labels_df["location"].values,
    "location_encoded": y,
    "split": "train",
})
labels_out = labels_out.set_index("gene_id")

# Stratified split
all_ids = labels_out.index.tolist()
train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=42, stratify=y)
labels_out.loc[train_ids, "split"] = "train"
labels_out.loc[test_ids, "split"] = "test"
labels_out.to_csv(OUT / "labels.csv")
print(f"  Train: {len(train_ids)}, Test: {len(test_ids)}")

# Check class balance per split
for split_name, split_ids in [("train", train_ids), ("test", test_ids)]:
    split_locs = labels_out.loc[split_ids, "location"].value_counts()
    print(f"  {split_name}: {dict(split_locs)}")

train_indices = np.array([all_ids.index(g) for g in train_ids])
test_indices = np.array([all_ids.index(g) for g in test_ids])
np.save(OUT / "train_idx.npy", train_indices)
np.save(OUT / "test_idx.npy", test_indices)

# === 7. Summary ===
with open(OUT / "summary.txt", "w") as f:
    f.write("Step 1 蛋白序列特征工程\n" + "=" * 50 + "\n")
    f.write(f"总蛋白数: {len(gene_ids)}\n")
    f.write(f"类别: {list(le.classes_)} ({len(le.classes_)} classes)\n")
    f.write(f"k=1 (AAC): 20 dim\nk=2 (DPC): 400 dim\nFusion: 420 dim\n")
    f.write(f"One-hot: ({len(gene_ids)}, {MAX_LEN}, 20)\n")
    f.write(f"序列窗口: {window_modes}\n")
    f.write(f"训练集: {len(train_ids)}, 测试集: {len(test_ids)}\n")
    f.write(f"多进程: {N_WORKERS} workers\n")

print(f"\nDone! Output -> {OUT}")
print("Ready to run? Say go!")
