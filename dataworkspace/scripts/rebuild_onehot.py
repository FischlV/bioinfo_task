
import numpy as np, pandas as pd
from pathlib import Path

DATA = Path("/data2/hyh/yeast_promoter_project/dataworkspace")
STEP0 = DATA / "step0_data"
STEP1 = DATA / "step1_features"
MAX_LEN = 4910
AAS = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {aa: i for i, aa in enumerate(AAS)}

# Load sequences
labels_df = pd.read_csv(STEP1 / "labels.csv").set_index("gene_id")
seq_df = pd.read_csv(STEP0 / "protein_labels.csv")

gene_ids = sorted(set(labels_df.index) & set(seq_df["gene_id"]))
print(f"Genes: {len(gene_ids)}, MAX_LEN={MAX_LEN}")

# One-hot encode with new max_len
X_onehot = np.zeros((len(gene_ids), MAX_LEN, 20), dtype=np.float32)
seq_map = dict(zip(seq_df["gene_id"], seq_df["sequence"])) if "sequence" in seq_df.columns else None

# Try loading from FASTA if sequence column not available
if seq_map is None or len(seq_map) == 0:
    from Bio import SeqIO
    seq_map = {}
    for rec in SeqIO.parse(str(STEP0 / "protein_sequences.fa"), "fasta"):
        gene_id = rec.id.split("|")[0]
        seq_map[gene_id] = str(rec.seq)
    print(f"Loaded {len(seq_map)} sequences from FASTA")

for i, gid in enumerate(gene_ids):
    seq = seq_map.get(gid, "")
    for j, aa in enumerate(seq[:MAX_LEN]):
        if aa in aa_to_idx:
            X_onehot[i, j, aa_to_idx[aa]] = 1.0
    if i % 1000 == 0:
        print(f"  {i}/{len(gene_ids)}")

# Save
np.save(STEP1 / "features_onehot.npy", X_onehot)
print(f"Saved onehot: {X_onehot.shape}")
print(f"Flattened dim: {X_onehot.shape[1] * X_onehot.shape[2]}")

