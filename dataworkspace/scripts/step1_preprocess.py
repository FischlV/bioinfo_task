#!/usr/bin/env python3
"""Step 1: 数据预处理 — 提取启动子序列 + 匹配表达量"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import sys

# === 配置 (脚本在 dataworkspace/scripts/ 下) ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ORIGIN = PROJECT_ROOT / "origin_datas"
OUT = PROJECT_ROOT / "dataworkspace" / "step1_preprocess"
OUT.mkdir(parents=True, exist_ok=True)

GFF_FILE = ORIGIN / "genomic.gff"
GENOME_FA = ORIGIN / "GCF_000146045.2_R64_genomic.fna"
RNA_SEQ_FILE = ORIGIN / "GSE301747_RNAseq.csv"

UPSTREAM = 800
DOWNSTREAM = 50
PROMOTER_LEN = UPSTREAM + DOWNSTREAM  # 850bp

# === 1. 解析 GFF，获取每个基因的 TSS ===
print("Loading GFF...")
genes = []
with open(GFF_FILE) as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split("\t")
        if len(parts) < 9:
            continue
        if parts[2].lower() == "gene":
            chrom = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attr_raw = parts[8]

            # Parse attributes with regex (handle both key=value and key "value")
            import re
            attrs = {}
            for m in re.finditer(r'(\w+)=([^;]+)', attr_raw):
                attrs[m.group(1)] = m.group(2).strip()

            gene_id = attrs.get("locus_tag")
            gene_name = attrs.get("gene") or attrs.get("Name") or gene_id
            gene_biotype = attrs.get("gene_biotype", "")

            if gene_id is None:
                continue

            # TSS: start for + strand, end for - strand
            tss = start if strand == "+" else end

            genes.append({
                "gene_id": gene_id,
                "gene_name": gene_name or gene_id,
                "chrom": chrom,
                "tss": tss,
                "strand": strand,
                "gene_biotype": gene_biotype or "",
            })

genes_df = pd.DataFrame(genes)
# 去重（每个基因多个 CDS 取第一个）
genes_df = genes_df.drop_duplicates(subset="gene_id", keep="first")
print(f"GFF 总基因数: {len(genes_df)}")

# === 2. 加载基因组序列 ===
print("Loading genome FASTA...")
genome = {}
current_chrom = None
current_seq = []
with open(GENOME_FA) as f:
    for line in f:
        if line.startswith(">"):
            if current_chrom is not None:
                genome[current_chrom] = "".join(current_seq)
            header = line[1:].strip().split()[0]
            current_chrom = header
            current_seq = []
        else:
            current_seq.append(line.strip())
    if current_chrom is not None:
        genome[current_chrom] = "".join(current_seq)

print(f"基因组染色体数: {len(genome)}")

# === 3. 截取启动子序列 ===
def extract_promoter(chrom, tss, strand, genome_dict):
    """从基因组截取 TSS 上游 UPSTREAM + 下游 DOWNSTREAM"""
    chrom_seq = genome_dict.get(chrom)
    if chrom_seq is None:
        return None

    if strand == "+":
        # + 链: 启动子在 TSS 上游 (坐标更小)
        # 提取 [TSS-800, TSS+49] (1-based) → 0-based: [TSS-801, TSS+48]
        start_pos = tss - UPSTREAM - 1  # 0-based start
        end_pos = tss + DOWNSTREAM - 2  # 0-based inclusive end
    else:
        # - 链: 启动子在 TSS 上游 (坐标更大)
        # 提取 [TSS-49, TSS+800] (1-based) → 0-based: [TSS-50, TSS+799]
        start_pos = tss - DOWNSTREAM  # 0-based start
        end_pos = tss + UPSTREAM - 1  # 0-based inclusive end

    if start_pos < 0 or end_pos >= len(chrom_seq):
        return None

    seq = chrom_seq[start_pos:end_pos + 1].upper()

    if strand == "-":
        # 反向互补，使序列方向与转录方向一致
        comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
        seq = "".join(comp.get(b, "N") for b in reversed(seq))

    return seq


print("Extracting promoters...")
promoters = []
for _, row in genes_df.iterrows():
    seq = extract_promoter(row["chrom"], row["tss"], row["strand"], genome)
    if seq is not None and len(seq) == PROMOTER_LEN:
        promoters.append({
            "gene_id": row["gene_id"],
            "gene_name": row["gene_name"],
            "chrom": row["chrom"],
            "tss": row["tss"],
            "strand": row["strand"],
            "promoter_seq": seq,
        })

prom_df = pd.DataFrame(promoters)
print(f"成功截取启动子: {len(prom_df)}")

# === 4. 加载 RNA-seq 数据 ===
print("Loading RNA-seq...")
rna_df = pd.read_csv(RNA_SEQ_FILE)
print(f"RNA-seq 基因数: {len(rna_df)}")

# 计算三个 replicate 的 mean count
count_cols = ["D_1_count", "D_2_count", "D_3_count"]
rna_df["mean_count"] = rna_df[count_cols].mean(axis=1)

# 提取需要的列
rna_slim = rna_df[["gene_id", "mean_count",
                     "D_1_count", "D_2_count", "D_3_count",
                     "D_1_fpkm", "D_2_fpkm", "D_3_fpkm",
                     "gene_name", "gene_chr", "gene_start", "gene_end",
                     "gene_strand", "gene_length"]].copy()

# === 5. 合并 ===
# Drop duplicate columns from prom_df before merge (rna_slim has gene_name etc.)
prom_for_merge = prom_df.drop(columns=["gene_name"], errors="ignore")
merged = prom_for_merge.merge(rna_slim, on="gene_id", how="inner")
print(f"合并后基因数: {len(merged)}")

# 过滤低表达 (mean_count >= 10)
merged = merged[merged["mean_count"] >= 10].copy()
print(f"过滤 (count>=10) 后: {len(merged)}")

# === 6. 输出 ===
# promoters.fa
print("Writing promoters.fa...")
with open(OUT / "promoters.fa", "w") as f:
    for _, row in merged.iterrows():
        f.write(f">{row['gene_id']}|{row['gene_name']}|{row['chrom']}:{row['tss']}:{row['strand']}\n")
        # 每行 80 个碱基
        seq = row["promoter_seq"]
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")

# expression.csv
print("Writing expression.csv...")
expr_cols = ["gene_id", "gene_name", "chrom", "tss", "strand", "gene_length",
             "mean_count", "D_1_count", "D_2_count", "D_3_count",
             "D_1_fpkm", "D_2_fpkm", "D_3_fpkm", "promoter_seq"]
merged[expr_cols].to_csv(OUT / "expression.csv", index=False)

# summary.txt
with open(OUT / "summary.txt", "w") as f:
    f.write(f"Step 1 数据预处理\n")
    f.write(f"{'='*50}\n")
    f.write(f"GFF 总基因数: {len(genes_df)}\n")
    f.write(f"成功截取启动子: {len(prom_df)}\n")
    f.write(f"RNA-seq 基因数: {len(rna_df)}\n")
    f.write(f"合并后基因数: {len(merged)}\n")
    f.write(f"过滤 (count>=10) 后: {len(merged)}\n")
    f.write(f"启动子长度: {PROMOTER_LEN}bp (上游{UPSTREAM}bp + 下游{DOWNSTREAM}bp)\n")
    f.write(f"表达量单位: raw count (3 replicates)\n")
    f.write(f"菌株: BY4741, 条件: glucose (SC medium)\n")
    f.write(f"基因组: S288c R64 (GCF_000146045.2)\n")

print(f"\nDone! Output -> {OUT}")
