#!/usr/bin/env python3
"""Step 0: 从 UniProt 获取酵母蛋白序列 + 亚细胞定位标签"""

import json
import subprocess
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "step0_data"
OUT.mkdir(parents=True, exist_ok=True)

# === 1. Download from UniProt ===
print("Downloading yeast proteome from UniProt...")
url = "https://rest.uniprot.org/uniprotkb/stream?query=%28proteome:UP000002311%29&format=json&fields=accession,gene_names,cc_subcellular_location,sequence&size=500"
result = subprocess.run(['curl', '-s', url], capture_output=True, text=True, timeout=120)
data = json.loads(result.stdout)
results = data.get('results', [])
print(f"Downloaded: {len(results)} proteins")

# === 2. Parse subcellular locations ===
# Map fine-grained terms to broad categories
LOCATION_MAP = {
    # Nucleus
    'Nucleus': 'Nucleus',
    'Nucleus, nucleolus': 'Nucleus',
    'Nucleus membrane': 'Nucleus',
    'Chromosome': 'Nucleus',
    'Nuclear pore complex': 'Nucleus',
    'Nuclear matrix': 'Nucleus',
    # Cytoplasm
    'Cytoplasm': 'Cytoplasm',
    'Cytoplasm, cytosol': 'Cytoplasm',
    'Cytoplasm, cytoskeleton': 'Cytoplasm',
    'Cytoplasm, P-body': 'Cytoplasm',
    'Cytoplasm, Stress granule': 'Cytoplasm',
    'Cytoplasmic granule': 'Cytoplasm',
    'Bud neck': 'Cytoplasm',
    'Bud': 'Cytoplasm',
    'Bud tip': 'Cytoplasm',
    'Spindle pole body': 'Cytoplasm',
    'Spindle': 'Cytoplasm',
    # Mitochondria
    'Mitochondrion': 'Mitochondria',
    'Mitochondrion inner membrane': 'Mitochondria',
    'Mitochondrion outer membrane': 'Mitochondria',
    'Mitochondrion matrix': 'Mitochondria',
    'Mitochondrion membrane': 'Mitochondria',
    'Mitochondrion intermembrane space': 'Mitochondria',
    # Membrane / Secretory pathway
    'Membrane': 'Membrane_Secretory',
    'Cell membrane': 'Membrane_Secretory',
    'Endoplasmic reticulum membrane': 'Membrane_Secretory',
    'Endoplasmic reticulum': 'Membrane_Secretory',
    'Golgi apparatus membrane': 'Membrane_Secretory',
    'Golgi apparatus': 'Membrane_Secretory',
    'Vacuole membrane': 'Membrane_Secretory',
    'Vacuole': 'Membrane_Secretory',
    'Endosome membrane': 'Membrane_Secretory',
    'Endosome': 'Membrane_Secretory',
    'Secreted, cell wall': 'Membrane_Secretory',
    'Secreted': 'Membrane_Secretory',
    'Cell wall': 'Membrane_Secretory',
    'Lipid droplet': 'Membrane_Secretory',
    'Peroxisome': 'Membrane_Secretory',
    'Peroxisome membrane': 'Membrane_Secretory',
}

# Priority: more specific organelles > generic membrane/cytoplasm
LOCATION_PRIORITY = ['Nucleus', 'Mitochondria', 'Membrane_Secretory', 'Cytoplasm']

def classify_location(subcellular_locations):
    """Map UniProt subcellular locations to broad category.
    For multi-localized proteins, use priority order."""
    broad_locs = set()
    for sl in subcellular_locations:
        loc = sl.get('location', {}).get('value', '')
        if loc in LOCATION_MAP:
            broad_locs.add(LOCATION_MAP[loc])
    
    if not broad_locs:
        return 'Other'
    
    # Use highest priority location
    for priority in LOCATION_PRIORITY:
        if priority in broad_locs:
            return priority
    
    return 'Other'

proteins = []
loc_counter = Counter()
multi_count = 0

for r in results:
    acc = r.get('primaryAccession', '')
    seq = r.get('sequence', {}).get('value', '')
    
    # Extract gene names
    gene_name = ''
    sgd_id = ''
    for g in r.get('genes', []):
        gn = g.get('geneName', {})
        if gn:
            gene_name = gn.get('value', '')
        ols = g.get('orderedLocusNames', [])
        if ols:
            sgd_id = ols[0].get('value', '')
    
    if not seq or not sgd_id:
        continue
    
    # Get subcellular locations
    sub_locs = []
    for c in r.get('comments', []):
        if 'SUBCELLULAR' in str(c.get('commentType', '')):
            sub_locs = c.get('subcellularLocations', [])
    
    if not sub_locs:
        continue
    
    # Collect original locations for reference
    orig_locs = [sl.get('location', {}).get('value', '') for sl in sub_locs]
    
    # Classify
    broad_loc = classify_location(sub_locs)
    loc_counter[broad_loc] += 1
    
    if len(set(orig_locs)) > 1:
        multi_count += 1
    
    proteins.append({
        'uniprot_id': acc,
        'gene_id': sgd_id,
        'gene_name': gene_name,
        'sequence': seq,
        'length': len(seq),
        'location': broad_loc,
        'original_locations': '; '.join(orig_locs),
    })

df = pd.DataFrame(proteins)
print(f"\nProteins with location + SGD ID: {len(df)}")
print(f"Multi-localized: {multi_count}")
print(f"\nCategory distribution:")
for loc, count in loc_counter.most_common():
    print(f"  {loc}: {count} ({100*count/len(df):.1f}%)")

# === 3. Save ===
# Full sequences
with open(OUT / "protein_sequences.fa", "w") as f:
    for _, row in df.iterrows():
        f.write(f">{row['gene_id']}|{row['gene_name']}|{row['uniprot_id']}|{row['location']}\n")
        seq = row['sequence']
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")

# Metadata
df.drop(columns=['sequence']).to_csv(OUT / "protein_labels.csv", index=False)

# Summary
with open(OUT / "summary.txt", "w") as f:
    f.write("Step 0 数据获取\n" + "=" * 50 + "\n")
    f.write(f"来源: UniProt proteome UP000002311 (S. cerevisiae S288c)\n")
    f.write(f"总蛋白数: {len(results)}\n")
    f.write(f"有亚细胞定位: {len(df)}\n")
    f.write(f"多定位蛋白: {multi_count}\n")
    f.write(f"类别分布:\n")
    for loc, count in loc_counter.most_common():
        f.write(f"  {loc}: {count}\n")

print(f"\nDone! Output -> {OUT}")
print(f"  {OUT}/protein_sequences.fa")
print(f"  {OUT}/protein_labels.csv")
print(f"  {OUT}/summary.txt")
