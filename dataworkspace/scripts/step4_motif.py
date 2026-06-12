#!/usr/bin/env python3
"""
Step 4: Motif Validation for Protein Subcellular Localization
=============================================================
Analyses:
  1. Dipeptide enrichment (Mann-Whitney U + FDR)
  2. SGD/UniProt annotation cross-check
  3. Integrated validation table
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import requests

def benjamini_hochberg(p_values, alpha=0.05):
    """Manual Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return np.array([]), np.array([])
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    # BH critical values
    ranks = np.arange(1, n + 1)
    bh_crit = ranks * alpha / n
    # Find largest k where p_(k) <= k*alpha/n
    below = sorted_p <= bh_crit
    if np.any(below):
        max_k = np.max(np.where(below)[0])
        reject = order <= max_k
    else:
        reject = np.zeros(n, dtype=bool)
    # Adjusted p-values: min(1, p * n / rank)
    p_adj = np.minimum(1, sorted_p * n / ranks)
    # Make monotone
    for i in range(n - 2, -1, -1):
        p_adj[i] = min(p_adj[i], p_adj[i + 1])
    # Restore original order
    p_adj_ordered = np.ones(n)
    p_adj_ordered[order] = p_adj
    return reject, p_adj_ordered

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────
BASE = "/data2/hyh/yeast_promoter_project/dataworkspace"
LABELS_CSV = f"{BASE}/step1_features/labels.csv"
FASTA_FILE = f"{BASE}/step0_data/protein_sequences.fa"
SHAP_JSON = f"{BASE}/step3_shap/shap_top_features_per_class.json"
OUT_DIR = f"{BASE}/step4_motif"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ['Cytoplasm', 'Membrane_Secretory', 'Mitochondria', 'Nucleus', 'Other']

# ── Helpers ────────────────────────────────────────────────
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_labels():
    """Load labels CSV, return DataFrame with gene_id and location."""
    df = pd.read_csv(LABELS_CSV)
    log(f"Loaded labels: {df.shape[0]} rows, classes: {df['location'].value_counts().to_dict()}")
    return df

def load_fasta():
    """Load protein sequences from FASTA; returns dict gene_id -> sequence."""
    seqs = {}
    current_id = None
    current_seq = []
    with open(FASTA_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    seqs[current_id] = ''.join(current_seq)
                # Parse header: >gene_id|...|...|location
                parts = line[1:].split('|')
                current_id = parts[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id and current_seq:
            seqs[current_id] = ''.join(current_seq)
    log(f"Loaded {len(seqs)} protein sequences")
    return seqs

def load_shap_top():
    """Load SHAP top features JSON."""
    with open(SHAP_JSON) as f:
        data = json.load(f)
    # data is dict: class_name -> [[feature, importance], ...]
    result = {}
    for cls_name, features in data.items():
        # Strip "kmer_" prefix, keep only dipeptide features
        cleaned = []
        for feat, imp in features:
            name = feat.replace("kmer_", "")
            if len(name) == 2 and name.isalpha():
                cleaned.append((name, imp))
        result[cls_name] = cleaned
        log(f"  {cls_name}: {len(cleaned)} dipeptides loaded")
    return result

# ── Analysis 1: Dipeptide Enrichment ───────────────────────
def compute_dipeptide_frequencies(sequences, labels_df):
    """
    For each protein, compute the frequency of all dipeptides.
    Returns DataFrame: gene_id × dipeptides, values = count/length
    """
    genes = []
    rows = []
    
    # Get all unique dipeptides from SHAP data
    all_dipeps = set()
    for cls_name, feats in shap_data.items():
        for dp, _ in feats:
            all_dipeps.add(dp)
    all_dipeps = sorted(all_dipeps)
    log(f"Total unique dipeptides across all classes: {len(all_dipeps)}")
    
    for gene_id, seq in sequences.items():
        if gene_id not in labels_df.index:
            continue
        seq_len = len(seq)
        if seq_len < 2:
            continue
        freqs = {}
        for i in range(seq_len - 1):
            dp = seq[i:i+2]
            if dp in all_dipeps:
                freqs[dp] = freqs.get(dp, 0) + 1
        for dp in all_dipeps:
            freqs.setdefault(dp, 0)
        # Normalize by sequence length
        for dp in freqs:
            freqs[dp] = freqs[dp] / seq_len
        genes.append(gene_id)
        rows.append(freqs)
    
    freq_df = pd.DataFrame(rows, index=genes)
    return freq_df

def dipeptide_enrichment(labels_df, seqs, shap_data):
    """Run dipeptide enrichment analysis with Mann-Whitney U + FDR."""
    log("=" * 60)
    log("ANALYSIS 1: Dipeptide Enrichment")
    log("=" * 60)
    
    # Index labels by gene_id
    labels_indexed = labels_df.set_index('gene_id')
    
    # Compute frequencies
    freq_df = compute_dipeptide_frequencies(seqs, labels_indexed)
    log(f"Frequency matrix: {freq_df.shape}")
    
    # For each class and its top dipeptides, run Mann-Whitney U
    results = []
    n_tests = 0
    
    for cls_name in CLASS_NAMES:
        top_dipeps = shap_data.get(cls_name, [])
        if not top_dipeps:
            continue
        
        # Get gene IDs for this class
        class_genes = labels_indexed[labels_indexed['location'] == cls_name].index
        other_genes = labels_indexed[labels_indexed['location'] != cls_name].index
        
        class_genes = [g for g in class_genes if g in freq_df.index]
        other_genes = [g for g in other_genes if g in freq_df.index]
        
        log(f"  {cls_name}: {len(class_genes)} in-class, {len(other_genes)} other")
        
        for dp, shap_imp in top_dipeps:
            if dp not in freq_df.columns:
                continue
            
            class_vals = freq_df.loc[class_genes, dp].values
            other_vals = freq_df.loc[other_genes, dp].values
            
            # Remove zeros to avoid degenerate distributions
            # But Mann-Whitney can handle ties
            mean_in = np.mean(class_vals)
            mean_other = np.mean(other_vals)
            
            # Effect size
            ratio = mean_in / mean_other if mean_other > 0 else float('inf')
            
            # Mann-Whitney U test
            try:
                stat, p_value = mannwhitneyu(class_vals, other_vals, alternative='two-sided')
            except Exception as e:
                log(f"    Warning: MWU failed for {dp}/{cls_name}: {e}")
                p_value = 1.0
            
            results.append({
                'dipeptide': dp,
                'class': cls_name,
                'mean_in_class': mean_in,
                'mean_in_other': mean_other,
                'ratio': ratio,
                'p_value': p_value,
                'SHAP_importance': shap_imp,
            })
            n_tests += 1
    
    results_df = pd.DataFrame(results)
    log(f"Total tests: {n_tests}")
    
    # FDR correction (Benjamini-Hochberg, manual implementation)
    if len(results_df) > 0:
        reject, p_adj = benjamini_hochberg(results_df['p_value'].values)
        results_df['p_adj'] = p_adj
        results_df['significant'] = reject
    else:
        results_df['p_adj'] = 1.0
        results_df['significant'] = False
    
    # Sort and save
    results_df = results_df.sort_values(['class', 'p_adj'])
    out_csv = f"{OUT_DIR}/motif_enrichment.csv"
    results_df.to_csv(out_csv, index=False)
    log(f"Saved enrichment results to {out_csv}")
    
    # Summary
    sig_count = results_df['significant'].sum()
    log(f"Significant dipeptides (FDR<0.05): {sig_count} / {n_tests}")
    for cls_name in CLASS_NAMES:
        sub = results_df[results_df['class'] == cls_name]
        sig_sub = sub[sub['significant']]
        log(f"  {cls_name}: {len(sig_sub)}/{len(sub)} significant")
        for _, row in sig_sub.head(10).iterrows():
            log(f"    {row['dipeptide']}: ratio={row['ratio']:.2f}, p_adj={row['p_adj']:.2e}")
    
    # ── Plot ──
    plot_enrichment(results_df)
    
    return results_df

def plot_enrichment(results_df):
    """Grouped horizontal bar chart of top 3-5 dipeptides per class."""
    fig, axes = plt.subplots(len(CLASS_NAMES), 1, figsize=(12, 4 * len(CLASS_NAMES)))
    
    for idx, cls_name in enumerate(CLASS_NAMES):
        ax = axes[idx] if len(CLASS_NAMES) > 1 else axes
        sub = results_df[results_df['class'] == cls_name].nlargest(5, 'ratio')
        
        if sub.empty:
            ax.text(0.5, 0.5, f'{cls_name}: No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(cls_name)
            continue
        
        # Sort by ratio descending (but plot ascending for horizontal bar)
        sub = sub.sort_values('ratio', ascending=True)
        colors = ['#2ecc71' if s else '#e74c3c' for s in sub['significant']]
        
        bars = ax.barh(sub['dipeptide'], sub['ratio'], color=colors, edgecolor='white')
        
        # Add significance asterisks
        for i, (_, row) in enumerate(sub.iterrows()):
            stars = ''
            if row['p_adj'] < 0.001:
                stars = '***'
            elif row['p_adj'] < 0.01:
                stars = '**'
            elif row['p_adj'] < 0.05:
                stars = '*'
            ax.text(row['ratio'] + 0.02, i, stars, va='center', fontsize=12, fontweight='bold')
        
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
        ax.set_title(f'{cls_name} — Top Dipeptide Enrichment (in-class / other)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Enrichment Ratio')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='FDR < 0.05'),
            Patch(facecolor='#e74c3c', label='Not significant'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    out_png = f"{OUT_DIR}/motif_enrichment.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Saved enrichment plot to {out_png}")


# ── Analysis 2: SGD/UniProt Annotation Cross-check ─────────
def run_sgd_annotation_check(labels_df, n_sample=200):
    """Query SGD REST API for GO annotations, check for localization signals."""
    log("=" * 60)
    log("ANALYSIS 2: SGD Annotation Cross-check")
    log("=" * 60)
    
    # Stratified sample
    samples = []
    for cls_name in CLASS_NAMES:
        class_genes = labels_df[labels_df['location'] == cls_name]['gene_id'].values
        n = min(len(class_genes), max(30, n_sample // len(CLASS_NAMES)))
        sampled = np.random.choice(class_genes, size=n, replace=False)
        for g in sampled:
            samples.append((g, cls_name))
    
    log(f"Sampled {len(samples)} genes for SGD API queries")
    
    # Keywords to search in GO terms
    keywords = {
        'signal_peptide': ['signal peptide', 'signal sequence', 'signal recognition'],
        'nuclear_localization': ['nuclear localization', 'NLS', 'nuclear import', 'nuclear transport', 
                                  'nucleoplasm', 'nuclear pore'],
        'mitochondrial_targeting': ['mitochondrial', 'mitochondrion', 'mitochondria'],
        'transmembrane': ['transmembrane', 'integral component of membrane', 'membrane spanning'],
        'ER_targeting': ['endoplasmic reticulum', 'ER', 'secretory pathway'],
    }
    
    # Results
    annotation_hits = []
    success_count = 0
    fail_count = 0
    
    for i, (gene_id, cls_name) in enumerate(samples):
        if (i + 1) % 20 == 0:
            log(f"  Progress: {i+1}/{len(samples)} genes queried ({success_count} OK, {fail_count} failed)")
        
        try:
            url = f"https://www.yeastgenome.org/backend/locus/{gene_id}/go_details"
            resp = requests.get(url, timeout=15)
            time.sleep(0.5)  # Rate limiting
            
            if resp.status_code != 200:
                fail_count += 1
                continue
            
            data = resp.json()
            success_count += 1
            
            # Collect all GO term names and descriptions
            go_texts = []
            for item in data:
                go_term = item.get('go_term', {})
                go_name = go_term.get('display_name', '')
                go_desc = go_term.get('description', '')
                go_texts.append(f"{go_name} {go_desc}".lower())
            
            combined_text = ' '.join(go_texts)
            
            # Check each keyword category
            row = {'gene_id': gene_id, 'class': cls_name}
            for kw_cat, kw_list in keywords.items():
                found = any(kw.lower() in combined_text for kw in kw_list)
                row[kw_cat] = 1 if found else 0
            annotation_hits.append(row)
            
        except Exception as e:
            fail_count += 1
            if fail_count <= 5:
                log(f"  Error for {gene_id}: {e}")
            continue
    
    log(f"Completed: {success_count} successful, {fail_count} failed")
    
    hits_df = pd.DataFrame(annotation_hits)
    
    # Build contingency table
    if not hits_df.empty:
        kw_cols = list(keywords.keys())
        contingency_rows = []
        for cls_name in CLASS_NAMES:
            sub = hits_df[hits_df['class'] == cls_name]
            if sub.empty:
                continue
            row = {'class': cls_name, 'n_queried': len(sub)}
            for kw in kw_cols:
                row[f'{kw}_count'] = int(sub[kw].sum())
                row[f'{kw}_pct'] = sub[kw].mean() * 100
            contingency_rows.append(row)
        
        contingency = pd.DataFrame(contingency_rows)
        out_csv = f"{OUT_DIR}/motif_uniprot_crosscheck.csv"
        contingency.to_csv(out_csv, index=False)
        log(f"Saved crosscheck to {out_csv}")
        log("\nContingency table:")
        log(contingency.to_string())
    else:
        log("WARNING: No annotation data collected!")
        contingency = pd.DataFrame()
    
    return hits_df, contingency


# ── Analysis 3: Integrated Validation Table ────────────────
def build_integrated_validation(enrichment_df, annotation_hits_df):
    """Combine SHAP + enrichment + SGD annotations into final table."""
    log("=" * 60)
    log("ANALYSIS 3: Integrated Validation Table")
    log("=" * 60)
    
    if enrichment_df is None or enrichment_df.empty:
        log("ERROR: No enrichment data to integrate")
        return
    
    # Literature support lookup: known targeting motifs
    literature_known = {
        # Nuclear localization signals
        'KR': 'Known NLS component (K-K/R-X-K/R pattern)',
        'KK': 'Known NLS component (bipartite NLS)',
        'RR': 'Known NLS component (arginine-rich NLS)',
        'RK': 'Known NLS component',
        'PR': 'Proline-arginine rich motifs in nuclear proteins',
        # Mitochondrial targeting
        'RR': 'Mitochondrial targeting sequence (amphipathic helix)',
        'LL': 'Mitochondrial import signal component',
        'FF': 'Hydrophobic mitochondrial targeting',
        # Membrane/secretory
        'LL': 'Signal peptide hydrophobic core',
        'LF': 'Signal peptide hydrophobic core',
        'LI': 'Signal peptide hydrophobic core',
        'FF': 'Signal peptide hydrophobic core',
        'VL': 'Signal peptide hydrophobic core',
        'AL': 'Signal peptide hydrophobic core',
        'LA': 'Signal peptide hydrophobic core',
        # General
        'DD': 'Acidic patch, common in many compartments',
        'EE': 'Acidic patch, common in many compartments',
        'PP': 'Polyproline helix, various functions',
        'GG': 'Flexible linker, common everywhere',
        'SS': 'Phosphorylation target, various',
    }
    
    rows = []
    for _, row in enrichment_df.iterrows():
        dp = row['dipeptide']
        cls_name = row['class']
        
        # SGD annotation hits for this class+dipeptide
        sgd_hits = ""
        if annotation_hits_df is not None and not annotation_hits_df.empty:
            # We don't have direct dipeptide→annotation mapping,
            # so we use class-level annotation percentages
            class_ann = annotation_hits_df[annotation_hits_df['class'] == cls_name]
            if not class_ann.empty:
                ann_parts = []
                for col in class_ann.columns:
                    if col.endswith('_pct') and class_ann[col].values[0] > 0:
                        ann_parts.append(f"{col.replace('_pct','')}={class_ann[col].values[0]:.0f}%")
                sgd_hits = '; '.join(ann_parts)
        
        # Literature support
        lit = literature_known.get(dp, '')
        
        # Verdict
        if row['significant'] and lit:
            verdict = "Confirmed"
        elif row['significant'] and not lit:
            verdict = "Novel finding"
        elif not row['significant'] and lit:
            verdict = "Contradicts known biology"
        else:
            verdict = "Partially supported"
        
        rows.append({
            'dipeptide': dp,
            'class': cls_name,
            'SHAP_importance': row.get('SHAP_importance', np.nan),
            'mean_in_class': row['mean_in_class'],
            'mean_in_other': row['mean_in_other'],
            'ratio': row['ratio'],
            'p_value': row['p_value'],
            'p_adj': row['p_adj'],
            'SGD_annotation_hits': sgd_hits,
            'literature_support': lit,
            'verdict': verdict,
        })
    
    final_df = pd.DataFrame(rows)
    final_df = final_df.sort_values(['class', 'p_adj'])
    
    out_csv = f"{OUT_DIR}/motif_validation_final.csv"
    final_df.to_csv(out_csv, index=False)
    log(f"Saved final validation table to {out_csv}")
    
    # Summary
    log("\nVerdict distribution:")
    log(final_df['verdict'].value_counts().to_string())
    log("\nSample rows:")
    log(final_df.head(20).to_string())
    
    return final_df


# ── Main ───────────────────────────────────────────────────
def main():
    global shap_data
    
    log("=" * 60)
    log("STEP 4: Motif Validation")
    log("=" * 60)
    
    # Load data
    log("\nLoading data...")
    labels_df = load_labels()
    seqs = load_fasta()
    shap_data = load_shap_top()
    
    # ── Analysis 1 ──
    enrichment_df = dipeptide_enrichment(labels_df, seqs, shap_data)
    
    # ── Analysis 2 ──
    annotation_hits_df, contingency_df = run_sgd_annotation_check(labels_df, n_sample=150)
    
    # ── Analysis 3 ──
    final_df = build_integrated_validation(enrichment_df, annotation_hits_df)
    
    # ── Summary ──
    log("\n" + "=" * 60)
    log("STEP 4 COMPLETE")
    log("=" * 60)
    
    with open(f"{OUT_DIR}/summary.txt", 'w') as f:
        f.write("Step 4: Motif Validation Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Analysis 1: Dipeptide Enrichment\n")
        f.write("-" * 40 + "\n")
        if enrichment_df is not None:
            sig = enrichment_df[enrichment_df['significant']]
            f.write(f"Total dipeptide-class pairs tested: {len(enrichment_df)}\n")
            f.write(f"Significant after FDR correction: {len(sig)}\n\n")
            for cls_name in CLASS_NAMES:
                sub = enrichment_df[enrichment_df['class'] == cls_name]
                sig_sub = sub[sub['significant']]
                f.write(f"{cls_name}: {len(sig_sub)}/{len(sub)} significant\n")
                for _, row in sig_sub.head(5).iterrows():
                    f.write(f"  {row['dipeptide']}: ratio={row['ratio']:.2f}, p_adj={row['p_adj']:.2e}\n")
            f.write("\n")
        
        f.write("Analysis 2: SGD Annotation Cross-check\n")
        f.write("-" * 40 + "\n")
        if contingency_df is not None and not contingency_df.empty:
            f.write(contingency_df.to_string())
            f.write("\n\n")
        
        f.write("Analysis 3: Integrated Validation\n")
        f.write("-" * 40 + "\n")
        if final_df is not None:
            f.write(f"Verdict distribution:\n{final_df['verdict'].value_counts().to_string()}\n\n")
    
    log("Summary written to summary.txt")
    log("All outputs in: " + OUT_DIR)


if __name__ == '__main__':
    main()
