import sys
import numpy as np
import pandas as pd

def generate_dataset(sample_size, seed_value, output_file_path):
    taxa = []
    sequences = []
    rng = np.random.default_rng(seed_value)
    prob=0.5
    prob_frac = prob/3.0
    taxa_A = [ ''.join(rng.choice(['A', 'C', 'G', 'T'], size=150, p=[prob, prob_frac, prob_frac, prob_frac])) for _ in range(sample_size)]
    taxa_C = [ ''.join(rng.choice(['A', 'C', 'G', 'T'], size=150, p=[prob_frac, prob, prob_frac, prob_frac])) for _ in range(sample_size)]
    taxa_G = [ ''.join(rng.choice(['A', 'C', 'G', 'T'], size=150, p=[prob_frac, prob_frac, prob, prob_frac])) for _ in range(sample_size)]
    taxa_T = [ ''.join(rng.choice(['A', 'C', 'G', 'T'], size=150, p=[prob_frac, prob_frac, prob_frac, prob])) for _ in range(sample_size)]
    taxa.extend(np.repeat('taxa_A', len(taxa_A)))
    sequences.extend(taxa_A)
    taxa.extend(np.repeat('taxa_C', len(taxa_C)))
    sequences.extend(taxa_C)
    taxa.extend(np.repeat('taxa_G', len(taxa_G)))
    sequences.extend(taxa_G)
    taxa.extend(np.repeat('taxa_T', len(taxa_T)))
    sequences.extend(taxa_T)
    df = pd.DataFrame({'taxa_id':taxa, 'read':sequences})
    df.loc[:, 'prop_counts'] = df.apply(lambda x: np.unique(np.array([c for c in x.read]), return_counts=True), axis=1)
    df.loc[:, 'nucl_prop'] = df.apply(lambda x: x.prop_counts[1][np.where(x.prop_counts[0] == x.taxa_id[-1])[0][0]]/np.sum(x.prop_counts[1]), axis=1)
    df = df.drop(columns='prop_counts')
    _save_tsv_file(output_file_path, df)

def _save_tsv_file(file_path, df):
    df.to_csv(file_path, sep='\t', header=True, index=False)
    print(f'Saved tsv file of shape {df.shape} to {file_path}.')

if __name__ == '__main__':
    sample_size = int(sys.argv[1])
    seed_value = int(sys.argv[2])
    output_file_path = sys.argv[3]
    generate_dataset(sample_size, seed_value, output_file_path)
