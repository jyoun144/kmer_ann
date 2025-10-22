import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from skbio.stats.distance import DistanceMatrix
from skbio.stats.distance import permanova

def process_embedding_file(embeddings_file_path, taxa_keys_file_path, class_labels_file_path, num_of_pca_comp, permutations):
    arr = _retrieve_numpy_file(embeddings_file_path)
    df_taxa =  _retrieve_tsv_file(taxa_keys_file_path)
    dict_taxa_key_to_label = _get_taxa_key_to_label_dict(class_labels_file_path)
    df_taxa.loc[:, 'label'] = df_taxa.taxa_key.apply(lambda x: int(dict_taxa_key_to_label[x]))
    group_names = df_taxa.label.values
    num_of_groups = df_taxa.label.unique().shape[0]
    arr_pca, explained_var_pca = _perform_pca_transform(arr, num_of_pca_comp)
    permanova_result, rsquared = _perform_permanova_analysis(arr_pca, group_names, num_of_groups, permutations)

    return df_taxa, arr_pca

def save_pca_and_taxa_keys_files(arr_pca, df_taxa_keys, source_embedding_file_path, output_dir_path):
    output_file_path_without_ext = _parse_file_path(source_embedding_file_path, output_dir_path)
    _save_numpy_file(arr_pca, f'{output_file_path_without_ext}.npy')
    _save_tsv_file(df_taxa_keys, f'{output_file_path_without_ext}.tsv')

def _retrieve_numpy_file(file_path):
    arr = np.load(file_path)
    print(f'Retrieved numpy file of shape {arr.shape} from {file_path}.')

    return arr

def _retrieve_tsv_file(file_path):
   df = pd.read_csv(filepath_or_buffer=file_path, sep='\t', header=0)
   print(f'Retrieved tsv file of shape {df.shape} from {file_path}.')

   return df

def _get_taxa_key_to_label_dict(class_labels_file_path):
    df_class_labels = _retrieve_tsv_file(class_labels_file_path)
    dict_taxa_key_to_label = { row[1].taxa_key:row[1].label for row in df_class_labels.iterrows() }

    return dict_taxa_key_to_label

def _perform_pca_transform(arr, num_of_components=2):
    sc = StandardScaler()
    arr_scaled = sc.fit_transform(arr)
    pca = PCA(n_components=num_of_components)
    mat_pca = pca.fit_transform(arr_scaled)
    print(f'Explained variance ratios: {pca.explained_variance_ratio_}')

    return mat_pca, pca.explained_variance_ratio_

def _perform_permanova_analysis(arr_minus_shuffled_pca, group_names, num_of_groups, permutations):
    distances_arr_minus_shuffled_pca = squareform(pdist(arr_minus_shuffled_pca, metric='braycurtis'))
    dist_arr = DistanceMatrix(distances_arr_minus_shuffled_pca)
    permanova_result = permanova(dist_arr, grouping=group_names, permutations=permutations)
    print(permanova_result)
    rsquared = permanova_result['test statistic']/(permanova_result['test statistic'] + (permanova_result['sample size'] - num_of_groups))
    print(f'\nr-squared: {rsquared:.8f}')

    return permanova_result, rsquared

def _save_tsv_file(df, file_path):
    df.to_csv(path_or_buf=file_path, sep='\t', header=True, index=False)
    print(f'Wrote tsv file of shape {df.shape} to {file_path}.')

def _save_numpy_file(arr, file_path):
    np.save(file_path, arr)
    print(f'Wrote numpy file of shape {arr.shape} to {file_path}.')

def _parse_file_path(input_file_path, output_dir_path):
    file_name = os.path.basename(input_file_path)
    file_name_without_ext =  os.path.splitext(file_name)[0]
    output_file_path_without_ext = os.path.join(output_dir_path, f'{file_name_without_ext}_pca-permanova-embeddings_all-classes')

    return output_file_path_without_ext

if __name__ == '__main__':
    embeddings_file_path = sys.argv[1]
    taxa_keys_file_path = sys.argv[2]
    class_labels_file_path = sys.argv[3]
    output_dir_path = sys.argv[4]
    num_of_pca_comp = int(sys.argv[5])
    permutations = int(sys.argv[6])

df_taxa, arr_pca = process_embedding_file(embeddings_file_path, taxa_keys_file_path, class_labels_file_path, num_of_pca_comp, permutations)
save_pca_and_taxa_keys_files(arr_pca, df_taxa, embeddings_file_path, output_dir_path)



