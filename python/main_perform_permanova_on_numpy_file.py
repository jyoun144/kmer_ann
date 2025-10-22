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

def process_embedding_file(logits_file_path, labels_file_path, start_index, end_index, num_of_pca_comp, permutations):
    arr_logits = _retrieve_numpy_file(logits_file_path, start_index, end_index)
    arr_labels = _retrieve_numpy_file(labels_file_path, start_index, end_index)
    num_of_groups = np.unique(arr_labels).shape[0]
    arr_pca, explained_var_pca = _perform_pca_transform(arr_logits, num_of_pca_comp)
    permanova_result, rsquared = _perform_permanova_analysis(arr_pca, arr_labels, num_of_groups, permutations)

    return arr_labels, arr_pca

def save_pca_and_label_files(arr_pca, arr_labels, start_index, end_index, output_dir_path):
    output_file_path_pca = os.path.join(output_dir_path, f'pca_results_data_{start_index}_to_{end_index}.npy')
    output_file_path_labels = os.path.join(output_dir_path, f'pca_results_labels_{start_index}_to_{end_index}.npy')
    _save_numpy_file(arr_pca, output_file_path_pca)
    _save_numpy_file(arr_labels, output_file_path_labels)

def _retrieve_numpy_file(file_path, start, end):
    arr = np.load(file_path)[start:end]
    print(f'Retrieved numpy file of shape {arr.shape} from {file_path}.')

    return arr

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

def _save_numpy_file(arr, file_path):
    np.save(file_path, arr)
    print(f'Wrote numpy file of shape {arr.shape} to {file_path}.')

if __name__ == '__main__':
    logits_file_path = sys.argv[1]
    labels_file_path = sys.argv[2]
    start_index = int(sys.argv[3])
    end_index = int(sys.argv[4])
    num_of_pca_comp = int(sys.argv[5])
    permutations = int(sys.argv[6])
    output_dir_path = sys.argv[7]

    arr_labels, arr_pca = process_embedding_file(logits_file_path, labels_file_path, start_index, end_index, num_of_pca_comp, permutations)
    save_pca_and_label_files(arr_pca, arr_labels, start_index, end_index, output_dir_path)



