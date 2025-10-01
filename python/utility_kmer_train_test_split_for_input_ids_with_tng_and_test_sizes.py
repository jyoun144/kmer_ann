import pandas as pd
import numpy as np
from InputIDDataFrameDataset import InputIDDataFrameDataset
from itertools import product

def generate_train_and_test_datasets_from_input_ids(input_file_path, train_prop, kmer_length):
    df_source, vocab_size, num_of_classes = _retrieve_tsv_file(input_file_path, kmer_length)
    max_length = _get_max_token_length(df_source.input_ids)
    df_train, df_val = _get_train_test_split_datasets(df_source, train_prop)
    print(df_train.label.value_counts())
    print(df_val.label.value_counts())
    print(f'Generated train/val split datasets with shapes {df_train.shape} and {df_val.shape}, respectively.')
    dataset_train, dataset_val = _format_result(df_train, df_val)
    
    return dataset_train, dataset_val, max_length, vocab_size, num_of_classes

def _retrieve_tsv_file(file_path, kmer_length, sep='\t', header=None, names=['taxa_id', 'read_name', 'read']):
    df = pd.read_csv(file_path, sep=sep, header=header, names=names)
    dict_keys = _get_kmer_dict(kmer_length)
    dict_labels = _get_class_labels_dict(df)
    df.loc[:, 'input_ids'] = df.read.apply(lambda x: [ dict_keys[x[idx:idx+kmer_length]]  for idx in range(len(x) - kmer_length + 1)])
    df.loc[:, 'label'] = df.taxa_id.apply(lambda x: dict_labels[x])
    vocab_size = len(dict_keys)
    num_of_classes = len(dict_labels)
    print(f'Kmer length:{kmer_length}\nVocab size: {vocab_size}')

    return df, vocab_size, num_of_classes

def _get_kmer_dict(kmer_length):
    dict_keys = { key:idx for idx, key in enumerate([ ''.join(tup) for tup in product(['A', 'C', 'G', 'T' ], repeat=kmer_length) ])}

    return dict_keys

def _get_class_labels_dict(df):
    dict_labels = {  taxa_id:idx for idx, taxa_id in enumerate(df.taxa_id.unique()) }

    return dict_labels

def _get_max_token_length(list_input_ids):
    max_length = max({ len(item) for item in list_input_ids })
    print(f'Max sequence token length is {max_length}.')

    return max_length

def _get_train_test_split_datasets(df_source, train_prop):
    df_list_train = []
    df_list_test = []

    for label in df_source.label.unique():
        df_tmp = df_source[df_source.label == label]
        row_count = df_tmp.shape[0]
        train_size = int(np.ceil(train_prop * row_count))
        val_size = row_count - train_size
        df_train = df_tmp.sample(n=train_size)
        df_test = df_tmp[~df_tmp.index.isin(df_train.index)].sample(n=val_size)
        df_list_train.append(df_train)
        df_list_test.append(df_test)

    return pd.concat(df_list_train), pd.concat(df_list_test)

def _format_result( df_train, df_val):
    dataset_train = InputIDDataFrameDataset(df_train)
    dataset_val = InputIDDataFrameDataset(df_val)

    return dataset_train,  dataset_val
