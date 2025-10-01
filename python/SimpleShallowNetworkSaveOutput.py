import torch
from torch import nn
import numpy as np
import os
from scipy.special import softmax

class SimpleShallowNetworkSaveOutput(nn.Module):
    def __init__(self, embed_dim, vocab_size, feed_fwd_dim, num_of_classes):
        super().__init__()
        self.val_logits = []
        self.val_labels = []
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.input_layer = nn.Linear(embed_dim, feed_fwd_dim)
        self.relu_input_layer = nn.ReLU()
        self.hidden_layer = nn.Linear(feed_fwd_dim, feed_fwd_dim)
        self.relu_hidden_layer = nn.ReLU()
        self.output_layer =  nn.Linear(feed_fwd_dim, num_of_classes)

    def forward(self, input_ids, labels, save_output=False):
        embeddings = self.embedding_layer(input_ids)
        embeddings_mean = embeddings.mean(dim=1)
        logits = self.input_layer(embeddings_mean)
        logits = self.relu_input_layer(logits)
        logits = self.hidden_layer(logits)
        logits = self.relu_hidden_layer(logits)
        logits = self.output_layer(logits)
        if save_output:
            self.val_logits.append(logits.detach().cpu().numpy())
            self.val_labels.append(labels.detach().cpu().numpy())
        return logits

    def save_output(self, source_file_path):
        file_path_without_ext = os.path.splitext(source_file_path)[0]
        arr_logits = np.concatenate(self.val_logits)
        arr_labels = np.concatenate(self.val_labels)
        logits_file_path = f'{file_path_without_ext}_val_output_logits.npy'
        np.save(logits_file_path, arr_logits)
        print(f'Saved model validation logits of shape {arr_logits.shape} to {logits_file_path}.')
        labels_file_path = f'{file_path_without_ext}_val_output_labels.npy'
        np.save(labels_file_path, arr_labels)
        print(f'Saved model validation labels of shape {arr_labels.shape} to {labels_file_path}.')

    def print_softmax_results(self, threshold=0.5):
        arr_softmax = softmax(np.concatenate(self.val_logits), axis=1)
        arr_labels = np.concatenate(self.val_labels)
        softmax_indices = arr_softmax.argmax(axis=1)
        max_softmax_vals = np.array([ arr_softmax[idx, val] for idx,val in enumerate(softmax_indices) ])
        filter_softmax = max_softmax_vals > threshold
        reads_classified = filter_softmax.sum()
        num_correct = (softmax_indices[filter_softmax] == arr_labels[filter_softmax]).sum()
        total_reads = arr_labels.shape[0]
        print(f'Classification Results (threshold={threshold})')
        print(f'Total Reads: {total_reads}')
        print(f'Reads Classified: {reads_classified}.')
        print(f'Reads Correct: {num_correct}')
        print(f'Precision: {np.round(num_correct/reads_classified, 6)}')
        print(f'Recall: {np.round(num_correct/total_reads, 6)}') 
