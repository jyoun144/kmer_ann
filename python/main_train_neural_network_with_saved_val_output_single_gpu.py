import sys
import os
import numpy as np
import time
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from utility_kmer_train_test_split_for_input_ids_with_tng_and_test_sizes import generate_train_and_test_datasets_from_input_ids
from SimpleShallowNetworkSaveOutput import SimpleShallowNetworkSaveOutput

def train_neural_network(data_file_path, 
                         train_prop, 
                         num_of_epochs, 
                         learning_rate, 
                         batch_size, 
                         embed_dim, 
                         feed_fwd_dim, 
                         kmer_length):
    # Randomly generate training and validation datasets
    dataset_train, dataset_val, max_length, vocab_size, num_of_classes = \
                                  generate_train_and_test_datasets_from_input_ids(data_file_path, train_prop, kmer_length)

    # Configure ANN model, loss function and stochastic gradient descent method
    model, loss_fn, optimizer = _configure_gpu_model(learning_rate, embed_dim, vocab_size, feed_fwd_dim, num_of_classes)
    model, train_dataloader, val_dataloader, device = _get_training_components(model, dataset_train, dataset_val)
    train_start = time.time()
    # Train/test ANN network
    for epoch in range(num_of_epochs):
        epoch_start = time.time()
        train_loss = _train_loop(train_dataloader, model, loss_fn, optimizer, epoch, device)
        save_output = False if epoch != (num_of_epochs - 1) else True
        val_loss, val_accuracy = _test_loop(val_dataloader, model, loss_fn, device, save_output)
        epoch_duartion = time.time() - epoch_start
        print(f'Epoch {epoch+1}/{num_of_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, \
        Validation Accuracy: {val_accuracy:.4f}. Epoch Duration: {epoch_duartion:.1f} seconds')
    model.print_softmax_results()
    train_duration = time.time() - train_start
    print(f"Training finished. Took {train_duration:.1f} seconds")

    return model

def _get_training_components(model, dataset_train, dataset_val):
    device = _get_gpu_device()
    torch.cuda.set_device(device)
    train_dataloader = _get_dataloader(dataset_train,
                                        True,
                                        batch_size)
    val_dataloader = _get_dataloader(dataset_val,
                                     False,
                                     batch_size)
    model = model.to(device)

    return model, train_dataloader, val_dataloader, device

def _get_gpu_device():
    device = None
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f'Number of available GPUs: {gpu_count}')
        device = torch.device("cuda:0")
    else:
        raise Exception('ERROR: The availability of a GPU is required to run this code.',
                        'Unavailable GPUs')

    return device

def _get_dataloader(dataset, shuffle, batch_size):
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             collate_fn=_collate_fn)

    return data_loader

def get_file_path_without_ext(file_path):
    dir_path = os.path.dirname(file_path)
    file_name_without_ext = os.path.splitext(os.path.basename(file_path))[0]
    file_path_without_ext = os.path.join(dir_path, file_name_without_ext)
    
    return file_path_without_ext

def _get_gpu_device():
    device = None
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f'Number of available GPUs: {gpu_count}')
        device = torch.device("cuda:0")
    else:
        raise Exception('ERROR: The availability of a GPU is required to run this code.',
                        'Unavailable GPUs')

    return device

def _configure_gpu_model(learning_rate, embed_dim, vocab_size, feed_fwd_dim, num_of_classes):
    model = SimpleShallowNetworkSaveOutput(embed_dim, vocab_size, feed_fwd_dim, num_of_classes)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, loss_fn, optimizer

def _train_loop(dataloader, model, loss_fn, optimizer, epoch, device):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    train_loss = 0.0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        optimizer.zero_grad()
        pred = model(X, y)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().cpu().item() * y.detach().cpu().size(0)
    train_loss /= size

    return train_loss

def _test_loop(dataloader, model, loss_fn, device, save_output):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X, y, save_output=save_output)
            loss = loss_fn(logits, y)
            logits = logits.detach().cpu().numpy()
            loss = loss.detach().cpu().item()
            batch_labels = y.detach().cpu().numpy()
            val_loss += loss * batch_labels.shape[0]
            val_accuracy += (logits.argmax(axis=1) == batch_labels).sum()
        val_loss /= len(dataloader.dataset)
        val_accuracy /= len(dataloader.dataset)

    return val_loss, val_accuracy

def _collate_fn(batch):
    labels = []
    input_ids = []
    for row_input_ids, row_label in batch:
        labels.append(row_label)
        input_ids.append(torch.LongTensor(row_input_ids))
    labels = torch.LongTensor(labels)
    input_ids = torch.stack(input_ids)

    return input_ids, labels

def save_model(model, source_model_file_path):
    output_file_path = f'{os.path.splitext(source_model_file_path)[0]}_saved_model_params.pkl'
    torch.save(model.state_dict(), output_file_path)
    print(f'Saved trained model at {output_file_path}')

if __name__ == '__main__':
    data_file_path = sys.argv[1]
    train_prop = float(sys.argv[2])
    num_of_epochs = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    batch_size = int(sys.argv[5])
    embed_dim = int(sys.argv[6])
    feed_fwd_dim = int(sys.argv[7])
    kmer_length = int(sys.argv[8])

    # Invoke ANN model training workflow
    trained_model = train_neural_network(data_file_path, 
                                         train_prop, 
                                         num_of_epochs, 
                                         learning_rate, 
                                         batch_size, 
                                         embed_dim, 
                                         feed_fwd_dim, 
                                         kmer_length)
    trained_model.save_output(data_file_path)
    
    # Save new model
    save_model(trained_model, data_file_path)


