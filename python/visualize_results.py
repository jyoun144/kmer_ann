import matplotlib.pyplot as plt
import numpy as np
import os

def get_numpy_files(start, end):
    prefix = '/mnt/d/pca_tutorial_results/pca_results'
    prefix = '/mnt/d/data_pca/pca_results'
    data_file_path = f'{prefix}_data_{start}_to_{end}.npy'
    labels_file_path = f'{prefix}_labels_{start}_to_{end}.npy'
    arr_data = np.load(data_file_path)
    arr_labels = np.load(labels_file_path)

    return arr_data, arr_labels

colors = ['blue', 'green', 'black', 'orange']

arr_data, arr_labels = get_numpy_files(0, 4000)
x = arr_data[:, 0]
y = arr_data[:, 1]
labels = arr_labels
my_colors = [ colors[val] for val in labels ]
fig, ax1 = plt.subplots()   
ax1.scatter(x,y, c=my_colors, s=8)
ax1.set_title('PCA Results (Epoch 0/15)\nr-squared=0.01335 / p-value=0.462 / test stat=54.06')
ax1.set_xlabel('PC1 (77.9%)')
ax1.set_ylabel('PC2 (20.4%)')
plt.show()
