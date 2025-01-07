import math
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def normalize_vectors(
    dataset: np.ndarray,  # 2 dimensional ndarray, 
    discreet_attributes: list, 
    continuous_attributes: list
) -> np.ndarray:
    
    normalized_dataset = np.zeros_like(dataset, dtype = 'float32')
    
    min_values = np.min(dataset[:, continuous_attributes], axis = 0).astype(np.float32)
    range_of_attributes = np.max(dataset[:, continuous_attributes], axis = 0).astype(np.float32) - min_values

    # in case an attribute has the same value for all vectors, don't let a division by zero occur
    range_of_attributes[range_of_attributes == 0] = 1.0

    normalized_dataset[:, continuous_attributes] = (dataset[:, continuous_attributes].astype(np.float32) - min_values) / (range_of_attributes)

    for col in discreet_attributes:
        _, indices = np.unique(dataset[:, col], return_inverse = True)
        normalized_dataset[:, col] = indices.astype(np.float32)


    return normalized_dataset

dataset = pd.read_csv("test_5000.csv", header = 0).values

# discard columns not wanted
dataset = dataset[:, 2:-1]

# find non nan values
not_nan_indices = []
for i in range(dataset.shape[0]):
    # 0 in the ratings is a null
    if not 0 in dataset[i, 6:20]:
        # some of the delays are NaN
        if not math.isnan(dataset[i, -2]) and not math.isnan(dataset[i, -1]):
            not_nan_indices.append(i)

dataset_without_nulls = dataset[not_nan_indices, :]

normalized_dataset = normalize_vectors(
    dataset = dataset_without_nulls, 
    discreet_attributes = [0, 1, 3, 4], 
    continuous_attributes = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
)

k = 4  # Typically set to min_samples - 1
neigh = NearestNeighbors(n_neighbors=k)
nbrs = neigh.fit(normalized_dataset)
distances, indices = nbrs.kneighbors(normalized_dataset)

# Sort the distances
distances = np.sort(distances[:, k-1], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title('K-NN Distance Plot')
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{k}-th Nearest Neighbor Distance')
plt.grid(True)
plt.show()

