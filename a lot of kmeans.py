import math
from matplotlib import pyplot as plt
import pandas as pd
import sklearn.cluster
import numpy as np

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

dataset = pd.read_csv("train.csv", header = 0).values

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

kmeans_sse = []
num_of_reps = 50
for i in range(num_of_reps):
    km_out = sklearn.cluster.KMeans(n_clusters=i+1,init='random',n_init=10).fit(normalized_dataset)
    kmeans_sse.append(km_out.inertia_)

# db_out = sklearn.cluster.DBSCAN(eps=1.0, min_samples=)
# filtered_dataset = 

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_of_reps+1), kmeans_sse, marker='o')
plt.title('Elbow Plot for KMeans Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.grid(True)
plt.show()