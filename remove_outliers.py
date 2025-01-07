import math

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.cluster



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

dataset = pd.read_csv("test_2000.csv", header = 0).values

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

# dbscan_output = sklearn.cluster.DBSCAN(eps=2, min_samples=3).fit(normalized_dataset)

num_splits = 25
sse = [[ 0 for _ in range(num_splits)] for i in range(30)]
for samples_in_rad in range(2,15):
    for eps in range(num_splits):
        dbscan_output = sklearn.cluster.DBSCAN(eps=(eps+1)*0.1, min_samples=samples_in_rad*5+20, n_jobs=4).fit(normalized_dataset).labels_
        # Calculate the centroids and SSE 
        # unique_labels = set(dbscan_output) 
        for label in dbscan_output: 
            if label != -1: # Exclude noise points 
                cluster_points = normalized_dataset[dbscan_output == label]
                centroid = cluster_points.mean(axis=0) 
                sse[samples_in_rad][eps] += np.sum((cluster_points - centroid) ** 2)

plt.figure(figsize=(10, 6))

plt.plot(range(1, num_splits+1), sse[2], marker='o')
plt.plot(range(1, num_splits+1), sse[3], marker='v')
plt.plot(range(1, num_splits+1), sse[4], marker='s')
plt.plot(range(1, num_splits+1), sse[5], marker='x')
plt.plot(range(1, num_splits+1), sse[6], marker='*')
plt.plot(range(1, num_splits+1), sse[7], marker='^')
plt.plot(range(1, num_splits+1), sse[8], marker='<')
plt.plot(range(1, num_splits+1), sse[9], marker='>')
plt.plot(range(1, num_splits+1), sse[10], marker='p')
plt.plot(range(1, num_splits+1), sse[11], marker='P')
plt.plot(range(1, num_splits+1), sse[12], marker='+')
plt.plot(range(1, num_splits+1), sse[13], marker='X')
plt.plot(range(1, num_splits+1), sse[14], marker='d')

plt.title('Plot for DBSCAN Clustering')
plt.xlabel('eps * 10')
plt.ylabel('Sum of Squared Errors (SSE)')
# plt.grid(True)
plt.show()
# print("Total number of points: ", len(normalized_dataset))
# print("Outliers from DBSCAN: ", np.sum(dbscan_output.labels_ == -1))
