from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import kmeans
from sklearn.decomposition import PCA

def pca_on_dataset(original_dataset):

    # discard columns not wanted
    dataset = original_dataset[:, 2:-1]

    dataset = kmeans.normalize_vectors(
        dataset = dataset, 
        nominal_attributes = [0, 1, 3, 4], 
        continuous_attributes = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    )

    non_null_datapoints_indices = (np.sum((dataset[:, np.arange(6, 20)] == 0), axis = 1) == 0) & (~(np.isnan(dataset[:, -1])))

    original_dataset_non_null = original_dataset[non_null_datapoints_indices, :]

    dataset = dataset[non_null_datapoints_indices]

    pca = PCA(n_components = 2)

    pca_on_dataset = pca.fit_transform(dataset)

    # use original satisfaction information to colour
    satisfied = np.where(original_dataset_non_null[:, 24] == "satisfied")[0]
    # pca_satisfied = pca_on_dataset[satisfied]
    neutral_or_dissatisfied = np.where(original_dataset_non_null[:, 24] == "neutral or dissatisfied")[0]


    plt.figure()

    plt.scatter(pca_on_dataset[satisfied, 0], pca_on_dataset[satisfied, 1], s = 20, alpha = 0.7)
    plt.scatter(pca_on_dataset[neutral_or_dissatisfied, 0], pca_on_dataset[neutral_or_dissatisfied, 1], s = 20, alpha = 0.7)
    # plt.xlim(np.min(normalized_dataset[:, 0]), np.max(normalized_dataset[:, 0]))
    # plt.ylim(np.min(normalized_dataset[:, 1]), np.max(normalized_dataset[:, 1]))

    plt.grid(True)
    plt.show()

pca_on_dataset(pd.read_csv("train.csv").values)