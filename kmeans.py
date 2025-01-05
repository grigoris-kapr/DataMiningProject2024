import os

import numpy as np
import pandas as pd



#! CONFIGURE PROPERLY PATH FOR THE DATASET FILE
path_to_dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) + '\\Datasets\\Airline_Passenger_Satisfaction' + '\\train.csv'




def similarity_assymetric_sensitive(
    vector_1: np.ndarray, 
    vector_2: np.ndarray, 
    # list that contains the indices of continuous value attributes of these vectors
    continuous_attributes: list, 
    # list that contains the indices of discreet value attributes of these vectors
    discreet_attributes: list, 
    # list that contains the indices of the assymetric attributes of these vectors
    assymetric_attributes: list
) -> float:
    
    number_of_attributes_that_count = vector_1.shape[0] - np.sum((vector_1[assymetric_attributes] == 0) & (vector_2[assymetric_attributes] == 0))
    sum_of_discreet = np.sum(vector_1[discreet_attributes] == vector_2[discreet_attributes])
    sum_of_continuous = np.sum(1 - np.abs(vector_1[continuous_attributes] - vector_2[continuous_attributes]))
    return (sum_of_continuous + sum_of_discreet) / number_of_attributes_that_count


def broacasted_similarity_assymetric_sensitive(
    dataset, 
    centers, 
    # list that contains the indices of continuous value attributes of these vectors
    continuous_attributes: list, 
    # list that contains the indices of discreet value attributes of these vectors
    discreet_attributes: list, 
    # list that contains the indices of the assymetric attributes of these vectors
    assymetric_attributes: list
) -> np.ndarray:
    
    datapoint_dim = dataset.shape[1]

    # broadcast
    dataset = dataset[:, None, :]
    centers = centers[None, :, :]
    datapoint_dim = np.array(datapoint_dim)[:, :]

    number_of_attributes_that_count = datapoint_dim - np.sum((dataset[:, :, assymetric_attributes] == 0) & (centers[:, :, assymetric_attributes] == 0), axis = 2)
    sum_of_discreet = np.sum(dataset[:, :, discreet_attributes] == centers[:, :, discreet_attributes], axis = 2)
    sum_of_continuous = np.sum(1 - np.abs(dataset[:, :, continuous_attributes] - centers[:, :, continuous_attributes]), axis = 2)
    return (sum_of_discreet + sum_of_continuous) / number_of_attributes_that_count




def kmeans(
    k: int, 
    dataset: np.ndarray, 
    dist_metric, 
    iterations: int = None, 
    initial_centers: np.ndarray = None,  # None option picks the centers randomly
):
    dataset_size = dataset.shape[0]
    datapoint_dim = dataset.shape[1]

    if initial_centers is None:
        initial_centers = np.random.rand(k, datapoint_dim)

    current_centers = initial_centers

    while True:

        clusters: list[list] = [[] for _ in range(k)]

        min_datapoint_dist_from_center = (None, None)

        for datapoint in dataset:

            for i, center in enumerate(current_centers):

                dist = dist_metric(datapoint, center)

                if i == 0:
                    min_datapoint_dist_from_center = (i, dist)
                elif dist < min_datapoint_dist_from_center:
                    min_datapoint_dist_from_center = (i, dist)

            clusters[min_datapoint_dist_from_center[0]].append(datapoint)

        new_centers = []
        for i in range(k):
            new_centers.append(
                np.sum(np.array(clusters[i]), axis = 0) / datapoint_dim
            )
        new_centers = np.array(new_centers)

        if current_centers == new_centers:
            return (new_centers, clusters)

        if iterations is not None:
            iterations -= 1
            if iterations == 0:
                return (new_centers, clusters)
            



def optimized_kmeans(
    k: int, 
    dataset: np.ndarray, 
    dist_metric, 
    iterations: int = None, 
    initial_centers: np.ndarray = None,  # None option picks the centers randomly
):
    dataset_size = dataset.shape[0]
    datapoint_dim = dataset.shape[1]

    if initial_centers is None:
        initial_centers = np.random.rand(k, datapoint_dim)

    current_centers = initial_centers

    while True:

        datapoint_to_new_clusters = np.argmax(dist_metric(dataset_size, current_centers), axis = 1)

        new_centers = []
        for i in range(k):
            new_centers.append(
                np.sum(dataset[np.where(datapoint_to_new_clusters == i)], axis = 0) / datapoint_dim
            )
        new_centers = np.array(new_centers)

        if current_centers == new_centers:
            return new_centers, datapoint_to_new_clusters

        if iterations is not None:
            iterations -= 1
            if iterations == 0:
                return new_centers, datapoint_to_new_clusters
                



if __name__ == '__main__':

    dataset = pd.read_csv(path_to_dataset, header = 1).values

    #! TODO : CONTINUE HERE