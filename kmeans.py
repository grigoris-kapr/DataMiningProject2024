import os
import math

import numpy as np
import cupy as cp
import pandas as pd



#! CONFIGURE PROPERLY PATH FOR THE DATASET FILE
path_to_dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) + '\\Datasets\\Airline_Passenger_Satisfaction' + '\\train.csv'



name_to_lib = {
    'numpy' : np, 
    'cupy' : cp
}



def distance_metric_for_discreet_and_continuous_with_assymetric_support(
    dataset: np.ndarray | cp.ndarray,  # 2 dimensional ndarray of shape (num of datapoints, num of attributes)
    centers: np.ndarray | cp.ndarray,  # 2 dimensional ndarray of shape (num of centers, num of attributes)
    nominal_attributes: list,  # a list containing the indices of nominal value attributes
    continuous_attributes: list,  # a list containing the indices of continuous value attributes 
    assymetric_attributes: list | None,  # a list containing the indices of assymetric value attributes
    nplikelib: str = None,  # either 'numpy' or 'cupy' depending on whether cuda is supported and code needs to run on a gpu
) -> np.ndarray | cp.ndarray:  # 2 dimensional ndarray containing the similarity between each datapoint and each center, shape: (num of datapoint, num of centers)

    if nplikelib is None:
        nplike = np
    else:
        nplike = name_to_lib[nplikelib]
    
    datapoint_dim = dataset.shape[1]

    # broadcast
    dataset = dataset[:, None, :]
    centers = centers[None, :, :]
    datapoint_dim = nplike.array(datapoint_dim)[None, None]

    # if there are any assymetric attributes
    if assymetric_attributes is not None:

        # calculate for each pair of datapoint - center the number of attributes that are assymetric and both 0, so they are removed from 
        # the total number of attributes that each datapoint - center distance needs to be divided so that calculation is complete
        number_of_attributes_that_count = datapoint_dim - nplike.sum((dataset[:, :, assymetric_attributes]) & (centers[:, :, assymetric_attributes]), axis = 2)
    
    # else if they are not just divide each pair of datapoint - center distance calculation with the number of all attributes
    else:

        number_of_attributes_that_count = datapoint_dim

    # calculate for each pair of datapoint - center the sum of all nominal 
    # attributes (1 value if they are different so they have distance of 1, 0 if they have the same value)
    sum_of_nominal = nplike.sum(dataset[:, :, nominal_attributes] != centers[:, :, nominal_attributes], axis = 2)

    # calculate for each pair of datapoint - center the sum of all continuous 
    # attributes (|x - y| is used for the continuous attribute distance metric and since each value is normalized to 
    # interval [0, 1], this will be the range of |x - y| for each attribute
    sum_of_continuous = nplike.sum(nplike.abs(dataset[:, :, continuous_attributes] - centers[:, :, continuous_attributes]), axis = 2)

    # for each datapoint - center distance combine the 2 partial sums for nominal and continuous and divide by number of attributes
    # that are not assymetric and both 0
    return (sum_of_nominal + sum_of_continuous) / number_of_attributes_that_count



def our_similarity_metric(dataset, centers):
    return distance_metric_for_discreet_and_continuous_with_assymetric_support(
        dataset = dataset, 
        centers = centers, 
        nominal_attributes = [0, 1, 3, 4], 
        continuous_attributes = np.setdiff1d(np.arange(dataset.shape[1]), np.array([0, 1, 3, 4])).tolist(), 
        assymetric_attributes = None
    )



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
    datapoint_dim = dataset.shape[1]

    if initial_centers is None:
        initial_centers = np.random.rand(k, datapoint_dim)

    current_centers = initial_centers

    while True:

        datapoint_to_new_clusters = np.argmax(dist_metric(dataset, current_centers), axis = 1)

        new_centers = []
        for i in range(k):
            new_centers.append(
                np.sum(dataset[np.where(datapoint_to_new_clusters == i)], axis = 0) / datapoint_dim
            )
        new_centers = np.array(new_centers)

        if (current_centers == new_centers).all():
            return new_centers

        if iterations is not None:
            iterations -= 1
            if iterations == 0:
                return new_centers
                



if __name__ == '__main__':

    dataset = pd.read_csv(path_to_dataset, header = 0).values

    # discard columns not wanted
    dataset = dataset[:, 2:-1]

    # find non nan values of the last attr
    not_nan_indices = []
    for i in range(dataset.shape[0]):
        if not math.isnan(dataset[i, -1]):
            not_nan_indices.append(i)
    
    dataset_without_null_at_last_attr = dataset[not_nan_indices, :]

    normalized_dataset = normalize_vectors(
        dataset = dataset_without_null_at_last_attr, 
        discreet_attributes = [0, 1, 3, 4], 
        continuous_attributes = [2, 5, 21]
    )

    optimized_kmeans(
        k = 10, 
        dataset = normalized_dataset, 
        dist_metric = our_similarity_metric, 
        iterations = 500
    )
