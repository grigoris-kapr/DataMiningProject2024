import os
import math

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt



#! CONFIGURE PROPERLY PATH FOR THE DATASET FILE
path_to_dataset = os.path.abspath(os.path.join(os.path.dirname(_file_), '..', '..')) + '\\Datasets\\Airline_Passenger_Satisfaction' + '\\train.csv'



name_to_lib = {
    'numpy' : np, 
    'cupy' : cp
}



# Calculates distance matrix for every pair of datapoint of the dataset
def calculate_distance_matrix(
    dataset, 
    batch_size, 
    nominal_attributes = [0, 1, 3, 4], 
    continuous_attributes = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    assymetric_attributes = None, 
    nplikelib: str = None,  # either 'numpy' or 'cupy' depending on whether cuda is supported and code needs to run on a gpu
) -> np.ndarray | cp.ndarray:
    
    if nplikelib is None:
        nplike = np
    else:
        nplike = name_to_lib[nplikelib]

    distance_matrix = []
    for i in range(0, dataset.shape[0], batch_size):
        distance_matrix.append(
            distance_metric_for_discreet_and_continuous_with_assymetric_support(
                dataset[i : i + batch_size],
                dataset,
                nominal_attributes = nominal_attributes, 
                continuous_attributes = continuous_attributes,
                assymetric_attributes = assymetric_attributes, 
                nplikelib = nplikelib
            )
        )
    return nplike.vstack(distance_matrix)



def distance_metric_for_discreet_and_continuous_with_assymetric_support(
    dataset: np.ndarray | cp.ndarray,  # 2 dimensional ndarray of shape (num of datapoints, num of attributes)
    centers: np.ndarray | cp.ndarray,  # 2 dimensional ndarray of shape (num of centers, num of attributes)
    nominal_attributes: list | None = None,  # a list containing the indices of nominal value attributes
    continuous_attributes: list | None = None,  # a list containing the indices of continuous value attributes 
    assymetric_attributes: list | None  = None,  # a list containing the indices of assymetric value attributes
    nplikelib: str = None,  # either 'numpy' or 'cupy' depending on whether cuda is supported and code needs to run on a gpu
) -> np.ndarray | cp.ndarray:  # 2 dimensional ndarray containing the similarity between each datapoint and each center, shape: (num of datapoint, num of centers)

    if nplikelib is None:
        nplike = np
    else:
        nplike = name_to_lib[nplikelib]
    
    number_of_datapoints = dataset.shape[0]
    number_of_centers = centers.shape[0]

    # broadcast
    datapoint_dim = nplike.array(dataset.shape[1])[None, None]
    dataset = dataset[:, None, 🙂
    centers = centers[None, :, 🙂
    

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
    if nominal_attributes is not None:
        sum_of_nominal = nplike.sum(dataset[:, :, nominal_attributes] != centers[:, :, nominal_attributes], axis = 2)
    else:
        sum_of_nominal = nplike.zeros((number_of_datapoints, number_of_centers))

    # calculate for each pair of datapoint - center the sum of all continuous 
    # attributes (|x - y| is used for the continuous attribute distance metric and since each value is normalized to 
    # interval [0, 1], this will be the range of |x - y| for each attribute
    if continuous_attributes is not None:
        sum_of_continuous = nplike.sum(nplike.abs(dataset[:, :, continuous_attributes] - centers[:, :, continuous_attributes]), axis = 2)
    else:
        sum_of_continuous = nplike.zeros((number_of_datapoints, number_of_centers))

    # for each datapoint - center distance combine the 2 partial sums for nominal and continuous and divide by number of attributes
    # that are not assymetric and both 0
    return (sum_of_nominal + sum_of_continuous) / number_of_attributes_that_count



def our_similarity_metric(dataset, centers, nplikelib):
    return distance_metric_for_discreet_and_continuous_with_assymetric_support(
        dataset = dataset, 
        centers = centers, 
        nominal_attributes = [0, 1, 3, 4], 
        continuous_attributes = np.setdiff1d(np.arange(dataset.shape[1]), np.array([0, 1, 3, 4])).tolist(), 
        assymetric_attributes = None
    )



def normalize_vectors(
    dataset: np.ndarray,  # 2 dimensional ndarray, 
    nominal_attributes: list | None = None, 
    continuous_attributes: list | None = None
) -> np.ndarray:
    
    normalized_dataset = np.zeros_like(dataset, dtype = 'float32')

    # for converting nominal values to integer aliases (starting from 0 up to n - 1 where n is number of distinct nominal values)
    if nominal_attributes is not None:

        for col in nominal_attributes:
            _, indices = np.unique(dataset[:, col], return_inverse = True)
            normalized_dataset[:, col] = indices.astype(np.float32)
    else:
        nominal_attributes = []
    
    # for continuous values (if any was given) normalization to [0, 1] interval
    if continuous_attributes is not None:
        min_values = np.min(dataset[:, continuous_attributes], axis = 0).astype(np.float32)
        range_of_attributes = np.max(dataset[:, continuous_attributes], axis = 0).astype(np.float32) - min_values

        # in case an attribute has the same value for all vectors, don't let a division by zero occur
        range_of_attributes[range_of_attributes == 0] = 1.0

        normalized_dataset[:, continuous_attributes] = (dataset[:, continuous_attributes].astype(np.float32) - min_values) / (range_of_attributes)
    else:
        continuous_attributes = []

    # convert all other attributes to np.float32 and pass as-is
    rest_attributes = np.setdiff1d(np.arange(dataset.shape[1]), np.union1d(np.array(continuous_attributes), np.array(nominal_attributes)))

    if rest_attributes.size != 0:

        normalized_dataset[:, rest_attributes] = dataset[:, rest_attributes].astype(np.float32)


    return normalized_dataset
            


def kmeans(
    k: int, 
    dataset: np.ndarray | cp.ndarray, 
    dist_metric, 
    tolerance: float = 1e-5, 
    iterations: int = None, 
    initial_centers: np.ndarray = None,  # None option picks the centers randomly
    nplikelib: str = None,  # either 'numpy' or 'cupy' depending on whether cuda is supported and code needs to run on a gpu
):
    if nplikelib is None:
        nplike = np
    else:
        nplike = name_to_lib[nplikelib]

    dataset_size = dataset.shape[0]
    datapoint_dim = dataset.shape[1]

    if initial_centers is None:
        initial_centers = nplike.random.rand(k, datapoint_dim)

    current_centers = initial_centers

    while True:

        datapoint_to_new_clusters = nplike.argmin(dist_metric(dataset, current_centers, nplikelib), axis = 1)

        new_centers = nplike.zeros_like(current_centers)

        datapoint_to_new_cluster_one_hot_like = nplike.zeros((dataset_size, k))
        datapoint_to_new_cluster_one_hot_like[nplike.arange(dataset_size), datapoint_to_new_clusters] = 1

        datapoints_per_cluster = nplike.sum(datapoint_to_new_cluster_one_hot_like, axis = 0)

        empty_clusters = datapoints_per_cluster == 0

        vector_of_each_cluster_sum = nplike.matmul(datapoint_to_new_cluster_one_hot_like.T, dataset)

        new_centers[empty_clusters] = current_centers[empty_clusters]

        new_centers[~empty_clusters] = vector_of_each_cluster_sum[~empty_clusters] / datapoints_per_cluster[~empty_clusters][:, None]

        # if (nplike.abs(current_centers - new_centers) < tolerance).all():
        if (nplike.linalg.norm(current_centers - new_centers) < tolerance):
            return new_centers, datapoint_to_new_clusters

        if iterations is not None:
            iterations -= 1
            if iterations == 0:
                return new_centers, datapoint_to_new_clusters
            
        current_centers = new_centers



def test_sklearn_and_my_for_a_specific_k(k):

    dataset = pd.read_csv(path_to_dataset).values

    # discard columns not wanted
    dataset = dataset[:, 2:-1]

    # convert nominal from str to integer encoding (so it can be stored as np.float)
    dataset = normalize_vectors(
        dataset = dataset, 
        nominal_attributes = [0, 1, 3, 4], 
    )

    # discard null values
    dataset = dataset[(np.sum((dataset[:, np.arange(6, 20)] == 0), axis = 1) == 0) & (~(np.isnan(dataset[:, -1])))]

    # apply normalization on the dataset
    normalized_dataset = normalize_vectors(
        dataset = dataset, 
        continuous_attributes = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    )

    normalized_dataset_on_cuda = cp.asarray(normalized_dataset)

    # apply k means
    sklearn_kmeans_result = KMeans(n_clusters = k, random_state = 0).fit(normalized_dataset)
    sklearn_centers = sklearn_kmeans_result.cluster_centers_, sklearn_kmeans_result.labels_
    my_centers = kmeans(
                    k = k, 
                    dataset = normalized_dataset_on_cuda, 
                    dist_metric = our_similarity_metric, 
                    tolerance = 1e-4, 
                    # iterations = 10000, 
                    nplikelib = 'cupy'
                )

    # after each iteration visualize centers and clusters after applying PCA on the dataset
    # (kmeans is applied on the dataset BEFORE PCA, then PCA is applied on the resulting dataset - centers so that 
    # it can be visualized)

    pca = PCA(n_components = 2)

    my_current_centers = cp.asnumpy(my_centers[0])
    my_current_clusters = cp.asnumpy(my_centers[1])

    sklearn_current_centers = sklearn_centers[0]
    sklearn_current_clusters = sklearn_centers[1]

    SSE_for_kmeans_sklearn = np.sum((normalized_dataset - sklearn_current_centers[sklearn_current_clusters]) ** 2)
    SSE_for_Kmeans_homebrew = np.sum((normalized_dataset - my_current_centers[my_current_clusters]) ** 2)
    # distance_matrix = kmeans.distance_metric_for_discreet_and_continuous_with_assymetric_support_v2(
    #                         dataset = normalized_dataset, 
    #                         centers = normalized_dataset, 
    #                         nominal_attributes = [0, 1, 3, 4], 
    #                         continuous_attributes = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    #                         assymetric_attributes = None, 
    #                         nplikelib = 'cupy'
    #                     )
    # silhouette_score_for_kmeans = silhouette_score(distance_matrix, my_current_clusters, metric = 'precomputed')

    print(f'SEE for sklearn for k = {k}: {SSE_for_kmeans_sklearn}')
    print(f'SEE for homebrew for k = {k}: {SSE_for_Kmeans_homebrew}', end = '\n\n')
    # print(f'Silhouette: {silhouette_score_for_kmeans}', end = '\n\n')

    reduced_all_points_my_centers = pca.fit_transform(np.vstack((normalized_dataset, my_current_centers)))
    reduced_all_points_sklearn_centers = pca.fit_transform(np.vstack((normalized_dataset, sklearn_current_centers)))

    reduced_datapoints_my_centers = reduced_all_points_my_centers[:normalized_dataset.shape[0], 🙂
    reduced_centers_my_centers = reduced_all_points_my_centers[normalized_dataset.shape[0]:, 🙂

    reduced_datapoints_sklearn_centers = reduced_all_points_sklearn_centers[:normalized_dataset.shape[0], 🙂
    reduced_centers_sklearn_centers = reduced_all_points_sklearn_centers[normalized_dataset.shape[0]:, 🙂

    colormap = plt.get_cmap('rainbow_r', k)
    
    fig = plt.figure()

    for i in range(k):
        plt.scatter(reduced_datapoints_my_centers[my_current_clusters == i, 0], reduced_datapoints_my_centers[my_current_clusters == i, 1], s = 7, alpha = 0.7, color = colormap(i / k), label = f'cl{i + 1}')

    if k < 16:
        for i in range(k):
            plt.scatter(reduced_centers_my_centers[i, 0], reduced_centers_my_centers[i, 1], s = 100, color = 'black', marker = 'X')
            plt.text(reduced_centers_my_centers[i, 0] + 0.05, reduced_centers_my_centers[i, 1] + 0.05, f"C{i + 1}", fontsize = 10, color = 'black', weight = "bold")
    else:
        for i in range(k):
            plt.scatter(reduced_centers_my_centers[i, 0], reduced_centers_my_centers[i, 1], s = 100, color = 'black', marker = 'X')
            plt.text(reduced_centers_my_centers[i, 0] + 0.05, reduced_centers_my_centers[i, 1] + 0.05, f"C{i + 1}", fontsize = 10, color = 'black', weight = "bold")
 
    if k < 16:
        plt.legend()
    plt.grid(True)
    # plt.savefig(os.path.abspath(os.path.join(os.path.dirname(_file_), '..')) + '\\kmeans_my_results' + f'\\my_k_{k}')
    plt.show()
    plt.close(fig)

    colormap = plt.get_cmap('rainbow_r', k)
    fig = plt.figure()

    for i in range(k):
        plt.scatter(reduced_datapoints_sklearn_centers[sklearn_current_clusters == i, 0], reduced_datapoints_sklearn_centers[sklearn_current_clusters == i, 1], s = 7, alpha = 0.7, color = colormap(i / k), label = f'cl{i + 1}')

    if k < 16:
        for i in range(k):
            plt.scatter(reduced_centers_sklearn_centers[i, 0], reduced_centers_sklearn_centers[i, 1], s = 100, color = 'black', marker = 'X')
            plt.text(reduced_centers_sklearn_centers[i, 0] + 0.05, reduced_centers_sklearn_centers[i, 1] + 0.05, f"C{i + 1}", fontsize = 10, color = 'black', weight = "bold")
    else:
        for i in range(k):
            plt.scatter(reduced_centers_sklearn_centers[i, 0], reduced_centers_sklearn_centers[i, 1], s = 100, color = 'black', marker = 'X')
            plt.text(reduced_centers_sklearn_centers[i, 0] + 0.05, reduced_centers_sklearn_centers[i, 1] + 0.05, f"C{i + 1}", fontsize = 10, color = 'black', weight = "bold")


    plt.legend()
    plt.grid(True)
    # plt.savefig(os.path.abspath(os.path.join(os.path.dirname(_file_), '..')) + '\\kmeans_sklearn_results' + f'\\sklearn_k_{k}')
    plt.show()
    plt.close(fig)



def multiple_k_clustering_for_sk_and_homebrew():

    dataset = pd.read_csv(path_to_dataset).values

    # discard columns not wanted
    dataset = dataset[:, 2:-1]

    # convert nominal from str to integer encoding (so it can be stored as np.float)
    dataset = normalize_vectors(
        dataset = dataset, 
        nominal_attributes = [0, 1, 3, 4], 
    )

    # discard null values
    dataset = dataset[(np.sum((dataset[:, np.arange(6, 20)] == 0), axis = 1) == 0) & (~(np.isnan(dataset[:, -1])))]

    # apply PCA on the dataset before any normalization or clustering is applied and keep only first two principle components for visualization
    pca = PCA(n_components = 2)
    reduced_dataset_without_normalization = pca.fit_transform(dataset)

    # plot and save
    plt.figure()
    plt.scatter(reduced_dataset_without_normalization[:, 0], reduced_dataset_without_normalization[:, 1], s = 20, alpha = 0.7)
    plt.grid(True)
    plt.savefig(os.path.abspath(os.path.join(os.path.dirname(_file_), '..')) + '\\dataset_after_pca' + '\\no_norm_data_pca')
    # plt.show()

    # apply normalization on the dataset
    normalized_dataset = normalize_vectors(
        dataset = dataset, 
        continuous_attributes = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    )


    my_centers = []
    sklearn_centers = []

    normalized_dataset_on_cuda = cp.asarray(normalized_dataset)

    # test a range of k values on the normalized dataset
    for k in range(16, 101):
        # ds
        # avg_time = []
        # timer_start = time.time()
        # de
        sklearn_kmeans_result = KMeans(n_clusters = k, random_state = 0).fit(normalized_dataset)
        sklearn_centers.append(
            (sklearn_kmeans_result.cluster_centers_, sklearn_kmeans_result.labels_)
        )
        my_centers.append(
            kmeans(
                k = k, 
                dataset = normalized_dataset_on_cuda, 
                dist_metric = our_similarity_metric, 
                tolerance = 1e-4, 
                # iterations = 10000, 
                nplikelib = 'cupy'
            )
        )

        # after each iteration visualize centers and clusters after applying PCA on the dataset
        # (kmeans is applied on the dataset BEFORE PCA, then PCA is applied on the resulting dataset - centers so that 
        # it can be visualized)
        my_current_centers = cp.asnumpy(my_centers[-1][0])
        my_current_clusters = cp.asnumpy(my_centers[-1][1])

        sklearn_current_centers = sklearn_centers[-1][0]
        sklearn_current_clusters = sklearn_centers[-1][1]

        SSE_for_kmeans_sklearn = np.sum((normalized_dataset - sklearn_current_centers[sklearn_current_clusters]) ** 2)
        SSE_for_Kmeans_homebrew = np.sum((normalized_dataset - my_current_centers[my_current_clusters]) ** 2)

        # distance_matrix = kmeans.distance_metric_for_discreet_and_continuous_with_assymetric_support_v2(
        #                         dataset = normalized_dataset, 
        #                         centers = normalized_dataset, 
        #                         nominal_attributes = [0, 1, 3, 4], 
        #                         continuous_attributes = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        #                         assymetric_attributes = None, 
        #                         nplikelib = 'cupy'
        #                     )
        # silhouette_score_for_kmeans = silhouette_score(distance_matrix, my_current_clusters, metric = 'precomputed')

        print(f'SEE for sklearn for k = {k}: {SSE_for_kmeans_sklearn}')
        print(f'SEE for homebrew for k = {k}: {SSE_for_Kmeans_homebrew}', end = '\n\n')

        reduced_all_points_my_centers = pca.fit_transform(np.vstack((normalized_dataset, my_current_centers)))
        reduced_all_points_sklearn_centers = pca.fit_transform(np.vstack((normalized_dataset, sklearn_current_centers)))

        reduced_datapoints_my_centers = reduced_all_points_my_centers[:normalized_dataset.shape[0], 🙂
        reduced_centers_my_centers = reduced_all_points_my_centers[normalized_dataset.shape[0]:, 🙂

        reduced_datapoints_sklearn_centers = reduced_all_points_sklearn_centers[:normalized_dataset.shape[0], 🙂
        reduced_centers_sklearn_centers = reduced_all_points_sklearn_centers[normalized_dataset.shape[0]:, 🙂

        colormap = plt.get_cmap('rainbow_r', k)
        
        fig = plt.figure()

        for i in range(k):
            plt.scatter(reduced_datapoints_my_centers[my_current_clusters == i, 0], reduced_datapoints_my_centers[my_current_clusters == i, 1], s = 7, alpha = 0.7, color = colormap(i / k), label = f'cl{i + 1}')

        if k < 16:
            for i in range(k):
                plt.scatter(reduced_centers_my_centers[i, 0], reduced_centers_my_centers[i, 1], s = 100, color = 'black', marker = 'X')
                plt.text(reduced_centers_my_centers[i, 0] + 0.05, reduced_centers_my_centers[i, 1] + 0.05, f"C{i + 1}", fontsize = 10, color = 'black', weight = "bold")
        else:
            for i in range(k):
                plt.scatter(reduced_centers_my_centers[i, 0], reduced_centers_my_centers[i, 1], s = 100, color = 'black', marker = 'X')

        if k < 16:
            plt.legend()
        plt.grid(True)
        plt.savefig(os.path.abspath(os.path.join(os.path.dirname(_file_), '..')) + '\\kmeans_my_results' + f'\\my_k_{k}')
        # plt.show()
        plt.close(fig)

        colormap = plt.get_cmap('rainbow_r', k)
        fig = plt.figure()

        for i in range(k):
            plt.scatter(reduced_datapoints_sklearn_centers[sklearn_current_clusters == i, 0], reduced_datapoints_sklearn_centers[sklearn_current_clusters == i, 1], s = 7, alpha = 0.7, color = colormap(i / k), label = f'cl{i + 1}')

        if k < 16:
            for i in range(k):
                plt.scatter(reduced_centers_sklearn_centers[i, 0], reduced_centers_sklearn_centers[i, 1], s = 100, color = 'black', marker = 'X')
                plt.text(reduced_centers_sklearn_centers[i, 0] + 0.05, reduced_centers_sklearn_centers[i, 1] + 0.05, f"C{i + 1}", fontsize = 10, color = 'black', weight = "bold")
        else:
            for i in range(k):
                plt.scatter(reduced_centers_sklearn_centers[i, 0], reduced_centers_sklearn_centers[i, 1], s = 100, color = 'black', marker = 'X')

        if k < 16:
            plt.legend()
        plt.grid(True)
        plt.savefig(os.path.abspath(os.path.join(os.path.dirname(_file_), '..')) + '\\kmeans_sklearn_results' + f'\\sklearn_k_{k}')
        # plt.show()
        plt.close(fig)



if _name_ == '_main_':

    multiple_k_clustering_for_sk_and_homebrew()