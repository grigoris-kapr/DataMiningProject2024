# import cupy as cp 
import heapq
import math
import time
import numpy as np
import pandas as pd
import scipy
import scipy.spatial
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

name_to_lib = {
    'numpy' : np, 
    # 'cupy' : cp
}

def distance_metric_for_discreet_and_continuous_with_assymetric_support(
    dataset: np.ndarray, #| cp.ndarray,  # 2 dimensional ndarray of shape (num of datapoints, num of attributes)
    centers: np.ndarray, #| cp.ndarray,  # 2 dimensional ndarray of shape (num of centers, num of attributes)
    nominal_attributes: list,  # a list containing the indices of nominal value attributes
    continuous_attributes: list,  # a list containing the indices of continuous value attributes 
    assymetric_attributes: list | None,  # a list containing the indices of assymetric value attributes
    nplikelib: str = None,  # either 'numpy' or 'cupy' depending on whether cuda is supported and code needs to run on a gpu
) -> np.ndarray  :#| cp.ndarray:  # 2 dimensional ndarray containing the similarity between each datapoint and each center, shape: (num of datapoint, num of centers)

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

def clusters_kmeans_to_hierarchical(labels):
    num_clusters = max(labels) + 1
    clusters = [ [] for _ in range(num_clusters) ]
    for i in range(len(labels)):
        clusters[labels[i]].append(i)

    return clusters

def fast_hierarchical(
        data:np.ndarray,
        clusters_num, # the number of clusters at which the algorithm stops
        algorithm:bool = True, # True for single link/min, False for complete link/max
        initial_clusters: list[list[int]] = None # None by default, can provide initial clusters (from k-means)
):
    n = len(data)
    
    # Create distance matrix
    distance_matrix = np.full((n,n), np.inf, dtype=np.float16)
    for i in range(n//2):
        interim_distance_matrix = distance_metric_for_discreet_and_continuous_with_assymetric_support(
            data,
            data[2*i:2*i+1],
            nominal_attributes=[0, 1, 3, 4], 
            continuous_attributes=[2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            assymetric_attributes=None
        )
        distance_matrix[2*i] = interim_distance_matrix[0]
        distance_matrix[2*i+1]
    if n%2 != 0:
        distance_matrix[n-1] = distance_metric_for_discreet_and_continuous_with_assymetric_support(
            data,
            data[n-2:n-1],
            nominal_attributes=[0, 1, 3, 4], 
            continuous_attributes=[2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            assymetric_attributes=None
        )[1]
        print(distance_matrix[n-1])


    # add distances to a heap for complexity speed-up
    # distances to be added depend on initial_clusters and are negative for complete link
    heap = []

    # STEP 1: each cluster is stored as a dict int -> list[int]
    # the keys are the cluster ids and the items are datapoint indexes. when there are no initial_clusters 
    # provided, the cluster ids are initialized to the same as the index in the one-point list
    # STEP 2: add cluster distances to heap (entries in the data matrix when no initial_clusters provided; 
    # cluster distances when initial_clusters are provided)
    if initial_clusters == None:
        # STEP 1
        clusters = { i: [i] for i in range(n)}
        # STEP 2
        if algorithm == True:
            for i in range(n):
                for j in range(i):
                    heapq.heappush(heap, (distance_matrix[i][j], i, j))
        else:
            for i in range(n):
                for j in range(i):
                    # negative because heapq always uses min
                    heapq.heappush(heap, (-distance_matrix[i][j], i, j))

    else:
        # STEP 1
        clusters = { i: [] for i in range(len(initial_clusters)) }
        for cl_id in range(len(initial_clusters)):
            clusters[cl_id] = initial_clusters[cl_id].copy()
        # STEP 2
        if type == True:
            for cl1 in range(len(clusters)):
                for cl2 in range(cl1):
                    min_cluster_dist = np.min(distance_matrix[clusters[cl1]].transpose()[clusters[cl2]])
                    heapq.heappush(heap, (min_cluster_dist, cl1, cl2))
        else:
            for cl1 in range(len(clusters)):
                for cl2 in range(cl1):
                    # max dist instead of min
                    max_cluster_dist = np.max(distance_matrix[clusters[cl1]].transpose()[clusters[cl2]])
                    # negativ because heapq always uses min
                    heapq.heappush(heap, (max_cluster_dist, cl1, cl2))

    # initialize id for next cluster to be created and iterate on every loop
    # this is slightly faster than len(clusters) and also works with deleting old clusters
    new_cluster_id = len(clusters) - 1 # -1 because it's iterated before reference

    while len(clusters) > clusters_num:
        # find clusters to merge
        _, i_min, j_min = heapq.heappop(heap)
        if not i_min in clusters.keys() or not j_min in clusters.keys(): 
            continue

        # add the new cluster to the list 
        new_cluster = np.concatenate((clusters[i_min], clusters[j_min]), axis=0)
        new_cluster_id += 1
        # and delete old ones
        del clusters[i_min]
        del clusters[j_min]

        if algorithm == True:
            for cluster_id in clusters.keys():
                # calculate distance between new_cluster and every other cluster
                min_dist = np.min(distance_matrix[clusters[cluster_id]].transpose()[new_cluster])
                heapq.heappush(heap, (min_dist, new_cluster_id, cluster_id))
        else:
            for cluster_id in clusters.keys():
                # calculate distance between new_cluster and every other cluster
                max_dist = np.max(distance_matrix[clusters[cluster_id]].transpose()[new_cluster])
                heapq.heappush(heap, (-max_dist, new_cluster_id, cluster_id))

        # add now so as not to exclude in the earlier loops
        clusters[new_cluster_id] = new_cluster

    return clusters

if __name__ == '__main__':

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
    
    """ optimized_kmeans(
        k = 1, 
        dataset = normalized_dataset, 
        dist_metric = our_similarity_metric, 
        iterations = 500
    )  """

    kmeans_output = sklearn.cluster.KMeans(
        n_clusters= np.ceil(np.sqrt(np.sqrt(len(normalized_dataset)))).astype(np.int32),
        init='random'
        ).fit(normalized_dataset)

    kmeans_clusters = clusters_kmeans_to_hierarchical(kmeans_output.labels_)
    print("Clusters from kmeans:")
    for i in range(len(kmeans_clusters)):
        print(i, ": ", kmeans_clusters[i])

    # for cluster in kmeans_clusters: print(cluster)
    start = time.time()
    hierarchical_output = fast_hierarchical(
        normalized_dataset, 
        clusters_num=5,
        algorithm=True,
        initial_clusters=kmeans_clusters
    )
    end = time.time()
    print(end - start)

    print("Final output:")
    # for cluster in hierarchical_output.items(): print(cluster)
    print(len(normalized_dataset))