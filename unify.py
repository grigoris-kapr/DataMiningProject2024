# import cupy as cp 
import heapq
import math
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.linalg
import scipy.spatial
import sklearn
import sklearn.cluster
import sklearn.decomposition
import cupy as cp

def normalize_vectors(
    dataset: np.ndarray,  # 2 dimensional ndarray, 
    nominal_attributes: list, 
    continuous_attributes: list
) -> np.ndarray:
    
    normalized_dataset = np.zeros_like(dataset, dtype = 'float32')
    
    min_values = np.min(dataset[:, continuous_attributes], axis = 0).astype(np.float32)
    range_of_attributes = np.max(dataset[:, continuous_attributes], axis = 0).astype(np.float32) - min_values

    # in case an attribute has the same value for all vectors, don't let a division by zero occur
    range_of_attributes[range_of_attributes == 0] = 1.0

    normalized_dataset[:, continuous_attributes] = (dataset[:, continuous_attributes].astype(np.float32) - min_values) / (range_of_attributes)

    for col in nominal_attributes:
        _, indices = np.unique(dataset[:, col], return_inverse = True)
        normalized_dataset[:, col] = indices.astype(np.float32)


    return normalized_dataset

name_to_lib = {
    'numpy' : np, 
    'cupy' : cp
}

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
    dataset = dataset[:, None, :]
    centers = centers[None, :, :]
    

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

def our_similarity_metric(dataset, centers):
    return distance_metric_for_discreet_and_continuous_with_assymetric_support(
        dataset = dataset, 
        centers = centers, 
        nominal_attributes = [0, 1, 3, 4], 
        continuous_attributes = np.setdiff1d(np.arange(dataset.shape[1]), np.array([0, 1, 3, 4])).tolist(), 
        assymetric_attributes = None
    )

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
    start_dist = time.time()
    # Create distance matrix
    distance_matrix = np.full((n,n), np.inf, dtype=np.float16)
    for i in range(n):
        interim_distance_matrix = distance_metric_for_discreet_and_continuous_with_assymetric_support(
            data[i:i+1],
            data,
            nominal_attributes = [], 
            continuous_attributes = [0, 1, 2], # it is run after PCA, so they act as continuous
            assymetric_attributes=None
        )
        distance_matrix[i] = interim_distance_matrix[0]
    end_dist = time.time()
    print("Dist. Matrix calculation time: ", end_dist - start_dist)


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
        if algorithm == True:
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
                    heapq.heappush(heap, (-max_cluster_dist, cl1, cl2))

    # initialize id for next cluster to be created and iterate on every loop
    # this is slightly faster than len(clusters) and also works with deleting old clusters
    new_cluster_id = len(clusters) - 1 # -1 because it's iterated before reference

    while len(clusters) > clusters_num:
        # find clusters to merge
        _, i_min, j_min = heapq.heappop(heap)
        if not i_min in clusters.keys() or not j_min in clusters.keys(): 
            continue

        # create the new cluster for distance calculations
        new_cluster = np.concatenate((clusters[i_min], clusters[j_min]), axis=0)
        new_cluster_id += 1
        # and delete old ones
        del clusters[i_min]
        del clusters[j_min]

        if algorithm == True:
            for cluster_id in clusters.keys():
                # calculate distance between new_cluster and every other cluster
                min_dist = np.min(distance_matrix[clusters[cluster_id]].transpose()[new_cluster])
                min_id = min(new_cluster_id, cluster_id)
                max_id = max(new_cluster_id, cluster_id)
                heapq.heappush(heap, (min_dist, max_id, min_id))
        else:
            for cluster_id in clusters.keys():
                # calculate distance between new_cluster and every other cluster
                max_dist = np.max(distance_matrix[clusters[cluster_id]].transpose()[new_cluster])
                min_id = min(new_cluster_id, cluster_id)
                max_id = max(new_cluster_id, cluster_id)
                heapq.heappush(heap, (-max_dist, max_id, min_id))

        # add now so as not to exclude in the earlier loops
        clusters[new_cluster_id] = new_cluster.copy()
    
    
    return list(clusters.values())

# because Global_keep_columns may change, always use this to keep these lists with up-to-date indexes
def current_attribute_selection_asym_nom_cont():
    asymmetric_cols = []
    nominal_cols = []
    continuous_cols = []
    for col_new_index in range(len(Global_keep_columns)):
        original_index = Global_keep_columns[col_new_index]
        if original_index in Global_asymmetric_cols:
            asymmetric_cols.append(col_new_index)
        if original_index in Global_continuous_cols:
            continuous_cols.append(col_new_index)
        if original_index in Global_nominal_cols:
            nominal_cols.append(col_new_index)
    
    return asymmetric_cols, nominal_cols, continuous_cols

def import_and_preprocess_data(filename):
    dataset_with_extra_cols = pd.read_csv(filename, header = 0).values

    # discard columns not wanted
    dataset = dataset_with_extra_cols[:, Global_keep_columns]
    # create the three lists below to pass as parameters for normalization
    asymmetric_cols, nominal_cols, continuous_cols = current_attribute_selection_asym_nom_cont()

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
        nominal_attributes = nominal_cols, 
        continuous_attributes = continuous_cols
    )

    return normalized_dataset

def plot_clustering(
        data: np.ndarray, 
        clustering: list[list[int]],
        enfornce3D: bool = True # by default, runs PCA to limit the data to 3 dimentions.
                                # setting this to False will produce a 3D visual of the data 
                                # with whichever happen to be the first three dimentions
    ):
    # First, make sure data is 3 dimentional if enforce3D is enabled
    if enfornce3D == True:
        data = sklearn.decomposition.PCA(n_components=3).fit_transform(data)

    plt.figure()
    ax = plt.axes(projection ='3d')

    cl_num = 0
    marker = 'o'
    for cluster in clustering:
        # change marker every 10 clusters so that each cluster has a unique shape + color combination
        # (matplotlib cycles between 10 colors if none are specified)
        if cl_num == 10: marker = 's'
        elif cl_num == 20: marker = 'v'
        elif cl_num == 30: marker = 'x'
        elif cl_num == 40: marker = ','
        cl_num += 1
        ax.scatter(data[cluster, 0], data[cluster, 1], data[cluster, 2], s=20, alpha=0.7, marker=marker)

    plt.grid(True)
    plt.show()

def find_outliers_in_clustered_data(data, clustering):
    cov_inv = scipy.linalg.inv(np.cov(data))
    mh_distances = np.zeros(len(data))
    for cluster in clustering:
        cov_inv = scipy.linalg.inv(np.cov(data[cluster], rowvar=False))
        print("calculating a cluster...")
        centroid = np.mean(data[cluster], axis=0)
        for index in cluster:
            mh_distances[index] = scipy.spatial.distance.mahalanobis(data[index], centroid, cov_inv)
    sorted_dists = np.sort(mh_distances)
    # Plot outliers
    plt.figure()
    plt.plot(range(1, len(data)+1), sorted_dists, marker='o')
    plt.title('Mahalanobis Distances in ascending order')
    plt.show()

    # set manually by looking at the plot above
    threshold = 3.0
    outliers = []
    for index in range(len(data)):
        if mh_distances[index] > threshold:
            outliers.append(index)
            for cluster in clustering:
                if index in cluster:
                    # delete index from cluster it used to be in
                    cluster = np.delete(cluster, np.where(cluster == index))

    clustering.append(outliers)

    return outliers, clustering
                    


# The variable below is used to change the feature selection. Best clustering results 
# were visually seen with a value of  [4, 5, 6, 7, 8, 9, 10, 14, 22, 23]. The variables
# Global_nominal_cols, Global_continuous_cols, and Global_asymmetric_cols
# all need to be used as an intersection with this variable. 
# E.g. nominal_attributes = list(Global_keep_columns.intersection(Global_nominal_cols))
Global_keep_columns = [
    # 0, # (no label)
    # 1, # id
    # 2, # Gender
    # 3, # Customer Type
    4, # Age
    5, # Type of Travel
    6, # Class
    7, # Flight Distance
    8, # Inflight wifi service
    9, # Departure/Arrival time convenient
    10, # Ease of Online booking
    11, # Gate location
    # 12, # Food and drink
    # 13, # Online boarding
    14, # Seat comfort
    # 15, # Inflight entertainment
    # 16, # On-board service
    # 17, # Leg room service
    # 18, # Baggage handling
    # 19, # Checkin service
    # 20, # Inflight service
    # 21, # Cleanliness
    22, # Departure Delay in Minutes
    23, # Arrival Delay in Minutes
    # 24  # satisfaction (pre-existing classification, always unused)
]

Global_asymmetric_cols = [
    22, # Departure Delay in Minutes
    23, # Arrival Delay in Minutes
]

Global_nominal_cols = [
    1, # id
    2, # Gender
    3, # Customer Type
    5, # Type of Travel
    6, # Class
    24  # satisfaction (pre-existing classification, always unused)
]

Global_continuous_cols = [
    4, # Age
    7, # Flight Distance
    8, # Inflight wifi service
    9, # Departure/Arrival time convenient
    10, # Ease of Online booking
    11, # Gate location
    12, # Food and drink
    13, # Online boarding
    14, # Seat comfort
    15, # Inflight entertainment
    16, # On-board service
    17, # Leg room service
    18, # Baggage handling
    19, # Checkin service
    20, # Inflight service
    21, # Cleanliness
    22, # Departure Delay in Minutes
    23, # Arrival Delay in Minutes
]

if __name__ == '__main__':

    normalized_dataset = import_and_preprocess_data("test.csv")

    normalized_dataset = sklearn.decomposition.PCA(n_components=3).fit_transform(normalized_dataset)
    
    # =================================================================================================
    # Where the magic happens
    k1 = 30 # np.ceil(np.sqrt(len(normalized_dataset))).astype(np.int32)
    k2 = 6

    _, kmeans_output = kmeans(
        k1,
        normalized_dataset,
        dist_metric=our_similarity_metric
        tolerance=1e-4, 
        # iterations = 10000, 
        nplikelib = 'cupy'
    )

    kmeans_clusters = clusters_kmeans_to_hierarchical(kmeans_output)

    # Time and plot hierarchical alg. to bring the clusters down from k1 to k2
    start = time.time()
    hierarchical_output = fast_hierarchical(
        normalized_dataset, 
        clusters_num=k2,
        algorithm=True,
        initial_clusters=kmeans_clusters
    )
    end = time.time()
    print(end - start)
    plot_clustering(normalized_dataset, hierarchical_output)

    # PLOT IT ALL
    # Plot every iteration of the hierarchical algorithm, starting with 
    # the 0-th (the output of k-means)
    # cluster colors and symbols don't stay the same between iterations 
    # (will probably need changing the plot_clustering function)
    """ current_clusters = kmeans_clusters
    plot_clustering(normalized_dataset, current_clusters)
    for i in range(k1-1, k2-1, -1):
        current_clusters = fast_hierarchical(
            normalized_dataset,
            clusters_num=i,
            algorithm=False,
            initial_clusters=current_clusters
        )
        plot_clustering(normalized_dataset, current_clusters)
 """
    outliers, clustering_with_outliers = find_outliers_in_clustered_data(normalized_dataset, hierarchical_output)

    plot_clustering(normalized_dataset, clustering_with_outliers)
    
        