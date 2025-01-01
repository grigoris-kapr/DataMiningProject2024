import os

import numpy as np
import pandas as pd



#! CONFIGURE PROPERLY PATH FOR THE DATASET FILE
path_to_dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) + '\\Datasets\\Airline_Passenger_Satisfaction' + '\\train.csv'




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
        initial_centers = np.random.rand(dataset_size, datapoint_dim)

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
                



if __name__ == '__main__':

    dataset = pd.read_csv(path_to_dataset, header = 1).values

    #! TODO : CONTINUE HERE