import heapq
import time
import numpy as np
import pandas as pd

def is_asymetric(attribute):
    return attribute == 20 or attribute == 21 # only delays are asymetric

def is_nominal(attribute):
    return attribute in [0, 1, 3, 4]

# custom distance function
def dist(x, y):
    sum_d = 0 # number of attributes considered
    sum_dist = 0
    for attr in range(22):
        if not ((is_asymetric(attr) == True and x[attr] == 0 and y[attr] == 0 ) or x[attr] == None or y[attr] == None):
            sum_d += 1
            if is_nominal(attr) == True and x[attr] != y[attr]:
                sum_dist += 1
            elif not is_nominal(attr):
                
                sum_dist += abs( x[attr] - y[attr] ) 
    return sum_dist / sum_d 

def find_min_distance(distance_matrix):
    i_min = -1
    j_min = -1
    minimum = np.inf
    i_min, j_min = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
    return i_min, j_min

# data: the data to run on
# clusters_num: the number of clusters at which the algorithm stops
# type: select which algorithm to implement, True for min False for max
# clusters: None by default, can provide initial clusters (from k-means)
def hierarchical(data, clusters_num, type=True, clusters=None):
    n = len(data)
    # calculate proximity matrix
    distance_matrix = np.full((n, n), np.inf, dtype=np.float16)
    for i in range(n):
        for j in range(i):
            distance_matrix[i][j] = dist(data[i], data[j])
        if i%100==0: print(i)
    
    # keep a copy of the original distance matrix to save on computaions
    # comment out if out of memory (re-calculate manually)
    full_distance_matrix = distance_matrix.copy()
    print("Successfully allocated memory for full_distance_matrix")

    # each cluster is stored as a list of indexes to all the points belonging in that cluster
    # initialize with all points as discrete clusters
    if clusters == None: # only initialize if no clusters provided
        clusters = [ [i] for i in range(n) ]
    # merging loop
    while len(clusters) > clusters_num:
        # find clusters to merge: j will be merged into i
        i_min, j_min = find_min_distance(distance_matrix)
        # always merge smaller cluster onto the bigger one for more efficient distance_matrix calculations
        if len(clusters[j_min]) > len(clusters[i_min]):
            temp = i_min
            i_min = j_min
            j_min = temp

        # merge clusters and keep a copy of the one that was merged into the other
        clusters[i_min].extend(clusters[j_min])
        merged_cluster = clusters[j_min].copy()
        del clusters[j_min]

        # delete cluster j from distance matrix (i will be overwritten)
        distance_matrix = np.delete(distance_matrix, j_min, 0) # delete j-th row
        distance_matrix = np.delete(distance_matrix, j_min, 1) # delete j-th col

        # in case i_min > j_min, reference to [i_min] will now be broken (might even cause OOB error)
        # in that case, i_min should be reduced by 1 (because the row/col deleted came before it)
        if i_min > j_min: i_min -= 1

        # update matrix for single-link
        if type == True:
            # update i-th row of distance matrix
            for col in range(i_min):
                # distance is minimum between every pair of points in clusters[i_min] & clusters[col]
                # clusters[i_min] is comprised of the old clusters[i_min] and the merged_cluster
                # min of the old clusters[i_min] is old_best. only need to calculate new_best from merged_cluster and take min of both
                old_best = distance_matrix[i_min][col]
                new_best = min(full_distance_matrix[p1][p2] for p2 in clusters[col] for p1 in merged_cluster)
                distance_matrix[i_min][col] = min(old_best, new_best)
            # update i-th col of distance matrix
            for row in range(i_min+1, len(clusters)):
                # distance is minimum between every pair of points in clusters[i_min] & clusters[row]
                # clusters[i_min] is comprised of the old clusters[i_min] and the merged_cluster
                # min of the old clusters[i_min] is old_best. only need to calculate new_best from merged_cluster and take min of both
                old_best = distance_matrix[row][i_min]
                new_best = min( full_distance_matrix[p1][p2] for p2 in merged_cluster for p1 in clusters[row])
                distance_matrix[row][i_min] = min(old_best, new_best)
        
        # update matrix for complete link
        elif type == False:
            # update i-th row of distance matrix
            for col in range(i_min):
                # distance is maximum between every pair of points in clusters[i_min] & clusters[col]
                # clusters[i_min] is comprised of the old clusters[i_min] and the merged_cluster
                # min of the old clusters[i_min] is old_best. only need to calculate new_best from merged_cluster and take min of both
                old_best = distance_matrix[i_min][col]
                new_best = max(full_distance_matrix[p1][p2] for p2 in clusters[col] for p1 in merged_cluster)
                distance_matrix[i_min][col] = max(old_best, new_best)
            # update i-th col of distance matrix
            for row in range(i_min+1, len(clusters)):
                # distance is maximum between every pair of points in clusters[i_min] & clusters[row]
                # clusters[i_min] is comprised of the old clusters[i_min] and the merged_cluster
                # min of the old clusters[i_min] is old_best. only need to calculate new_best from merged_cluster and take min of both
                old_best = distance_matrix[row][i_min]
                new_best = max( full_distance_matrix[p1][p2] for p2 in merged_cluster for p1 in clusters[row])
                distance_matrix[row][i_min] = max(old_best, new_best)

        # if type is neither, throw error
        else:
            raise ValueError("'type' must be boolean for either complete link (false) or single ling (default/true)")

    return clusters

def fast_hierarchical(data, clusters_num, type=True, clusters=None):
    n = len(data)
    
    # Create an array to hold the distances 
    distance_matrix = np.full((n, n), np.inf, dtype=np.float16) 
    # calculate distances and add to a heap for complexity speed-up
    heap = []
    for i in range(n):
        for j in range(i):
            distance_matrix[i][j] = dist(data[i], data[j])
            # add to heap (negative for complete link as heapq uses min)
            if type == True: 
                heapq.heappush(heap, (distance_matrix[i][j], i, j))
            else:
                heapq.heappush(heap, (-distance_matrix[i][j], i, j))
        if i%100==0: print(i)

    # each cluster is stored as a set of indexes to all the points belonging in that cluster
    # set is useful as we can utilize the union function + deprecate clusters (faster than other method)
    # initialize with all points as discrete clusters
    if clusters == None: # only initialize if no clusters provided
        clusters = [ {i} for i in range(n) ]

    while len(clusters) - clusters.count(set()) > clusters_num:
        # find clusters to merge
        _, i_min, j_min = heapq.heappop(heap)
        if not clusters[i_min] or not clusters[j_min]: continue
        if len(clusters[j_min]) > len(clusters[i_min]):
            temp = i_min
            i_min = j_min
            j_min = temp

        # append the new cluster to the list and delete old ones
        new_cluster = clusters[i_min].union(clusters[j_min])
        clusters.append(new_cluster)
        clusters[i_min] = set()
        clusters[j_min] = set()

        for cl in range(len(clusters) - 1):
            if clusters[cl]:
                
                # new_distance = 
                new_distance = min(distance_matrix[point1][point2] 
                                   for point1 in clusters[cl] 
                                   for point2 in clusters[len(clusters)-1] 
                                   )
                # use trig matrix
                if cl > len(clusters):
                    pair = (cl, len(clusters)-1)
                else:
                    pair = (len(clusters)-1, cl)
                heapq.heappush(heap, (new_distance, pair[0], pair[1]))
                # heapq.heappush(heap, (new_distance, cl, len(clusters)-1))

    final_clusters = []
    for cluster in clusters:
        if cluster != set(): 
            final_clusters.append(cluster)
    return final_clusters

# read data into data
data = pd.read_csv('test_2000.csv', sep=',',header=0).values

# drop unused data
data = np.delete(data, [0, 1, 24], 1)

# replace 0s in satisfaction ratings with None
for i in range(len(data)):
    for attr in range(22):
        if attr >= 6 and attr <= 19 and data[i][attr] == 0:
            data[i][attr] = None

start = time.time()
output1 = hierarchical(data, 9)
end = time.time()
print(end-start)

# start = time.time()
# output2 = fast_hierarchical(data, 9)
# end = time.time()
# print(end-start)

# print(output1)
# print(output2)