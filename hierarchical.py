
import numpy as np
import pandas as pd

list_of_attributes = [

]
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
            else:
                sum_dist += abs( x[attr] - y[attr] ) 
    return sum_dist / sum_d 


# type: select which algorithm to implement, can be "min" or "max"
def hierarcical(type="min"):
    # do sth
    return 0

# read data into data
data = pd.read_csv('train.csv', sep=',',header=0).values

# drop unused data
data = np.delete(data, [0, 1, 24], 1)

# replace 0s in satisfaction ratings with None
for i in range(len(data)):
    for attr in range(len(i)):
        if attr >= 6 and attr <= 19 and data[i][attr] == 0:
            data[i][attr] = None


