import random

satisfaction_attributes = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Check-in service",
    "Inflight service",
    "Cleanliness"
]

def rand_gender():
    if random.randint()%2 == 0:
        return "Female"
    return "Male"
    
def rand_customer():
    if random.randint()%2 == 0:
        return "Loyal Customer"
    return "disloyal Customer"
    
def rand_age():
    return random.randrange(7, 85)

def rand_travel_type():
    if random.randint()%2 == 0:
        return "Business travel"
    return "Personal Travel"

def rand_class():
    if random.randint()%2 == 0:
        return "Business"
    elif random.randint()%10 > 0:
        return "Eco"
    return "Eco Plus"

def rand_distance():
    return random.randrange(31, 4983)

# include_nulls: selects whether the satisfactions can be null (0 is interpreted as null).
#   set to 0 to not include nulls, set to 1 to include nulls
def rand_satisfaction(include_nulls = 0):
    return random.randint()%(5+include_nulls) + 1 - include_nulls

# approximately exponential distribution, mean value for lambd calculated from train dataset
def rand_dep_delay():
    random.expovariate(1.0/14.8)
# approximately exponential distribution, mean values for lambd calculated from train dataset
def rand_arr_delay():
    return random.expovariate(1.0/15.2)

# creates random point
# include_nulls: see rand_satisfaction
def create_rand_point(include_nulls = 0):
    datapoint = {}
    # random gender
    datapoint["Gender"] = rand_gender()
    # random customer type
    datapoint["Customer Type"] = rand_customer()
    # random age (uniform in same range as original dataset)
    datapoint["Age"] = rand_age()
    # random travel type
    datapoint["Type of Travel"] = rand_travel_type()
    # radnom class
    datapoint["Class"] = rand_class()
    # random distance
    datapoint["Flight distance"] = rand_distance()
    # random various satisfactions
    for attribute in satisfaction_attributes:
        datapoint[attribute] = rand_satisfaction(include_nulls=include_nulls)
    # random departure & arrival delay 
    datapoint["Departure Delay in Minutes"] = rand_dep_delay()
    datapoint["Arrival Delay in Minutes"] = rand_arr_delay()

    return datapoint

# size: the number of points in this random cluster (including source), default is 10
# change_rate: used to determine how many attributes will be different, lower values lead to 
#   more attributes changing, must be an int
# include_nulls: see rand_satisfaction
def create_rand_cluster_based_on_point(size = 10, change_rate = 10, include_nulls = 0):
    source = create_rand_point()
    datapoints = {} * (size-1)
    for point in datapoints:
        point = source.copy()
        for attr in source.keys():
            if random.randint()%change_rate == 0:
                if attr == "Gender":
                    point[attr] = rand_gender()
                elif attr == "Customer Type":
                    point[attr] = rand_customer()
                elif attr == "Age":
                    point[attr] = rand_age()
                elif attr == "Type of Travel":
                    point[attr] = rand_travel_type()
                elif attr == "Class":
                    point[attr] = rand_class()
                elif attr == "Flight distance":
                    point[attr] = rand_distance()
                elif attr in satisfaction_attributes:
                    point[attr] = rand_satisfaction(include_nulls=include_nulls)
                elif attr == "Departure Delay in Minutes":
                    point[attr] = rand_dep_delay()
                elif attr == "Arrival Delay in Minutes":
                    point[attr] = rand_arr_delay()
                else:
                    print("Encountered unknown attribute: " + attr)
    datapoints.add(source)
    return datapoints


    



