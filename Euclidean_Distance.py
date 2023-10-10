import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    #knnalgos

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
    #votes becomes a list of lists with 2 elements each
    #first is the distance then the group to sort it
    votes = [i[1] for i in sorted(distances)[:k]]
    #gets the most common group
    #most_common returns a tuple of the group and distance
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result
#data frame
df = pd.read_csv("data/breast-cancer-wisconsin.data")
df.replace('?', -99999, inplace = True)
df.drop(['id'], axis = 1, inplace = True)
#convert everything to a float some were strings
full_data = df.astype(float).values.tolist()
#shuffle up the data
random.shuffle(full_data)

#how much of the data to test percentagewise
test_size = .2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k = 5)
        if group == vote:
            correct+=1
        total+=1
print('Accuracy:', correct/total)