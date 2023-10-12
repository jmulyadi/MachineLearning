import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

accuracies = []
for i in range(25):
    #df is dataframe
    df = pd.read_csv('data/breast-cancer-wisconsin.data')
    #will recognize -99999 as an outlier
    df.replace('?', -99999, inplace=True)
    #1 means to do column 0 is for row
    df.drop(['id'], axis = 1, inplace=True) 

    #features
    X = np.array(df.drop(['class'], axis = 1))
    #labels
    y = np.array(df['class'])
    #test size is .2 or 20%
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size = .2)

    clf = neighbors.KNeighborsClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    #print(accuracy)

    example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,4,4,4,4,16,8,8,4]])

    example_measures = example_measures.reshape(len(example_measures),-1)

    prediction = clf.predict(example_measures)
    #print(prediction)
    accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))