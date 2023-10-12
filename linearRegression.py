import pandas as pd
import quandl as qd
import math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

qd.ApiConfig.api_key = "DZzrugx1RoYdcZ-Yd-6M"
data = qd.get('WIKI/GOOGL')
data = data[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
data['HL_%C'] = (data['Adj. High'] - data['Adj. Close']) / data['Adj. Close'] * 100
data['%C'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100

#               Price
data = data[['Adj. Close', 'HL_%C','%C','Adj. Volume',]]

forecast_col = 'Adj. Close'

data.fillna(-99999, inplace=True)
#regression used to predict stuff such as stock prices
forecast_out = int(math.ceil(0.01*len(data)))
#this is how far out I want to predict
#print(forecast_out)

data['label'] = data[forecast_col].shift(-forecast_out)


#print(data)
#features
X = np.array(data.drop(['label'],axis = 1))
#scale the data 
X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]


data.dropna(inplace=True)
#label
Y = np.array(data['label'])

# X_train = X[:-2 * forecast_out]
# Y_train = Y[forecast_out:-forecast_out]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y, test_size = .2)
#this is the algorithm
#clf = svm.SVR(kernel = 'poly')
clf = LinearRegression(n_jobs=-1)
#clf stands for classifier
clf.fit(X_train, Y_train)
#save the classifier with pickling
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf,f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, Y_test)

#print(accuracy)

forecast_set = clf.predict(X_lately)
#forecast_out is the number of days out to predict
print(forecast_set, accuracy, forecast_out)

#a bunch of empty data 
data['Forecast'] = np.nan

last_date = data.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    #nan stands for not a number or undefined values
    # creating a list of empyt values
    # reason to add [i] is to add another column
    data.loc[next_date] = [np.nan for _ in range(len(data.columns)-1)] + [i]

data['Adj. Close'].plot()
data['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Create a separate plot for the forecasted data
# plt.twinx()
# plt.plot(data.index[-forecast_out:], forecast_set, 'g', label='Forecast')
# plt.legend(loc=2)
# plt.ylabel('Price (Forecast)')

# plt.show()