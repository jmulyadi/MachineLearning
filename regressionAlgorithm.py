from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

#default is np.float64
xs = np.array([1,2,3,4,5,6], dtype = np.float64)
ys = np.array([5,4,6,5,6,7], dtype = np.float64)

def best_fit_slope(xs, ys):
    m = ((mean(xs)*mean(ys)-mean(xs*ys))/(
        mean(xs)**2 - mean(xs**2)))
    return m

m = best_fit_slope(xs,ys)

def find_intercept(xs,ys,m):
    b = mean(ys) - m*mean(xs)
    return b

b = find_intercept(xs,ys,m)

regression_line = [(m*x)+b for x in xs]
#can predict data
predict_x = 8
predict_y = (m*predict_x)+b

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, color = 'g')
plt.plot(xs,regression_line)
plt.show()