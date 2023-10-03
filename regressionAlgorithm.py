from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random 

style.use('fivethirtyeight')

#default is np.float64
# xs = np.array([1,2,3,4,5,6], dtype = np.float64)
# ys = np.array([5,4,6,5,6,7], dtype = np.float64)

#hm is how much data, variance for spread out, how far to move up the y value
# correlation how related the data is
def create_dataset(hm, variance, step = 2, correlation = 'False'):
    #orignal value of y
    val = 1
    ys = []
    for _ in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    xs = [i for i in range(hm)]
    return np.array(xs, dtype = np.float64), np.array(ys, dtype = np.float64)

xs,ys = create_dataset(40, 40, 2, 'neg')

def best_fit_slope(xs, ys):
    m = ((mean(xs)*mean(ys)-mean(xs*ys))/(
        mean(xs)**2 - mean(xs**2)))
    return m

m = best_fit_slope(xs,ys)

def find_intercept(xs,ys,m):
    b = mean(ys) - m*mean(xs)
    return b

b = find_intercept(xs,ys,m)

def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line =[mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_mean= squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_mean)

regression_line = [(m*x)+b for x in xs]
#can predict data
predict_x = 8
predict_y = (m*predict_x)+b
#find the error or r^2 value how accurate the best fit line is
r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs,ys)
#extra point to check a point on the regression line
plt.scatter(predict_x,predict_y, s = 100, color = 'g')
plt.plot(xs,regression_line)
plt.show()