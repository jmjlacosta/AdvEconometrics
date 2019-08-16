import numpy as np  #general library, will come in handy later
import pandas as pd  #another nice library for storing matrices, it rely's on numpy

import matplotlib.pyplot as plt  #this library is for graphing things

from sklearn.linear_model import Lasso  #These libraries have the necessary models
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
#load data
df = pd.read_csv('credit.csv')
print(df.describe())
print(df.mean())
#get the variable names in a list
column_names = df.columns.values[1:-1]  #select the columns we want
df_extended = df.copy()  #make a copy to edit

for column in column_names:
    interaction_name = 'Limit*%s' % column
    df_extended[interaction_name] = df_extended[column] * df_extended['Limit']

print(df_extended.mean())
#generate some fake data
nobs = 1000

#the first coefficient is much more important than the second
beta = np.array([10, 1])
X = np.random.random((nobs, 2))
e = np.random.random(nobs)
y = 1 + np.dot(X, beta) + e

#fit the lasso to it, notice the second parameter is 0
#why do you think that is?
lasso = Lasso(alpha=.5)  #note that alpha corresponds to lambda
lasso.fit(X, y)
print(lasso.coef_)
print(lasso.intercept_)

#prepare fitted data to compare using MSE function

fitted_y = lasso.predict(X)
print(fitted_y.mean())

for i in range(5):
    #compute start/end of fold
    start_index = int((nobs / 5)*i)
    end_index = int((nobs / 5)*(i + 1))

    #partition data
    X_test = X[start_index:end_index]
    y_test = y[start_index:end_index]

    X_train = np.concatenate((X[0:start_index], X[end_index:]))
    y_train = np.concatenate((y[0:start_index], y[end_index:]))

    #estimate model
    l = Lasso(alpha=.5)
    l.fit(X_train, y_train)
    print('Fold %s, Coefficients: %s, Intercept: %s \n' % (i, l.coef_,
                                                           l.intercept_))

#from here you can figure out CV_n

#Specify a grid of lambda values to optimize over.
lambda_values = .1 * np.array(range(1, 5))

lass_cv = LassoCV(cv=5, alphas=lambda_values)
lass_cv = lass_cv.fit(X, y)

#The lass gets as close as possible (given our grid)
print(lass_cv.coef_)
print(lass_cv.intercept_)
print(lass_cv.alpha_)
#iteratively fit model and create a graph
intercepts = []
lambda_values = .1 * np.array(range(1, 5))

print(lambda_values)

for lamb in lambda_values:
    l = Lasso(alpha=lamb)
    l.fit(X, y)
    intercepts.append(lasso.intercept_)
print(intercepts)

#graph result
plt.plot(lambda_values, intercepts)
