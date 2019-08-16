import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np  #general library, will come in handy later
import pandas as pd  #another nice library for storing matrices, it rely's on numpy
import matplotlib.pyplot as plt  #this library is for graphing things
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso  #These libraries have the necessary models
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
#load data
raw_df = pd.read_csv('credit.csv')
output = pd.DataFrame(raw_df['Balance'])
output = (output - output.mean()) / output.std()
y = output.values
df = raw_df.drop(['Unnamed: 0', 'Balance', 'Ethnicity', 'Gender', 'Student', 'Married'], axis=1)
df = (df-df.mean())/df.std()

m = {'Male' : 1, 'Female' : 0}
df['Gender'] = raw_df['Gender'].map(m)
m = {'Yes' : 1, 'No' : 0}
df['Student'] = raw_df['Student'].map(m)
m = {'Yes' : 1, 'No' : 0}
df['Married'] = raw_df['Married'].map(m)
m = {'Asian' : 1, 'Caucasian' : 0, 'African American' : 0}
df['Asian'] = raw_df['Ethnicity'].map(m)
m = {'African American' : 1, 'Caucasian' : 0, 'Asian' : 0}
df['African American'] = raw_df['Ethnicity'].map(m)

#get the variable names in a list
column_names = df.columns  #select the columns we want
print(column_names)
df_extended = df.copy()  #make a copy to edit
count = 0
for i in range(len(column_names)-2):
  column = str(column_names[i])
  j = 0
  if count > 5:
    j=1
  for interact in range(i+j, len(column_names)):
    interact = str(column_names[interact])
    interaction_name = column+'*'+interact
    df_extended[interaction_name] = df_extended[column] * df_extended[interact]
  count = count + 1
print(df_extended.describe())
X = df_extended



lasso = Lasso(alpha=.5)  #note that alpha corresponds to lambda
lasso.fit(X, y)
print(lasso.coef_)
print(lasso.intercept_)

#prepare fitted data to compare using MSE function

fitted_y = lasso.predict(X)
print(fitted_y.mean())

MSE = mean_squared_error(y, fitted_y)

print(MSE)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

alphas = np.logspace(-4, -1, 10)
scores = np.empty_like(alphas)
for i,a in enumerate(alphas):
    lasso = Lasso()
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    scores[i] = lasso.score(X_test, y_test)
    print('Lambda: ',a,' Coefs: ', lasso.coef_)
    
lassocv = LassoCV(cv=5)
lassocv.fit(X, y)
lassocv_score = lassocv.score(X, y)
lassocv_alpha = lassocv.alpha_
print('CV', lassocv.coef_)

plt.plot(alphas, scores, '-ko')
plt.axhline(lassocv_score, color='b', ls='--')
plt.axvline(lassocv_alpha, color='b', ls='--')
plt.xlabel(r'$\lambda$')
plt.ylabel('Score')
plt.xscale('log')
plt.savefig('my_graph.png')
plt.clf()   # Clear figure

for i in range(5):
    #compute start/end of fold
    start_index = int((80)*i)
    end_index = int((80)*(i + 1))

    #partition data
    X_test = X[start_index:end_index]
    y_test = y[start_index:end_index]

    X_train = np.concatenate((X[0:start_index], X[end_index:]))
    y_train = np.concatenate((y[0:start_index], y[end_index:]))

    #estimate model
    l = Lasso(alpha=.5)
    l.fit(X_train, y_train)
    fitted_y = lasso.predict(X_test)
    MSE = mean_squared_error(y_test, fitted_y)
    print('Fold %s, Coefficients: %s, Intercept: %s, MSE: %s \n' % (i+1, l.coef_, l.intercept_, MSE))


#from here you can figure out CV_n

#Specify a grid of lambda values to optimize over.
lambda_values = np.logspace(-4, -1, 10)

lass_cv = LassoCV(cv=5, alphas=lambda_values)
lass_cv = lass_cv.fit(X, y)

#The lasso gets as close as possible (given our grid)
print('lass_cv.coef_ : ', lass_cv.coef_)
print('lass_cv.intercept_ : ',lass_cv.intercept_)
print('lass_cv.alpha_ : ' ,lass_cv.alpha_)
#iteratively fit model and create a graph
intercepts = []

print(lambda_values)

for lamb in lambda_values:
    l = Lasso(alpha=lamb)
    l.fit(X, y)
    intercepts.append(lasso.intercept_)
print(intercepts)

#graph result
plt.plot(lambda_values, intercepts)
plt.savefig('Class_graph.png')