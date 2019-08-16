import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
 
df = pd.read_csv('heart.csv')
 
print(df.head())
print()
 
colors = {0:'blue', 1:'red'}
fig, axs = plt.subplots(2, 2, sharey=False)    #Sharey defines a shared y axis for the plots in each row. Similarly, sharex would share the x axis
df.plot(kind='scatter', x='Ca', y='Oldpeak', c= df['AHD_Yes'].apply(lambda x: colors[x]), ax=axs[0][0], figsize=(16, 8))
df.plot(kind='scatter', x='Oldpeak', y='MaxHR', c= df['AHD_Yes'].apply(lambda x: colors[x]), ax=axs[0][1])
df.plot(kind='scatter', x='Ca', y='MaxHR', c= df['AHD_Yes'].apply(lambda x: colors[x]), ax=axs[1][0])
fig.savefig('graph.png')
train_df = df.loc[df['Thal_normal'] == 1]
test_df = df.loc[df['Thal_normal'] == 0]
 
print(train_df.head())
print(test_df.head())
 
clf = svm.SVC(gamma=1)
svmfit = clf.fit(train_df[['MaxHR', 'Oldpeak', 'Ca']], train_df[['AHD_Yes']])
 
y_pred = svmfit.predict(test_df[['MaxHR', 'Oldpeak', 'Ca']])
 
print("Accuracy:",metrics.accuracy_score(test_df[['AHD_Yes']], y_pred))