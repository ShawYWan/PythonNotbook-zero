# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read data
dataset = pd.read_csv('50_Startups.csv')
dataset = dataset.fillna(round(dataset.mean(),0))
print(dataset)

# X and Y
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
print(X,Y)

# transform the datetype
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
print(X)

#split the dataset to train and test part
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

# train the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)

#plot
plt.plot(range(len(Y_pred)),Y_pred,'red', linewidth=2.5,label="predict data")
plt.plot(range(len(Y_test)),Y_test,'green',label="test data")
plt.legend(loc=2)
plt.show()#show the predict and test lines

"""
I am not familiar with muti-regressor.
There are a lot code I don't understand.
"""
