#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:22:26 2020

@author: mac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder

veriler = pd.read_csv('odev_tenis.csv')
outlook = veriler.iloc[:,:1].values
values = veriler.iloc[:,1:3]

veriler2 = veriler.iloc[:,3:]
veriler2 = veriler2.apply(LabelEncoder().fit_transform)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
outlook[:,0] = le.fit_transform(outlook[:,0])
print(outlook)

from sklearn.preprocessing import OneHotEncoder
onehotEncoder = OneHotEncoder(categorical_features='all')
outlook = onehotEncoder.fit_transform(outlook).toarray()

outlookDataFrame = pd.DataFrame(data=outlook,index=range(14),columns = ["overcast","rainy","sunny"])


mergeDataFrame = pd.concat([outlookDataFrame,values,veriler2],axis=1)
bagimsizDegiskenler = mergeDataFrame.iloc[:,:6]
bagimliDegisken = veriler2.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(bagimsizDegiskenler,bagimliDegisken,test_size=0.33,random_state=1)

from sklearn.linear_model import LinearRegression
regression= LinearRegression()
regression.fit(x_train,y_train)
y_pred = regression.predict(x_test)


import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values= mergeDataFrame.iloc[:,:-1], axis=1)

X_l = mergeDataFrame.iloc[:,[0,1,2,5]].values

model = sm.OLS(endog = mergeDataFrame.iloc[:,-1:],exog = X_l).fit()

print(model.summary())







