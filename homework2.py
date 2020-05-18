#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:59:06 2020

@author: mac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv("maaslar_yeni.csv")
bagimsiz_degiskenler = veriler.iloc[:,2:5]

bagimli_degisken = veriler.iloc[:,-1:]
X_data = bagimsiz_degiskenler.values
Y_data = bagimli_degisken.values
# Decision Tree
from sklearn.tree import DecisionTreeRegressor
dctObject = DecisionTreeRegressor(random_state = 0)
dctObject.fit(X_data,Y_data)
result_dct = dctObject.predict(X_data)

import statsmodels.api as sm
from sklearn.metrics import r2_score
model = sm.OLS(dctObject.predict(X_data),X_data).fit()
print(model.summary())

print("R2 Score DCT ------")
print(r2_score(Y_data,dctObject.predict(X_data)))


print("-----------------------------------------")
from sklearn.ensemble import RandomForestRegressor

randomForestRegressorObject = RandomForestRegressor(n_estimators = 10 , random_state = 0)

randomForestRegressorObject.fit(X_data , Y_data)
result_randomforest = randomForestRegressorObject.predict(X_data)
modelRandom = sm.OLS(randomForestRegressorObject.predict(X_data),X_data).fit()
print(modelRandom.summary())
print("R2 Score RandomForest ------")
print(r2_score(Y_data,randomForestRegressorObject.predict(X_data)))

print("-----------------------------------------")

from sklearn.linear_model import LinearRegression
regression= LinearRegression()
regression.fit(X_data,Y_data)
result_multi = regression.predict(X_data)
modelRegression = sm.OLS(regression.predict(X_data),X_data).fit()
print(modelRegression.summary())
print("R2 Score regression ------")
print(r2_score(Y_data,regression.predict(X_data)))

print("-----------------------------------------")
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X_data)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y_data)

svrObject = SVR(kernel='rbf',degree = 6)
svrObject.fit(x_olcekli,y_olcekli)
result_SVR = regression.predict(x_olcekli)
modelSVR = sm.OLS(regression.predict(x_olcekli),x_olcekli).fit()
print(modelSVR.summary())
print("R2 Score result_SVR ------")
print(r2_score(y_olcekli,svrObject.predict(x_olcekli)))
































