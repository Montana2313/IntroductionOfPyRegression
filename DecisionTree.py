#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 18:41:13 2020

@author: mac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('maaslar.csv')


x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X_data = x.values
Y_data = y.values

from sklearn.tree import DecisionTreeRegressor
dct = DecisionTreeRegressor(random_state = 0)
dct.fit(X_data,Y_data)

plt.scatter(X_data,Y_data,color="red")
plt.plot(X_data,dct.predict(X_data),color="blue")
plt.show()
'''
Veriler direk aslında sıralandı o yüzden böyle göstermektedir.
'''
other = pd.read_csv('veriler.csv')

boy_kilo = other.iloc[:,1:3]

yas = other.iloc[:,3:4]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(boy_kilo,yas,test_size=0.33,random_state=1)

dct2 = DecisionTreeRegressor(random_state = 0)
dct2.fit(x_train,y_train)

result = dct2.predict(x_test)


from sklearn.metrics import r2_score

print("R2 Score ------")
print(r2_score(Y_data,dct.predict(X_data)))
print(r2_score(y_test,result))










