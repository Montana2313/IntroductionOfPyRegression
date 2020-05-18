# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

'''
Random Forest Algoritması 

Kollektif öğrenme nedir ? 
Kollektif öğrenme birden çok algoritmanın yararları kullanarak birlikte iyi bir sonuç üretmesi
Tahminler veya sınıflandırmalara içinde geçerlidir
Random forest buna bir örnektir.

Bu algoritma diğer arkadaşları gibi trainin ve test olarak bölünüyor fakat,

Train kendi içersinde birden çok dct oluşturuyor ve her biri verinin başka yerinden
dct çizimi yapıyor. Peki tahmin veya sınıflandırma nasıl karar veriliyor ? 

Burada bilmemiz gereken bir tanım olarak karşımıza Majority voted learning çıkmaktadır.

Majority Voted : Bir çok oluşuturulmuş dct üzerinden istenene tahmin yapılır ve 
sınıflandırma yapılıyorsa en çok kararı veren sonuç kabul edilir
tahmin yapılıyor ise ortalamalaarı kabul edilir.

Random forest dct'ye göre daha istediğimiz türde veri tahmini yapabilmektedir.
Birden çok dct üzerinden ortalam değeri aldığı için

Fakat dct diğer kutulama yaparak bölgeye giren hangi değer olursa olsun belirli değeri döndürür
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('maaslar.csv')


x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X_data = x.values
Y_data = y.values

from sklearn.ensemble import RandomForestRegressor

randomForestRegressorObject = RandomForestRegressor(n_estimators = 10 , random_state = 0)

randomForestRegressorObject.fit(X_data , Y_data)
result = randomForestRegressorObject.predict(X_data)
                                
plt.scatter(X_data , Y_data , color = "red")
plt.plot(X_data,randomForestRegressorObject.predict(X_data) , color = "green")
plt.show()


from sklearn.metrics import r2_score

print("R2 Score ------")
print(r2_score(Y_data,randomForestRegressorObject.predict(X_data)))