#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:54:56 2020

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

##
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_data,Y_data)

plt.scatter(X_data,Y_data,color="red")
plt.plot(x,lin_reg.predict(X_data),color="blue")
plt.show()
##
# Polinomal artışa sahip veriler için 
# derecesine göre verilen verilerin tahmini daha doğru olacaktır
# x'2 düzleminde artan birşey için linear çizmek yanlış oalcaktır


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
x_poly = poly_reg.fit_transform(X_data)
print(x_poly)
lin_regression = LinearRegression()
lin_regression.fit(x_poly,y) # arttırılmılş boyutları ile öğrettik
plt.scatter(X_data,Y_data,color="red")
plt.plot(X_data,lin_regression.predict(x_poly),color="blue")
plt.show()


# Aslında amaç gelen değerli diğer uzaydaki değerleri ile eğitip
# tahminleri geliştiriyoruz
# x'2 x'3 gibi boyutlarını da öğretirsek daha iyi sonuçlar elde ederiz
# fakat bu veri zaten polynominal artışda yani bu her zaman için geçerli 
# olmayacaktır.

