# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
'''
 Support Vector Regression
 Aslında normalde bildiğimiz SVM mantığı ile aynı 
 bir margin çizerek o aralıktan geçecek en doğru doğruyu bulmaya yarar
 bu margin aralığında olan değerler wa + b = 1 diğer tarafı
 wa + b = -1 olarak alabilriz
 bu margin ile doğru arasından olanların dışında olan değerler ise 
 error olarak alınır
 
 Diğerlerinden farkı nedir ? 
 
 Diğeri her ne kadar bazı verilerde daha kullanışlı olsada 
 dağılımı linear olmayan veriler için verimliğiği düşmektedir
 
  SVR çok boyutlu veriler için RBF vb yaklaşımlar ile margin değerlerini çok iyi belirleyebilir
  çok boyutlu verilerden kastımız 
  wa + b fonksiyonu her zaman sadece input lar ile değil
  fonksiyonlar ile define edilebilimektedir.
  
  kernel function = Kernel fonksiyonu düşük boyuttaki veri noktanın
  başka boyutlarını çıkararak orda margin bulma işlemi
  Bir noktanın x. boyutta linear olarak ayrılamaz belki 
  fakat x + 1. boyutta linear olarak ayrılabilmektedir.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X_data = x.values
Y_data = y.values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_data,Y_data)

plt.scatter(X_data,Y_data,color="red")
plt.plot(x,lin_reg.predict(X_data),color="blue")
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
x_poly = poly_reg.fit_transform(X_data)
print(x_poly)
lin_regression = LinearRegression()
lin_regression.fit(x_poly,y) # arttırılmılş boyutları ile öğrettik
plt.scatter(X_data,Y_data,color="red")
plt.plot(X_data,lin_regression.predict(x_poly),color="blue")
plt.show()


from sklearn.preprocessing import StandardScaler
# normalizasyon olayı aslında 
standardScaler1 = StandardScaler()
standardScaler2 = StandardScaler()
X_scaled = standardScaler1.fit_transform(X_data)
Y_scaled = standardScaler2.fit_transform(Y_data)

from sklearn.svm import SVR

svrObject = SVR(kernel='rbf',degree = 3)
fitted = svrObject.fit(X_scaled,Y_scaled)

plt.scatter(X_scaled,Y_scaled,color="red")
plt.plot(X_scaled,svrObject.predict(X_scaled),color="blue")


from sklearn.metrics import r2_score

print("R2 Score ------")
print(r2_score(Y_scaled,svrObject.predict(X_scaled)))




















