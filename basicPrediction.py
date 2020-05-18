#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:23:52 2020

@author: mac
"""
# Prediction : Hem gelecek fakat aynı zaman geçmişte veya aradaki eksik
# verileride tahmin edebilmektedir
# Forecasting : Geleceğe yönelik benim verilerimden daha çok daha yüksek 
# değerler tahmin etmesi örnek olarak borsanını yıl sonu kapanış değeri vs.

# Doğrusal Regresyon
# Aslında temelde bizim bildiğimiz doğru çizmedeki y = ax + b formülü gelmektedir.
# y burada bağımlı değişken çünkü a x ve b değelerine göre bağımlı
# x burada bağımsız değişken olarak düşünebiliriz
# x i input değerler olarak düşünürsak bizim matematiksel formülümüze göre 
# x in geldiği noktadaki bizim doğrumuz ile kesişti noktanın y karşılığını veriyor olacaktır.

# bizim veri kümemeiz x ve y koordinatları üzerinde yayıldığını düşünebilriz bu çıkarımda 
# örnek aklıma gelen bir proje olabilir (yazdığım satır kodu sayısından bir projenin süreç tahmin)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('satislar.csv')

print(veriler)

aylar = veriler.iloc[:,:1] # bağımsız değişkenler
satislar = veriler["Satislar"] # bağımlı değişkenler
# .values dersen array olarak alır böyle olursa DF

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=1)
'''
from sklearn.preprocessing import StandardScaler
# normalizasyon olayı aslında 
standardScaler = StandardScaler()
X_Train = standardScaler.fit_transform(x_train)
X_Test = standardScaler.fit_transform(x_test)
Y_Train = standardScaler.fit_transform(y_train)
Y_Test = standardScaler.fit_transform(y_test)
'''
from sklearn.linear_model import LinearRegression

linearRegressionObject = LinearRegression()

linearRegressionObject.fit(x_train,y_train)
## indexler arası aya göre aynı satışı bularak öğrenmeye çalışacak
## bu trainleri alarak bir model oluştur.
## command i ile fonksiyon hakkında bilgi penceresi

result = linearRegressionObject.predict(x_test)

# Verilerin indexleri sıralı olmadığı için normal bir plot işleminde karmaşık olacaktır.
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,result)

X_data = x_test.values
Y_data = y_test.values


from sklearn.metrics import r2_score

print("R2 Score ------")
print(r2_score(Y_data,linearRegressionObject.predict(X_data)))

























