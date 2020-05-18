 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:19:36 2020

@author: mac
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('veriler.csv')


boy = veriler[["boy"]]

print(boy)

boy_kilo = veriler[['boy','kilo']]

print(boy_kilo)

# Kategorik Veriler

ülkeler = veriler.iloc[:,0:1].values
## ülkeler
print(ülkeler)
print("LabelEncoder ------")
# stringleri value ya çeiviriyor
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ülkeler[:,0] = le.fit_transform(ülkeler[:,0])
print(ülkeler)
##
print("OneHot Encoder ------")
# bu valueların düzenlenemesi
from sklearn.preprocessing import OneHotEncoder
onehotEncoder = OneHotEncoder(categorical_features='all')
ülkeler = onehotEncoder.fit_transform(ülkeler).toarray()
print(ülkeler)
# kategorik verileri daha rahat okuma yapabilmezi sağlayan fonksiyon

## DATAFrame yapma ve birleştirme işlemleri
## Neden yapıyoruz ? 
## buradaki değerleri bakacak olursan ülkerel ve yaş tablosunu ayrı ayrı işleml
## ler yaptık bunlar NAN ları çıkarma ortalama alma 
## veya kategorik sınıflandırma gibi 
## bunları bi araya getireceğiz ve indexeleme yapacağız şuan array halindeler 
print("------- DATAFRAME ÇEVİRME İŞLEMLERİ ----------")
ülkelerDataFrame = pd.DataFrame(data=ülkeler,
                                index= range(22),
                                columns=["France","Turkey","USA"])

print(ülkelerDataFrame)

yasDataFrame = pd.DataFrame(data=yas,
                            index= range(22),
                            columns = ["boy","kilo","yas"])

print(yasDataFrame)

## böylece index verdik ve coloumns lar belli oldu

cinsiyetVerileri = veriler.iloc[:,-1].values
## -1 demek aslında son veriye erişmek oluyor
convert = LabelEncoder()
cinsiyetVerileri = convert.fit_transform(cinsiyetVerileri)## çevirdik

cinsiyetDataFrame = pd.DataFrame(data=cinsiyetVerileri,index=range(22),columns = ["Cinsiyet"])
print(cinsiyetDataFrame)

mergeDataFrame = pd.concat([ülkelerDataFrame,yasDataFrame,cinsiyetDataFrame],axis=1)
## axis 0 olursa alt alta birleştirme yapacak ve olmayanlar için NAN yapacaktı
## 1 yazarsa row olarak birleştirme yapar
print(mergeDataFrame)

trainDataFrame = pd.concat([ülkelerDataFrame,yasDataFrame],axis=1)

## VERİLERİ TEST VE TRAİN OLARAK BÖLMEK
print("-------- TEST VE TRAIN ---------")

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(trainDataFrame,cinsiyetDataFrame,test_size=0.33,random_state=1)
## x train bizim verimiz öğreneceği x test test kısmı 
## y train ve y test bizim verimizin tahmin edeceği kısım

##
print("--------- ÖZNİTELİK ÖLÇEKLEME ------------")

from sklearn.preprocessing import StandardScaler
# normalizasyon olayı aslında 
standardScaler = StandardScaler()
X_Train = standardScaler.fit_transform(x_train)
X_Test = standardScaler.fit_transform(x_test)













