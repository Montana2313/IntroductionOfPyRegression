# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Multiple Regression

# birden çok bağımsız ve bağımlı değişken olması gibi durumlarda olabilir
# a = B0 + B1W1 + B2W2 + B3W3 + ERROR
# hesaplanacak bağlı değişken birden çok bağımsız değişken ile bağlı olması
# Dummy variable : Kukla değişken olarak çevrilen bu değişkenler birbiri ile aynı
# şeyi ifade eden ve birbirine bağlı olarak değişen değişken tiplerine denir
# örnek olarak : Cinsiyet verilerini OneHotEncoder ile E K 1 0 olarak çevirdikten sonra 
# bunları eğitirken göz ardı edilmediri 
# Önerilen : Eğer bir tarafını alıyorsan eğitim dummylerden diğer tarafları almamalıyız
# örnek olarak kadınları tahmin ederek ndiğer cinsiyet ve erkek kolonunu almamız gereklidir
# aynı şeyi ifade etmektedir. Eğer E K birlikte alıyorsak bunlar birbirine bağlı olduğu için alınmamalıdır.
# Fakat eğer birbirlerinden çıkaralamayacak değerler ise alınmalıdır.
# Örnek olarak Erkek 1 ise erkek 0 kadın olabileceği çıkarılırken 
# şehir tahmini olarak düşünürsek bunu tahmin edemeyiz (Polynominal bir örnek)
# -------
# null hipotez : Bir örneğin kutuda 100 tane kurabiye var her kutuda 100 tane vardır
# yani ilk olarak alacağımız hipotez temmel olarak alacağımız hipotez

# h1 alternative hipotez : 1 kutuda 95 tane olması durumu null hipotezi çürütemez durumu
# alternative durumu aslında 

# P-value ise ne kadar örnekte çürütebilir ne kadarı doğru kabul edebiliriz olarak düşünebilirz
# P değeri değiştikçe 
# p küçüldükçe temel varsayımın hatalı olma ihtimali artar
# p yükseldikçe alternatif olarak verilen varsayımın hata olması artar
# p değerini bir threshold olarak düşünürsek 0.005 den büyük ise H1 küçükse H0 hatalı diyebilriz.
# -------
# Peki Multiple'da hangi kolonları seçersek daha doğru olur ? 
# Yaklaşımlar 

# Bütün değişkenleri dahil etmek
# Sistemin hareketini öğrenmek veya değerlerin etkisini görmek için yapılan
# veya daha önce yapılmış modeli karşılaştırmak için yapılabilmektedir.,

# Geriye Doğru Seçilim
# İlk olarak sisteme tüm değişkenler dahil edilerek model oluşturulur
# P value hesaplanarak sistemin P valuesu netlik değerinden büyük ise 
# en yüksek P value değeri olan değişken sistemden çıkarılır ve tekrar bu model oluşuturulur
# tekrar işlemler dahil olur taa ki netlik değerinden büyük olmayana kadar

# İleri doğru seçilim
#ASlında Geriye doğru ile aynı fakat , sistem p valuse netlikden az olunca 
# sistemde en düşük p value sisteme eklenerek devam eder
# tek değişkenden başlayarak şartı sağlananana kadar ekleme devam eder

# Çift Yönlü seçilim
# GEne bir netlik değeri belirlenir
# Geri ve ileri birleşmiş hali diyebilriz.
# p value düşük olanlar eklenerek 
# p value yüksek olanlar çıkarılarak ilerlenir harman yöntem


# Skor Karşılaştırması
# Aslında kendi kriterini kendinin yazması ve modeller inşa edilmesi
# Kriter sounucu en iyi sağlananı bulana kadar modelleri denenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('veriler.csv')


yasVs = veriler.iloc[:,1:4].values
print(yasVs)

ülkeler = veriler.iloc[:,0:1].values
print(ülkeler)

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


cinsiyetVerileri = veriler.iloc[:,-1:].values
print(cinsiyetVerileri)
lb = LabelEncoder()
cinsiyetVerileri[:,0] = lb.fit_transform(cinsiyetVerileri[:,0])
print(cinsiyetVerileri)

yasVsDataFrame = pd.DataFrame(data=yasVs,index=range(22),columns = ["boy","kilo","yas"])

ülkeDataFrame = pd.DataFrame(data=ülkeler,index=range(22),columns = ["tr","fr","usa"])
cinsiyetDataFrame = pd.DataFrame(data=cinsiyetVerileri,index=range(22),columns = ["cinsiyet"])

yasvsAndÜlkeDataFrame = pd.concat([yasVsDataFrame,ülkeDataFrame],axis=1)


mergeDataFrame = pd.concat([yasVsDataFrame,ülkeDataFrame,cinsiyetDataFrame],axis=1)


## VERİLERİ TEST VE TRAİN OLARAK BÖLMEK
print("-------- TEST VE TRAIN ---------")

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(yasvsAndÜlkeDataFrame,cinsiyetDataFrame,test_size=0.33,random_state=1)
## x train bizim verimiz öğreneceği x test test kısmı 
## y train ve y test bizim verimizin tahmin edeceği kısım

from sklearn.linear_model import LinearRegression
regression= LinearRegression()
regression.fit(x_train,y_train)
y_pred = regression.predict(x_test)

## boyu alalım
boy = mergeDataFrame.iloc[:,:1].values
veriGeriKalan = mergeDataFrame.iloc[:,1:]
 

x_train,x_test,y_train,y_test = train_test_split(veriGeriKalan,boy,test_size=0.33,random_state=1)
regression_2= LinearRegression()
regression_2.fit(x_train,y_train)
y_pred_2 = regression_2.predict(x_test)

#modellerin başarısı için
import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int), values= veriGeriKalan, axis=1)

X_l = veriGeriKalan.iloc[:,[0,1,2,3,4,5]].values

model = sm.OLS(endog = boy,exog = X_l).fit()

print(model.summary())

X_data = x_test.values
Y_data = y_test.values


from sklearn.metrics import r2_score

print("R2 Score ------")
print(r2_score(Y_data,regression_2.predict(X_data)))


















 
