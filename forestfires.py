# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 13:10:48 2020

@author: Harsh
"""
import pandas as pd 
import numpy as np 
import seaborn as sns


forestfires = pd.read_csv("D:\STUDY\Excelr Assignment\Assignment 18 - Support Vector Machines\Forestfires.csv")
data =forestfires.head()
data
forestfires.describe()
forestfires.columns

sns.boxplot(x="size_category",y="FFMC",data=forestfires,palette = "hls")
sns.boxplot(x="FFMC",y="size_category",data=forestfires,palette = "hls")
#sns.pairplot(data=forestfires)


##Dropping the month and day columns
forestfires.drop(["month","day"],axis=1,inplace =True)

##Normalising the data as there is scale difference
predictors = forestfires.iloc[:,0:28]
target = forestfires.iloc[:,28]


def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return (x)

fires = norm_func(predictors)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.25, stratify = target)

model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)
print(pred_test_linear ) 
np.mean(pred_test_linear==y_test) # Accuracy = 100%
print(np.mean(pred_test_linear==y_test) )

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)

np.mean(pred_test_poly==y_test) #Accuacy = 100%

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)

np.mean(pred_test_rbf==y_test) #Accuracy = 74.6%

#'sigmoid'
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(x_train,y_train)
pred_test_sig = model_rbf.predict(x_test)

np.mean(pred_test_sig==y_test) #Accuracy = 73%

# As we can see in the data Kernel polynomial model and Linear model giving 100% accuracy while other two models are not 
#providing desired output and probability of burning samll area is high while burning of large area have much less probability. 