# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:36:35 2020

@author: Aryamaan
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data=pd.read_csv("final2.csv")
data.dropna(inplace=True)
X=data.iloc[:,:-1].values
y=data.iloc[:,3].values


"""data2=pd.read_csv("ACCIDENTAL.csv")
data2.dropna(inplace=True)
X2=data2.iloc[:,:-1].values
y2=data2.iloc[:,3].values


y=np.append(y,y2)
X=X.reshape(-1,1)
"""
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
"""np.any(np.isnan(X_test))
np.all(np.isfinite(X_test))
np.any(np.isnan(X_train))
np.all(np.isfinite(X_train))
np.any(np.isnan(y_test))
np.all(np.isfinite(y_test))
np.any(np.isnan(y_train))
np.all(np.isfinite(y_train))"""


from sklearn import preprocessing
#lab_enc = preprocessing.LabelEncoder()
#training_scores_encoded = lab_enc.fit_transform(y_train)


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



print(classifier.predict([[2,-9.8,2.2]]))


#plt.scatter(x,y,color="red")
