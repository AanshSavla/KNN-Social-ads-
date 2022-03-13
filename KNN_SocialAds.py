# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 20:36:05 2020

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

data = pd.read_csv("D:\AanshFolder\datasets\social_nw.csv")
#print(data.head())

x = data.iloc[:,[2,3]]
y = data.iloc[:,4]

#print(x)
#print(y)

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state=0)

sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)
#print(X_train)
# p=1:Manhattan dustance,p=2:Euclidean distance
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,Y_train)
pred_y = classifier.predict(X_test)
accuracy = classifier.score(X_test,Y_test)
print("Accuracy:",accuracy)

cm = confusion_matrix(Y_test,pred_y)
print("Confusion Matrix:",cm)


x_set,y_set = X_test,Y_test

X1,X2 = np.meshgrid(np.arange(x_set[:,0].min()-1,x_set[:,0].max()+1,step=0.01),
                    np.arange(x_set[:,1].min()-1,x_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(['red','green']))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0],x_set[y_set == j,1],c=ListedColormap(('red','green'))(i),label=j)
    
plt.title("Logistic Regression")
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



