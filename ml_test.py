from logistic_regression import LogisticRegression
from linear_regression import LinearRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer,load_boston
from sklearn.preprocessing import normalize

# ********************************************************************************************************
# def accuracy(y_true,y_pred):
#     acc = np.sum(y_true==y_pred)/len(y_true)
#     return acc

# data = load_breast_cancer()
# X,y = data.data,data.target

# X = normalize(X)

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# model = LogisticRegression(lr=0.1,n_iter=50000)
# model.fit(X_train,y_train)
# pred = model.predict(X_test)

# acc = accuracy(y_test, pred)
# print("Accuracy of Logistic Regression on Breast Cancer Data is : ",acc)

# ********************************************************************************************************

def MSE(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)    
    
boston = load_boston()
data = boston.data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
standardised_data = sc.fit_transform(data)
bos = pd.DataFrame(standardised_data)
bos['PRICE'] = boston.target

Y = bos['PRICE']
X = bos.drop('PRICE', axis = 1)

# Adding a new feature to the data which will contain only ones for ease in computation 
additional_feature = np.ones(boston.data.shape[0])

# Matrix having new additional feature X0 which will be multiplied with W0 for the ease of computation
feature_data = np.vstack((additional_feature,standardised_data.T)).T

X_train, X_test, y_train, y_test = train_test_split(feature_data, Y, test_size = 0.3, random_state = 0)

model = LinearRegression(lr=0.01,n_iter=10000)
model.fit(X_train,y_train)
pred = model.predict(X_test)

mse = MSE(y_test, pred)
print("MSE of Linear Regression on Boston Data is : ",mse)
