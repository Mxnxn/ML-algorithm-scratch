import numpy as np

class LogisticRegression:
    
    def __init__(self,lr=0.001,n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights =None
        self.bias =None
        
    def fit(self,X,y):
        n_samples,n_features = X.shape
        
        #weights init
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        #gradient descent
        for _ in range(self.n_iter):
            linear_model = np.dot(X,self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            #derivative of cost function
            dw = (1/n_samples)*np.dot(X.T,(y_pred - y))
            db = (1/n_samples)*np.sum(y_pred - y)
            
            #weights update
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
            
            
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def predict(self,X):
        linear_model = np.dot(X,self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        
        pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return pred_class
    
        