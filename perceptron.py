import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

inputs = np.array([[0,0,1],
                   [1,1,1],
                   [1,0,1],
                   [0,1,1]])

original_outputs = np.array([[0,1,1,0]]).T

# np.random.seed(1)

weights = 2*np.random.random((3,1)) - 1

print("Random starting weights are :\n{}".format(weights))

for i in range(10000):
    input_layer = inputs
    outputs = sigmoid(np.dot(input_layer,weights))
    
    error = original_outputs - outputs
    adjustments = error*sigmoid_derivative(outputs)
    
    weights += np.dot(input_layer.T,adjustments)
    
print("Results :\n{}".format(outputs))

