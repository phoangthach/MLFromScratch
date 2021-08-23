# ref1: https://pub.towardsai.net/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbf
# ref2: https://pythonmachinelearning.pro/perceptrons-the-first-neural-networks/

import numpy as np

# Define input features
input_features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

print('input features: \n', input_features)

# Define target output
target_output = np.array([[0, 1, 1, 1]])

# Convert target output to vector
target_output = target_output.reshape(4, 1)

print('target output: \n', target_output)

# Bias
bias = 0.3

# Define weights
weights = np.array([[0.1], [0.2]])
weights = np.insert(weights, 0, bias, axis=0)

# weights can be initialized with 0 as below
# weights = np.zeros(3).reshape(3, 1)
print('weights: \n', weights)

# Learning Rate
lr = 0.03

# Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

b = np.ones(4)
input_features = np.c_[b, input_features]
print('input_features: \n', input_features)

# Define predict func
def predict(x):
    return (x > 0.5).astype(int)

# the number of epoch larger the accuracy higher
for epoch in range(10000): 
    inputs = input_features

    # feedforward input
    z = np.dot(inputs, weights)    

    # feedforward ouput
    a = sigmoid(z)

    # backpropogation
    # calculating error
    error = a - target_output

    # updating weights
    inputs = input_features.T
    weights -= lr * np.dot(inputs, error)
    
print('optimal weights: \n', weights)

# prediction
print('---predict---')
single_point = np.array([1, 1])
# add bias to single_point
single_point = np.insert(single_point, 0, 1)

result1 = np.dot(single_point, weights)
result2 = sigmoid(result1)
print('result1: ', result1)
print('sigmoid: ', result2)
# predict result should equal 1
print('predict: ', predict(result2))
