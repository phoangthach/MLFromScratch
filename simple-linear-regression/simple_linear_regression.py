# ref: https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/
# Simple Linear Regression can be: y = b0 + b1*X

from random import seed, randrange
import numpy as np
from math import sqrt


# Calculate mean value
def mean(values):
    return sum(values) / len(values)


# Calculate variance
def variance(values, mean):
    # values as vector
    return sum((values - mean)**2)


# Calculate covariance
def covariance(x, mean_x, y, mean_y):
    return sum((x - mean_x)*(y - mean_y))


# Calculate coefficients
def coefficients(dataset):
    x = dataset[:, 0]
    y = dataset[:, -1]
    mean_x, mean_y = mean(x), mean(y)
    b1 = covariance(x, mean_x, y, mean_y) / variance(x, mean_x)
    b0 = mean_y - b1 * mean(x)
    return b0, b1     


# Define prediction
def simple_linear_regression(train, test):
    b0, b1 = coefficients(train)
    predictions = b0 + b1 * test
    return predictions


# Calculate root mean squared error
def rmse_metric(actual, predicted):
    sum_error = sum((predicted - actual)**2)
    mean_error = sum_error / len(actual)
    return sqrt(mean_error)


# Split dataset into train and test data
def train_test_split(dataset, split):
    train = np.empty((0, 2), float)
    train_size = split * len(dataset)
    
    dataset_copy = dataset.copy()
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train = np.vstack((train, dataset_copy[index]))
        np.delete(dataset_copy, index, 0)

    return train, dataset_copy


def evaluate_algorithm(dataset, algorithm, split):
    # make test data from dataset
    train, test = train_test_split(dataset, split)
    
    test_set = test.copy()
    test_set = test_set[:, 0]
    predicted = algorithm(train, test_set)
    rmse = rmse_metric(test[:, 1], predicted)
    return rmse


def load_csv(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data


seed(1)
filename = 'data.csv'
split = 0.6
dataset = load_csv(filename)
rmse = evaluate_algorithm(dataset, simple_linear_regression, split)
print('RMSE: ', rmse)

