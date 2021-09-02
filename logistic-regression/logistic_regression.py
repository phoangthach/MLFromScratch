# ref1: https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/
# ref2: https://scikit-learn.org/stable/modules/cross_validation.html

import numpy as np
from random import randrange, seed


# Make a prediction with coefficients
def predict(rows, coefficients):
    input_data = rows.copy()
    yhat = np.dot(input_data, coefficients)

    return 1 / (1 + np.exp(-yhat))


# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(dataset, l_rate, n_epoch):
    data_train = np.array(dataset.copy())
    Y = data_train[:, -1]
    X = data_train[:, :-1]
    # adding bias
    X = np.c_[np.ones(len(X)), X]
    # init coef
    n_input_features = np.array(X).shape[1]
    coef = np.zeros(n_input_features)

    for epoch in range(n_epoch):
        yhat = predict(X, coef)
        error = Y - yhat
        coef += l_rate * np.dot(X.T, (error * yhat * (1 - yhat)))

    return coef


def load_csv(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data


# Normalize dataset
def normalize_dataset(dataset):
    data = dataset.copy()
    return (data - np.min(data, 0)) / (np.max(data, 0)-np.min(data, 0))


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_copy = dataset.copy()
    dataset_split = list()
    fold_size = int(dataset.shape[0]/n_folds)

    for i in range(n_folds):
        fold = np.empty((0, dataset.shape[1]), float)
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold = np.vstack((fold, dataset_copy[index]))
            dataset_copy = np.delete(dataset_copy, index, 0)

        dataset_split.append(fold)

    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = np.equal(actual, predicted).astype(int)
    return sum(correct) / len(actual) * 100


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = folds.copy()
        train_set = [x for x in train_set if not (x==fold).all()]
        train_set = np.vstack(train_set)
        test_set = fold.copy()

        actual = test_set[:, -1].copy()
        predicted = algorithm(train_set, test_set, *args)
        accuracy = accuracy_metric(actual, predicted)

        scores.append(accuracy)

    return scores


# Linear Regression Algorithm with Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
    coef = coefficients_sgd(train, l_rate, n_epoch)

    # remove label (last column) + add bias
    test = test[:, :-1]
    test = np.c_[np.ones(len(test)), test]
    predictions = predict(test, coef)

    return np.round(predictions)


seed(1)
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
dataset = normalize_dataset(dataset)
n_folds = 5
l_rate = 0.1
n_epoch = 1000

scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
print('Scores: ', scores)
print('Mean Accuracy: ', sum(scores)/len(scores))
