import csv
import numpy as np
import concurrent.futures
from preprocess import preprocess


def import_data():
    X = np.genfromtxt("train_X_pr.csv", delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt("train_Y_pr.csv", delimiter=',', dtype=np.float64)
    return X, Y


def predict_target_values(test_X, weights):
    b = weights[0]
    W = weights.reshape(11, 1)[1:]
    A = sigmoid(np.dot(test_X, W) + b)
    A = A.T[0]
    pred_Y = np.where(A >= 0.4, 1, 0)
    return pred_Y


def calculate_accuracy(pred_Y, actual_Y):
    correct = 0
    for i in range(len(pred_Y)):
        if pred_Y[i] == actual_Y[i]:
            correct += 1

    return correct / len(pred_Y)


def calculate_precision(pred_Y, actual_Y):
    TP = 0
    for i in range(len(pred_Y)):
        if pred_Y[i] == 1 and actual_Y[i] == 1:
            TP += 1

    return TP / (pred_Y[pred_Y == 1]).size


def calculate_recall(pred_Y, actual_Y):
    TP = 0
    for i in range(len(pred_Y)):
        if pred_Y[i] == 1 and actual_Y[i] == 1:
            TP += 1

    return TP / (actual_Y[actual_Y == 1]).size


def f1_score(pred_Y, actual_Y):
    P = calculate_precision(pred_Y, actual_Y)
    R = calculate_recall(pred_Y, actual_Y)

    return 2 * P * R / (P + R)


def sigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    return s


def compute_gradients_using_regularization(X, Y, W, b, Lambda):
    m = len(X)
    A = np.dot(X, W) + b
    A = sigmoid(A)
    dW = 1/m * (np.dot((A-Y).T, X) + Lambda*(W.T))
    db = 1/m * np.sum(A-Y)
    dW = dW.T
    return dW, db


def compute_cost(X, Y, W, b, Lambda):
    M = len(X)
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    A[A == 1] = 0.99999
    A[A == 0] = 0.00001
    cost = (-1/M) * np.sum(np.multiply(Y,np.log(A)) + np.multiply((1-Y),np.log(1-A)))
    regularization_cost = (Lambda * np.sum(np.square(W))) / (2 * M)
    return cost + regularization_cost


def optimize_weights_using_gradient_descent(X, Y, W, b, learning_rate, limit):
    previous_iter_cost = 0
    iter_no = 0
    Lambda = 0.097
    while True:
        iter_no += 1
        dW, db = compute_gradients_using_regularization(X, Y, W, b, Lambda)
        W = W - learning_rate * dW
        b = b - learning_rate * db
        cost = compute_cost(X, Y, W, b, Lambda)
        if iter_no % 10000 == 0:
            print(iter_no, cost)

        if abs(previous_iter_cost - cost) < limit:
            print(iter_no, cost)
            break

        previous_iter_cost = cost
    return W, b


def train_model(X, Y, learning_rate, limit):
    Y = Y.reshape(len(X), 1)
    W = np.ones((X.shape[1],1))
    b = 1
    W, b = optimize_weights_using_gradient_descent(X, Y, W, b, learning_rate, limit)
    return np.vstack((np.array([[b]]), W.reshape(X.shape[1], 1)))


def save_model(weights, weights_file_name):
    with open(weights_file_name, 'w', newline='') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()


def split_data(X, Y, split_percent):
    m = len(X)
    spliting_row = int((100-split_percent)*m/100)
    train_X = X[:spliting_row]
    validation_X = X[spliting_row:]
    train_Y = Y[:spliting_row]
    validation_Y = Y[spliting_row:]

    return train_X, validation_X, train_Y, validation_Y


if __name__ == '__main__':
    X, Y = import_data()
    X = preprocess(X)
    learning_rate = 2
    limit = 0.000000000001

    train_X, validation_X, train_Y, validation_Y = split_data(X, Y, 27)

    weights = train_model(X, Y, learning_rate, limit)

    pred_Y = predict_target_values(validation_X, weights)

    print("accuracy "+str(calculate_accuracy(pred_Y, validation_Y)))
    print("precision "+str(calculate_precision(pred_Y, validation_Y)))
    print("recall "+ str(calculate_recall(pred_Y, validation_Y)))
    print("f1-score "+ str(f1_score(pred_Y, validation_Y)))

    save_model(weights, "WEIGHTS_FILE.csv")
