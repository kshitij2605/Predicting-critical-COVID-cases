import numpy as np
import statistics as stats
import sys
import csv


def import_data():
    #X = np.genfromtxt("train_X_pr.csv", delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt("train_Y_pr.csv", delimiter=',', dtype=np.float64)
    return Y


def replace_null_values(X):
    categorial_features = [0, 1, 3, 4, 6]
    n = len(X[0])


    for i in range(n):
        column = X[:, i]
        nan_indices = np.where(np.isnan(column))


        if i in categorial_features:
            mode = stats.mode( column[np.where(np.isnan(column)==0)] )
            column[nan_indices] = int(mode)




        else:
            mean = np.nanmean(column)
            column[nan_indices] = mean
    #print(X[categorial_features])
    return X



def remove_rows_with_null_values(X):
    X_with_no_nan = []
    for row in X:
        row_contains_nan = True in np.isnan(row)
        if not row_contains_nan:
            X_with_no_nan.append(row)
    return X_with_no_nan


def replace_null_values_with_zeros(X):
    array_with_true_where_nan = np.isnan(X)
    X[array_with_true_where_nan] = 0
    return X


def replace_null_values_with_mean(X):
    #Obtain mean of columns
    col_mean = np.nanmean(X, axis=0)

    #Find indicies that we need to replace
    inds = np.where(np.isnan(X))

    #Place column means in the indices. Align the arrays using take
    X[inds] = np.take(col_mean, inds[1])
    return X


def replace_zero_values_with_mean(X, col):
    #Obtain mean of columns
    col_mean = np.mean(X[:,col])

    #Find indicies that we need to replace
    inds = np.where(X[:,col] == 0)

    #Place column means in the indices. Align the arrays using take
    X[inds, col] = col_mean
    return X


def standardize(X, column_indices):
    for column_index in column_indices:
        column = X[:,column_index]
        mean = np.mean(column, axis=0)
        std = np.std(column, axis=0)
        X[:,column_index] = (column - mean) /std
    return X


def min_max_normalize(X, column_indices):
    for column_index in column_indices:
        column = X[:,column_index]
        min = np.min(column, axis=0)
        max = np.max(column, axis=0)
        difference = max- min
        X[:,column_index] = (column - min) /difference
    return X


def mean_normalize(X, column_indices):
    for column_index in column_indices:
        column = X[:,column_index]
        min = np.min(column, axis=0)
        max = np.max(column, axis=0)
        avg = np.average(column, axis=0)
        difference = max- min
        X[:,column_index] = (column - avg) /difference
    return X


def convert_to_numerical_labels(X):
    uniques_values = list(set(X))
    uniques_values.sort()

    # preparing map of unique values to numerical labels
    unique_values_labels_map = {}
    for i in range(0, len(uniques_values)):
        unique_values_labels_map[uniques_values[i]] = i

    # converting data to numerical labels
    X_numerical = []
    for i in range(0, len(X)):
        numerical_value = unique_values_labels_map[X[i]]
        X_numerical.append(numerical_value)

    return X_numerical


def apply_one_hot_encoding(X):
    #Y = np.sort(np.unique(X))
    unique_values = list(set(X))
    unique_values.sort()
    one_hot_encoding_map = {}
    counter = 0
    for x in unique_values:
        one_hot_encoding_map[x] = [0 for i in range(len(unique_values))]
        one_hot_encoding_map[x][counter] = 1
        counter += 1

    one_hot_encoded_X = []
    for x in X:
        one_hot_encoded_X.append(one_hot_encoding_map[x])

    one_hot_encoded_X = np.array(one_hot_encoded_X, dtype=int)
    return one_hot_encoded_X


def convert_given_cols_to_one_hot(X, column_indices):
    one_hot_encoded_X = np.zeros([len(X), 1])

    start_index = 0
    # acts column pointer in X

    for curr_index in column_indices:
        # adding the columns present before curr_index in X (and not present in one_hot_encoded_X), to one_hot_encoded_X
        one_hot_encoded_X = np.append(one_hot_encoded_X, X[:, start_index:curr_index], axis=1)

        # applying one hot encoding for current column
        one_hot_encoded_column = apply_one_hot_encoding(X[:, curr_index])

        # appending the obtained one hot encoded array to one_hot_encoded_X
        one_hot_encoded_X = np.append(one_hot_encoded_X, one_hot_encoded_column, axis=1)

        # moving the column pointer of X to next current_index
        start_index = curr_index + 1

    # adding any remaining columns to one_hot_encoded_X
    one_hot_encoded_X = np.append(one_hot_encoded_X, X[:, start_index:], axis=1)
    one_hot_encoded_X = one_hot_encoded_X[:, 1:]
    return one_hot_encoded_X


def get_correlation_matrix(X, Y):
    num_vars = len(X[0]) + 1
    m = len(X)
    correlation_matrix = np.zeros((num_vars,num_vars))
    Y = Y.reshape(len(X),1)
    for i in range(1):
        for j in range(i, num_vars):
            mean_Y = np.mean(Y)
            mean_j = np.mean(X[:, j])
            std_dev_Y = np.std(Y)
            std_dev_j = np.std(X[:, j])
            numerator = np.sum((Y - mean_Y)*(X[:, j] - mean_j))
            denominator = m*(std_dev_Y)*(std_dev_j)
            corr_i_j = numerator/denominator
            correlation_matrix[i][j] = corr_i_j
            correlation_matrix[j][i] = corr_i_j

    for i in range(1,num_vars):
        for j in range(i,num_vars):
            mean_i = np.mean(X[:,i])
            mean_j = np.mean(X[:,j])
            std_dev_i = np.std(X[:,i])
            std_dev_j = np.std(X[:,j])
            numerator = np.sum((X[:,i]-mean_i)*(X[:,j]-mean_j))
            denominator = (m)*(std_dev_i)*(std_dev_j)
            corr_i_j = numerator/denominator
            correlation_matrix[i][j] = corr_i_j
            correlation_matrix[j][i] = corr_i_j
    return correlation_matrix


def select_features(corr_mat, T1, T2):
    n=len(corr_mat)
    filtered_features = []
    for i in range(1,n):
        if (abs(corr_mat[i][0]) > T1):
            filtered_features.append(i)
    m = len(filtered_features)
    removed_features = []
    selected_features = list(filtered_features)
    for i in range(0,m):
        for j in range(i+1,m):
            f1 = filtered_features[i]
            f2 = filtered_features[j]
            if (f1 not in removed_features and f2 not in removed_features):
                if (abs(corr_mat[f1][f2]) > T2):
                    selected_features.remove(f2)
                    removed_features.append(f2)

    return selected_features


def preprocess(X):
    #Y = import_data()
    # X = replace_null_values_with_zeros(X)
    # X = replace_zero_values_with_mean(X, 5)
    X = replace_null_values(X)
    X = standardize(X, [2])
    X = min_max_normalize(X, [1,4,5,6])
    X = convert_given_cols_to_one_hot(X, (0,3))
    #corr_mat = get_correlation_matrix(X, Y)
    return X


