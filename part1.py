import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def read_data():
    df = pd.read_csv('https://raw.githubusercontent.com/MertBuyulu/linear-regression/main/abalone.data', header=None,
                      names=['sex', 'length', 'diam', 'height', 'w_weight', 'shk_weight', 'v_weight', 'shl_weight', 'rings'])
    return df

def preprocess(df):
    df['sex'].replace(['M', 'F', 'I'], [0, 1, 3], inplace=True)
    X = df.drop(['rings'], axis = 1)
    Y = df['rings']

    X_cols = X.columns
    s = StandardScaler()
    X = pd.DataFrame(s.fit(X).fit_transform(X), columns=X_cols)

    #X = pd.concat([pd.Series(1, index=X.index, name='bias'), df], axis=1)
    #print(X.head())

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
    #print(Y_test.to_markdown())

    return X_train, X_test, Y_train, Y_test

def gradient_descent(X, Y, learning_rate, iterations):
    costs = []
    weights = np.zeros(X.shape[1])
    bias = 0
    m = X.shape[0]

    for _ in range(iterations):
        y_pred = np.dot(X, weights) + bias
        cost = (1 / (2 * m)) * np.sum((y_pred - Y) ** 2)
        costs.append(cost)
        residual_weights = (1 / m) * (np.dot(X.T, y_pred - Y))
        residual_bias = (1 / m) * np.sum(y_pred - Y)

        weights -= learning_rate * residual_weights
        bias -= learning_rate * residual_bias

    return weights, bias, costs

def predict(X, Y, weights, bias):
    predictions = np.dot(X, weights) + bias
    predictions = np.rint(predictions)
    
    Y = pd.concat([pd.Series(predictions, index=Y.index, name='pred'), Y], axis=1)
    #print(Y.to_markdown())


if __name__ == '__main__':
    df = read_data()
    X_train, X_test, Y_train, Y_test = preprocess(df)
    weights, bias, costs = gradient_descent(X_train, Y_train, learning_rate=.003, iterations=10000)
    predict(X_test, Y_test, weights, bias)

    plt.figure()
    plt.scatter(x=list(range(10000)), y=costs)
    plt.show()