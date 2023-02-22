import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class LinearRegression:
    def __init__(self, learning_rate=0.003, iterations=10000, tolerance=0.000001):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = 0

    def preprocess_data(self):
        df = pd.read_csv('https://raw.githubusercontent.com/MertBuyulu/linear-regression/main/abalone.data', header=None,
                        names=['sex', 'length', 'diam', 'height', 'w_weight', 'shk_weight', 'v_weight', 'shl_weight', 'rings'])
    
        df['sex'].replace(['M', 'F', 'I'], [0, 1, 3], inplace=True)
        X = df.drop(['rings'], axis = 1)
        Y = df['rings']

        X_cols = X.columns
        s = StandardScaler()
        X = pd.DataFrame(s.fit(X).fit_transform(X), columns=X_cols)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

        return X_train, X_test, Y_train, Y_test

    def fit(self, X, Y):
        costs = []
        self.weights = np.zeros(X.shape[1])
        m = X.shape[0]

        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            mse = (1 / (2 * m)) * np.sum((y_pred - Y) ** 2)
            costs.append(mse)
            residual_weights = (1 / m) * (np.dot(X.T, y_pred - Y))
            residual_bias = (1 / m) * np.sum(y_pred - Y)

            self.weights -= self.learning_rate * residual_weights
            self.bias -= self.learning_rate * residual_bias

        return costs

    def predict(self, X, Y):
        predictions = np.dot(X, self.weights) + self.bias
        predictions = np.rint(predictions)
    
        Y = pd.concat([pd.Series(predictions, index=Y.index, name='pred'), Y], axis=1)
        
        return Y


if __name__ == '__main__':
    model = LinearRegression(learning_rate=0.003, iterations=10000, tolerance=0.000001)
    X_train, X_test, Y_train, Y_test = model.preprocess_data()
    costs = model.fit(X_train, Y_train)
    prediction_df = model.predict(X_test, Y_test)

    print(prediction_df.head())

    plt.figure()
    plt.scatter(x=list(range(len(costs))), y=costs)
    plt.show()