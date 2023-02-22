import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

class LinearRegression:
    def __init__(self, learning_rate=0.003, iterations=10000):
        self.learning_rate = learning_rate
        self.iterations = iterations
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

        y_pred = None
        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            mse = (1 / (2 * m)) * np.sum((y_pred - Y) ** 2)
            costs.append(mse)
            residual_weights = (1 / m) * (np.dot(X.T, y_pred - Y))
            residual_bias = (1 / m) * np.sum(y_pred - Y)

            self.weights -= self.learning_rate * residual_weights
            self.bias -= self.learning_rate * residual_bias

        return costs, y_pred

    def predict(self, X, Y):
        predictions = np.dot(X, self.weights) + self.bias
        predictions = np.rint(predictions)
    
        Y = pd.concat([pd.Series(predictions, index=Y.index, name='pred'), Y], axis=1)
        
        return Y
    
def metrics(model, Y, Y_pred):
    mse = mean_squared_error(Y_train, train_predictions)

    print("Parameters: learning_rate = {} iterations = {}".format(model.learning_rate, model.iterations))
    print("Estimated Coefficients: {}\n".format(model.weights))

    print("The model performance for training set")
    print("--------------------------------------")
    print('MSE is {}'.format(mse))
    print('R2 score is {}'.format(r2_score(Y_train, train_predictions)))
    print('Explained variance score is {}\n'.format(explained_variance_score(Y_train, train_predictions)))

    return mse


if __name__ == '__main__':
    model = LinearRegression(learning_rate=0.001, iterations=10000)
    X_train, X_test, Y_train, Y_test = model.preprocess_data()

    std_out = sys.stdout

    sys.stdout = open('log.txt', 'w')
    best_mse = float('inf')
    best_learning_rate = 0
    best_iterations = 0
    for learning_rate in [0.003, 0.005, 0.008]:
        for iterations in [10000, 15000]:
            model.learning_rate = learning_rate
            model.iterations = iterations
            costs, train_predictions = model.fit(X_train, Y_train)
            mse = metrics(model, Y_test, train_predictions)
            if mse < best_mse:
                best_mse = mse
                best_learning_rate = learning_rate
                best_iterations = iterations

    model.learning_rate = best_learning_rate
    model.iterations = best_iterations
    costs, train_predictions = model.fit(X_train, Y_train)
    prediction_df = model.predict(X_test, Y_test)
    sys.stdout = std_out

    metrics(model, Y_test, prediction_df['pred'])

    print("The model performance for testing set")
    print("--------------------------------------")
    print('MSE is {}'.format(mean_squared_error(Y_test, prediction_df['pred'])))
    print('R2 score is {}'.format(r2_score(Y_test, prediction_df['pred'])))
    print('Explained variance score is {}\n'.format(explained_variance_score(Y_test, prediction_df['pred'])))

    plt.figure()
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.title('MSE vs. Iterations')
    plt.scatter(x=list(range(len(costs))), y=costs)
    plt.show()