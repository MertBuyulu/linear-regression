import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

# class that implements linear regression from scratch
class LinearRegression:
    # parameterized constructor
    def __init__(self, learning_rate=0.003, iterations=10000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0

    # reads data into dataframe, preprocesses and normalizes data for training
    def preprocess_data(self):
        # read dataset into dataframe
        df = pd.read_csv('https://raw.githubusercontent.com/MertBuyulu/linear-regression/main/abalone.data', header=None,
                        names=['sex', 'length', 'diam', 'height', 'w_weight', 'shk_weight', 'v_weight', 'shl_weight', 'rings'])
    
        # convert categorical values into numerical values
        df['sex'].replace(['M', 'F', 'I'], [0, 1, 3], inplace=True)
        X = df.drop(['rings'], axis = 1)
        Y = df['rings']

        # normalize all values
        X_cols = X.columns
        s = StandardScaler()
        X = pd.DataFrame(s.fit(X).fit_transform(X), columns=X_cols)

        # split data into train and test sets 80/20
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

        return X_train, X_test, Y_train, Y_test

    # train the model using the gradient descent algorithm
    def fit(self, X, Y):
        # array of costs for plotting
        costs = []
        self.weights = np.zeros(X.shape[1])
        m = X.shape[0]

        # gradient descent algorithm
        y_pred = None
        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            mse = (1 / (2 * m)) * np.sum((y_pred - Y) ** 2)
            costs.append(mse)
            residual_weights = (1 / m) * (np.dot(X.T, y_pred - Y))
            residual_bias = (1 / m) * np.sum(y_pred - Y)

            self.weights -= self.learning_rate * residual_weights
            self.bias -= self.learning_rate * residual_bias

        # returns list of costs per iteration, final prediction of weights on training dataset
        return costs, y_pred

    # predicts the target of the test dataset based on current weights
    def predict(self, X, Y):
        predictions = np.dot(X, self.weights) + self.bias
        predictions = np.rint(predictions)
    
        Y = pd.concat([pd.Series(predictions, index=Y.index, name='pred'), Y], axis=1)
        
        return Y
    
# prints metrics such as MSE, R2, EVS for the model
def metrics(model, Y_train, Y_train_pred, Y_test, Y_test_pred):
    mse = mean_squared_error(Y_train, Y_train_pred)

    print("Parameters: learning_rate = {} iterations = {}".format(model.learning_rate, model.iterations))
    print("Estimated Coefficients: {}\n".format(model.weights))

    print("The model performance for training set")
    print("--------------------------------------")
    print('MSE is {}'.format(mse))
    print('R2 score is {}'.format(r2_score(Y_train, Y_train_pred)))
    print('Explained variance score is {}\n'.format(explained_variance_score(Y_train, Y_train_pred)))

    print("The model performance for test set")
    print("--------------------------------------")
    print('MSE is {}'.format(mean_squared_error(Y_test, Y_test_pred)))
    print('R2 score is {}'.format(r2_score(Y_test, Y_test_pred)))
    print('Explained variance score is {}\n'.format(explained_variance_score(Y_test, Y_test_pred)))

    return mse


# main 
if __name__ == '__main__':
    # create model
    model = LinearRegression(learning_rate=0.001, iterations=10000)
    X_train, X_test, Y_train, Y_test = model.preprocess_data()

    # route stdout to logfile for parameter tuning log
    std_out = sys.stdout
    sys.stdout = open('log.txt', 'w')
    best_mse = float('inf')
    best_costs = []
    best_train_predictions = None
    best_prediction_df = None
    # parameter grid, 3 learning rates and 2 iteration counts
    for learning_rate in [0.003, 0.005, 0.008]:
        for iterations in [10000, 15000]:
            model.learning_rate = learning_rate
            model.iterations = iterations
            costs, train_predictions = model.fit(X_train, Y_train)
            prediction_df = model.predict(X_test, Y_test)
            mse = metrics(model, Y_train, train_predictions, Y_test, prediction_df['pred'])

            # check if best error
            if mse < best_mse:
                best_mse = mse
                best_train_predictions = train_predictions
                best_prediction_df = prediction_df
                best_costs = costs

    # close logfile
    sys.stdout = std_out

    # print metrics for best parameters
    metrics(model, Y_train, best_train_predictions, Y_test, best_prediction_df['pred'])

    # plot graphs
    figure, axis = plt.subplots(2, 2, figsize=(12, 12))

    X1 = list(range(len(best_costs)))
    Y1 = best_costs

    X2 = X_test['diam']
    X3 = X_test['shl_weight']
    X4 = X_test['w_weight']
    Y2 = Y_test

    axis[0, 0].plot(X1, Y1)
    axis[0, 0].set_title('MSE vs. Iterations')
    axis[0, 1].scatter(X2, Y2)
    axis[0, 1].set_title('Normalized Diameter vs. Predicted # of Rings')
    axis[1, 0].scatter(X3, Y2)
    axis[1, 0].set_title('Normalized Shell Weight vs. Predicted # of Rings')
    axis[1, 1].scatter(X4, Y2)
    axis[1, 1].set_title('Normalized Whole Weight vs. Predicted # of Rings')

    plt.show()