import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# prints metrics such as MSE, R2, EVS for the model
def metrics(model, Y_train, Y_train_pred, Y_test, Y_test_pred):
    mse = mean_squared_error(Y_train, Y_train_pred)

    print("Parameters: learning_rate = {} iterations = {}".format(model.eta0, model.max_iter))
    print("Estimated Coefficients: {}\n".format(model.coef_))

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

if __name__ == '__main__':
    # read the data into a csv
    df = pd.read_csv('https://raw.githubusercontent.com/MertBuyulu/linear-regression/main/abalone.data', header=None,
                        names=['sex', 'length', 'diam', 'height', 'w_weight', 'shk_weight', 'v_weight', 'shl_weight', 'rings'])

    # Preprocess the data
    df['sex'].replace(['M', 'F', 'I'], [0, 1, 3], inplace=True)
    X = df.drop(['rings'], axis = 1)
    Y = df['rings']

    # Standardize or normalize
    X_col = X.columns
    scaler = s = StandardScaler()
    X = pd.DataFrame(scaler.fit(X).fit_transform(X), columns=X_col)

    # Split the data into 2 parts: training and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    # Model instatiation with appropriate hyperparameters
    model = SGDRegressor(loss="squared_error", alpha=0.000001, eta0=0.0001, max_iter = 50000, tol = 0.000001)

    # Fit the model to the training data using gradient descent
    model.fit(X_train, Y_train)

    # route stdout to logfile for parameter tuning log
    std_out = sys.stdout
    sys.stdout = open('log.txt', 'w')

    # save best parameters
    best_mse = float('inf')
    best_train_predictions = None
    best_test_predictions = None
    best_eta0 = 0
    best_max_iter = 0

    # parameter grid, 3 learning rates and 2 iteration counts
    for eta in [0.003, 0.005, 0.008]:
        for iterations in [10000, 15000]:
            model.eta0 = eta
            model.max_iter = iterations
            model.fit(X_train, Y_train)
            y_train_predict = model.predict(X_train)
            y_test_predict = model.predict(X_test)
            mse = metrics(model, Y_train, y_train_predict, Y_test, y_test_predict)

            # check if best error
            if mse < best_mse:
                best_eta0 = eta
                best_max_iter = iterations
                best_mse = mse
                best_train_predictions = y_train_predict
                best_test_predictions = y_test_predict

    # close logfile
    sys.stdout = std_out

    # set best parameters
    model.eta0 = best_eta0
    model.max_iter = best_max_iter

    # print metrics for best parameters
    metrics(model, Y_train, best_train_predictions, Y_test, best_test_predictions)

    # plot graphs
    figure, axis = plt.subplots(2, 2, figsize=(12, 12))
    X1 = X_test['height']
    X2 = X_test['diam']
    X3 = X_test['shl_weight']
    X4 = X_test['w_weight']
    Y2 = Y_test

    axis[0, 0].scatter(X1, Y2)
    axis[0, 0].set_title('Normalized Height vs. Predicted # of Rings')
    axis[0, 1].scatter(X2, Y2)
    axis[0, 1].set_title('Normalized Diameter vs. Predicted # of Rings')
    axis[1, 0].scatter(X3, Y2)
    axis[1, 0].set_title('Normalized Shell Weight vs. Predicted # of Rings')
    axis[1, 1].scatter(X4, Y2)
    axis[1, 1].set_title('Normalized Whole Weight vs. Predicted # of Rings')

    plt.show()
