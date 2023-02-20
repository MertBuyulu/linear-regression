import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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
model = SGDRegressor(loss="squared_error", alpha=0.0000001, eta0=0.001, max_iter = 10000, tol = 0.000001)

# Fit the model to the training data using gradient descent
model.fit(X_train, Y_train)

# Optimized paramaters (weight coefficients)
weights = model.coef_
print("Estimated Coefficients: {}\n".format(weights))

# Model evaluation for training set
y_train_predict = model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}\n'.format(r2))

# Model evaluation for testing set
y_test_predict = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


