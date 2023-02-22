# 4375 Assignment 1: Linear-Regression

This assignment implements linear regression using gradient descent on an abalone data set to predict the number of rings in the abalone's shell. Generally, the only way to find out the age of an abalone is to manually count the number of rings in its shell, a tedious and time consuming task. The goal of this regression model is to predict the number of rings using various features such as sex, length, diameter, height, whole weight, shucked weight, viscera weight, and shell weight. 

## Execution Instructions

This program runs with `Python 3.9.12` and above.

The following imports are required to run `part1.py`:

```python
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
```

The following imports are required to run `part2.py`:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
```

Run the program using any acceptable terminal with:

```bash
python <code_file>
```

where:

`code_file` can take the following values:
- `part1.py`: for running the linear regression algorithm implemented from scratch
- `part2.py`: for running the Scikit-Learn implementation of linear regression

## Dataset

The abalone dataset is hosted on our GitHub repository at:

```text
https://raw.githubusercontent.com/MertBuyulu/linear-regression/main/abalone.data
```

The dataset contains `4177` instances with `8` attributes per instance.

## Output

Running either of the source code files will output training and test metrics to the console. For example, running `part1.py`:

```text
Parameters: learning_rate = 0.008 iterations = 15000
Estimated Coefficients: [-0.34081133  0.07913217  0.94393049  0.39049125  2.3087969  -3.45632211
 -0.68823224  2.12904256]

The model performance for training set
--------------------------------------
MSE is 4.730770344750821
R2 score is 0.5434223706871154
Explained variance score is 0.543422370702205

The model performance for testing set
--------------------------------------
MSE is 5.528708133971292
R2 score is 0.47417061946748507
Explained variance score is 0.4741842279164812
```

All parameter tuning/selection output is printed to the log file `log.txt`.