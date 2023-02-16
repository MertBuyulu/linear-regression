### 1. PROJECT TASKS

#### PART 1 [75 points]

- [x] Choose a [dataset](https://archive.ics.uci.edu/ml/index.php) suitable for regression
- [ ] Host the dataset online [UTD Web account, AWS, etc]
- [ ] Pre-process the dataset
  - [ ] remove null or NA values
  - [ ] remove any redundant rows
  - [ ] convert categorical variables to numerical variables
  - [ ] remove attributes not suitable/correlated with the outcome
  - [ ] any other pre-processing that you may need to perform
- [ ] Split the dataset into training and test parts [80/20]
- [ ] Create a linear regression model [refer to the regression model constructed in class]
- [ ] Create a log file that containing paramaters used and error (MSE) value obtained for various trials
- [ ] Test the model and report the error values for the best set of parameters you obtained during training
- [ ] Answer: Are you satisfied that you have found the best solution? Explain

#### PART 2 [25 points)

- [ ] Create a linear model using ML library with the same dataset from part 1
- [ ] Answer: Are you satisfied that the package has found the best solution. How can you check. Explain

### 2. Additional Requirements

- Write clean code
- Optimize your parameters
- Provide as many plots as possible (e.g. MSE vs number of iterations, etc)
- Output as many evaluation statics as possible (weight coefficients, MSE, [R^2 value](https://en.wikipedia.org/wiki/Coefficient_of_determination))
- [Explained Variance](https://en.wikipedia.org/wiki/Explained_variation)

### 3. What to Submit

- Part 1 - python files
- Part 2 - python files
- README file indicating how to build and run code + the libraries used in part 2
- A report file containing log of your trials with different parameters, answer to questions, and plots

### 4. What Not to Submit

- Dataset Files
- Do not hardcode paths on your local computer
