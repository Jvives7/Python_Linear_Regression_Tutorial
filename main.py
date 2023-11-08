# Tutorial on how to use linear regression with Python
# Following tutorial from https://realpython.com/linear-regression-in-python/

######### Using scikit_learn classes #########
### Step 1: Import packages and classes ###
import numpy as np
# sklearn.learn_model class used to perform linear and polynomial regression and make predictions accordingly
from sklearn.linear_model import LinearRegression


### Step 2: Provide Data ###
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])


### Step 3: Create a model and fit it ###
model = LinearRegression().fit(x,y)


### Step 4: Get Results ###
# Obtain the coefficient R^2 with .score()
r_sq = model.score(x,y)
print(f'coefficient of determination: {r_sq}') # f is formatted string literal which may contain replacement fields denoted by {}

# Obtain the coefficient of intercept which represents bnaught with .intercept_
b_naught = model.intercept_
print(f'coefficient of intercept, b_naught: {b_naught}', type(b_naught)) # Data type of b_1 coefficient is a scalar

# Obtain the coeffient for slope which represents b1 with .coef_
b_1 = model.coef_
print(f'coefficient of slope, b_1: {b_1}', type(b_1)) # Data type of b_1 coefficient is an array


### Step 5: Predict Response ###
# 1st Method
y_pred_1 = model.predict(x) #passes the regressor as the argument and returns the corresponding predicteed response
print(f'1st method predicted response:\n {y_pred_1}')

# 2nd Method
y_pred_2 = model.intercept_ + model.coef_ * x.reshape(-1) # .reshape converts dimensions
print(f'2nd method predicted response:\n {y_pred_2}')
#


######### Using scikit_learn classes #########


