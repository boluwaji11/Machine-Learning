""" Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line """

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
# how many sameples and How many features?
# print(diabetes.data.shape)


# What does feature s6 represent?
# print(diabetes.DESCR)
# Represents the gluclose, blood sugar level of the individual

print(diabetes.data[1:3])
print(diabetes.target[1:3])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state=11
)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

linear_regression = LinearRegression()

linear_regression.fit(X=X_train, y=y_train)

# print out the coefficients
print(linear_regression.coef_)

# print out the intercept
print(linear_regression.intercept_)


predicted = linear_regression.predict(X_test)

expected = y_test

# plt.scatter(predicted, expected)
line = plt.plot(predicted, expected, ".")

x = np.linspace(0, 330, 100)

y = x

plt.plot(x, y)

plt.show()