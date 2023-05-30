# Taiyo Hayes
# ITP 449
# HW7
# Q1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression, Ridge
from yellowbrick.regressor import ResidualsPlot

pd.set_option('display.max_columns', None)

# read in the csv file as a dataframe
auto_df = pd.read_csv("auto-mpg.csv")

# A. Summarize the data set. What is the mean of mpg?
print(auto_df.describe())
print("\nThe mean of mpg is:", auto_df.describe()['mpg']['mean'])

# B. What is the median value of mpg?
print("The median of mpg is:", auto_df.describe()['mpg']['50%'])

# C. Which value is higher – mean or median? What does this indicate in terms of the skewness of the attribute values?
# Make a plot to verify your answer. Hint: Look
print("The mean is higher than the median, which indicates right skewness (see figure 1)")
plt.figure(1)
plt.hist(auto_df['mpg'])
plt.xlabel('mpg')
plt.ylabel('count')

# D. Plot the pairplot matrix of all the relevant numeric attributes. (don’t consider No)?
auto_df.drop(['No', 'car_name'], axis=1, inplace=True)
sb.pairplot(auto_df)
plt.show()

# E. Based on the pairplot matrix, which two attributes seem to be most strongly linearly correlated?
print("\nCylinders and displacement appear to be very strongly correlated, as do weight and displacement")

# F. Based on the pairplot matrix, which two attributes seem to be most weakly correlated.
print("Model_year and acceleration are most weakly correlated")

# G. Produce a scatterplot of the two attributes mpg and displacement (displacement on the x axis and mpg on the y axis)
plt.figure(3)
x = auto_df['displacement']
y = auto_df['mpg']
plt.scatter(x, y)
plt.xlabel('displacement')
plt.ylabel('mpg')

# H. Build a linear regression model with mpg as the target and displacement as the predictor.
# Answer the following questions based on the regression model.
model = LinearRegression()
X = auto_df[['displacement']]
model.fit(X, y)

# a. For your model, what is the value of the intercept β0?
print("\nLinear Regression Model")
print("Intercept:", model.intercept_)

# b. For your model, what is the value of the coefficient β1 of the attribute displacement?
print("Slope:", model.coef_[0])

# c. What is the regression equation as per the model?
print("Full equation:", model.intercept_, '+', str(model.coef_[0]) + 'x')

# d. For your model, does the predicted value for mpg increase or decrease as the displacement increases?
print("The predicted value of mpg decreases as displacement increases (negative correlation)")

# e. Given a car with a displacement value of 200, what would your model predict its mpg to be?
sample_pred = model.predict(np.array([200]).reshape(1,-1))
print("The predicted mpg for car with displacement of 200 is:", sample_pred[0])

# f. Display a scatterplot of the actual mpg vs displacement and superimpose the linear regression line.
y_pred = model.predict(X)
plt.scatter(x, y)
plt.plot(x, y_pred, 'r-')
plt.xlabel('displacement')
plt.ylabel('mpg')
plt.show()

# g. Plot the residuals
ridge = Ridge()
visualizer = ResidualsPlot(ridge)
visualizer.fit(X, y)
visualizer.show()