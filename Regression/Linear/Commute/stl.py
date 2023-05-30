# Taiyo Hayes
# ITP 449
# HW6
# Q2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression, Ridge
from yellowbrick.regressor import ResidualsPlot

# read in data file; City feature is unnecessary and can be dropped
commuteStL = pd.read_csv("CommuteStLouis.csv")
commuteStL.drop(columns=['City'], inplace=True)

# print statistical summary
print('Statistical Summary:')
print(commuteStL.describe())

# plot histogram of ages
plt.hist(commuteStL['Age'])
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Freq')
plt.show()

# print correlation matrix of the numeric variables
print('\nCorrelation Matrix:')
print(commuteStL.corr(numeric_only=True))
# Distance and time are most statistically correlated (coefficient = 0.830241

# plot scatterplot matrix of the same data as above
sb.pairplot(commuteStL[['Age', 'Distance', 'Time']])
plt.show()
# The figures on the diagonals are histograms of the various columns (since scaterplotting a column against itself
# makes no sense). Distance and Time are very right-skewed, while age is less so.

# plot a boxplot of distance traveled by sex
sb.boxplot(x='Sex', y='Distance', data=commuteStL)
plt.show()
# Yes, women tend to drive shorter distances

# plot time vs distance as a scatterplot, and add a linear regression line
model = LinearRegression()
x = commuteStL['Distance']
X = x[:, np.newaxis]
y = commuteStL['Time']
model.fit(X, y)
y_pred = model.predict(X)
plt.scatter(x, y)
plt.plot(x, y_pred)
plt.title('Scatterplot and Linear Regression of Time vs Distance')
plt.show()

# plot the distribution of residuals
ridge = Ridge()
visualizer = ResidualsPlot(ridge)
visualizer.fit(X, y)
visualizer.show()

