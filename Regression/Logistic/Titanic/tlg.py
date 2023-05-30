# Taiyo Hayes
# ITP 259 Spring 2023
# HW2, Problem 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scikitplot.metrics import plot_lift_curve

# 1. Read the dataset into a dataframe. (1)
titanicData = pd.read_csv("Titanic.csv")

# 2. Explore the dataset and determine what is the target variable. (1)
# Target variable is "Survived"

# 3. Drop factor(s) that are not likely to be relevant for logistic regression. (2)
survivability = titanicData["Survived"]
titanicData.drop(['Passenger', 'Survived'], axis=1, inplace=True)

# 4. Convert all categorical feature variables into dummy variables. (2)
dumTitanicData = pd.get_dummies(titanicData, drop_first=True)

# 5. Assign X and y (1)
X = dumTitanicData
y = survivability

# 6. Partition the data into train and test sets (70/30). Use random_state = 2023. Stratify the data. (2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023, stratify=y)

# 7. Fit the training data to a logistic regression model. (1)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 8. Display the accuracy of your predictions. (2)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 9. Plot the lift curve. (1)
plot_lift_curve(y_test, X_test)
plt.show()

# 10. Plot the confusion matrix along with the labels (Yes, No).  (2)
matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_).plot()
plt.show()

# 11. Now, display the predicted value of the survivability of a male adult passenger traveling in 3rd class. (3)
sample = np.array([[0, 1, 0, 1, 0]])
print(model.predict(sample))
