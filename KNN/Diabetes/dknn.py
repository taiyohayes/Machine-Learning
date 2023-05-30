# Taiyo Hayes
# ITP 259 Spring 2023
# HW2, Problem 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. Create a DataFrame “diabetes_knn” to store the diabetes data. (1)
diabetes_knn = pd.read_csv("diabetes.csv")

# 2. Create the Feature Matrix (X) and Target Vector (y). (1)
X = diabetes_knn.iloc[:, 0:8]
y = diabetes_knn.iloc[:, 8]

# 3. Standardize the attributes of Feature Matrix (2)
scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), columns=X.columns)

# 4. Split the Feature Matrix and Target Vector into three partitions. Training A, Training B and test.
#    They should be in the ratio 60-20-20. random_state = 2023, stratify = y.  (1)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=2023, stratify=y)
X_trainA, X_trainB, y_trainA, y_trainB = train_test_split(X_temp, y_temp, test_size=0.25, random_state=2023)

# 5. Develop a KNN based model based on Training A for various ks. K should range between 1 and 30. (1)
# 6. Compute the KNN score (accuracy) for training A and training B data for those ks. (2)
ks = np.arange(1, 31)
trainA_accuracy = np.empty(30)
trainB_accuracy = np.empty(30)
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_trainA, y_trainA)
    trainA_accuracy[k-1] = knn.score(X_trainA, y_trainA)
    trainB_accuracy[k-1] = knn.score(X_trainB, y_trainB)

# 7. Plot a graph of training A and training B accuracy and determine the best value of k. Label the plot. (1)
plt.plot(ks, trainA_accuracy, label="Training A accuracy")
plt.plot(ks, trainB_accuracy, label="Training B accuracy")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# k = 19

# 8. Now, using the selected value of k, score the test data set (1)
knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(X_test, y_test)
print("Accuracy:", knn.score(X_test, y_test))

# 9. Plot the confusion matrix (as a figure). (1)
y_pred = knn.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=knn.classes_).plot()
plt.show()

# 10. Predict the Outcome for a person with 2 pregnancies, 150 glucose, 85 blood pressure, 22 skin thickness,
#     200 insulin, 30 BMI, 0.3 diabetes pedigree, 55 age. (1)
sample = np.array([[2, 150, 85, 22, 200, 30, 0.3, 55]])
print(knn.predict(sample))
