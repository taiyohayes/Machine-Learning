from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

# Load the file and read the dataset into a dataframe. (1)
cancerDf = pd.read_csv('Breast_Cancer.csv')

# Make sure there are no missing values. (2)
print("Number of missing values for each column:")
print(cancerDf.isnull().sum())
print('No Missing Values!\n')

# Explore the dataset and determine what the target variable is. Define the features based on all remaining columns (4)
print('Target variable is \'diagnosis\'\n')
y = cancerDf['diagnosis']
X = cancerDf.drop(columns=['diagnosis'], axis=1)

# Get a countplot of the target. (4)
sb.countplot(cancerDf, x='diagnosis')

# Partition the data into train and test sets (75/25). Use random_state = 2023, startify = y (3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2023, stratify=y)

# Fit the training data to a logistic regression model. (3)
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display the accuracy, precision and recall of your predictions for diagnosis. (4)
# Hint: use metrics.classification_report()
print('Precision, Recall, and Accuracy:')
print(classification_report(y_test, y_pred))

# Print and plot the confusion matrix for the test set.(4)
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.show()


