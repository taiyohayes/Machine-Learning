import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Create a DataFrame “ccDefaults” to store the credit card default data and set option to display all columns without
# any restrictions on the number of columns displayed. (2)
pd.set_option('display.max_columns', None)
ccDefaults = pd.read_csv('ccDefaults.csv')

# Determine the number of non-null samples and feature data types. (3)
print('Number of non-null samples:')
print(ccDefaults.notnull().sum())

# Display the first 5 rows of ccDefaults. (1)
print('\nFirst 4 rows of DataFrame:')
print(ccDefaults.head(5))

# Determine the dimensions of ccDefaults. (2)
print('\nDataFrame Dimensions:', ccDefaults.shape)

# Drop the ‘ID’ column from ccDefaults. (3)
ccDefaults.drop(columns=['ID'], axis=1, inplace=True)

# Drop duplicates records from ccDefaults and identify if any duplicate records are dropped by printing out the
# dimensions of ccDefaults. (2)  Hint: ccDefaults.drop_duplicates(keep='first', inplace=True)
ccDefaults.drop_duplicates(keep='first', inplace=True)
print('Dimensions after dropping duplicates:', ccDefaults.shape)

# Print the correlation between all variable pairs. (3)
print('\nCorrelation Matrix:')
print(ccDefaults.corr())

# Create a Feature Matrix, including only the 4 most correlated variables with the target, and the Target Vector.
# Hint: Look at the column of the target in the correlation matrix and see which features have the highest correlation
# with the target.
X = ccDefaults[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4']]
y = ccDefaults['dpnm']

# Partition the data 70/30. (2) random_state=2023, stratify=y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023, stratify=y)

# Develop Decision Tree Classifier model. (4)  criterion=‘entropy', max_depth=4, random_state=2023
dt = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=2023)
dt.fit(X_train, y_train)

# Display the accuracy of the model on the Test partition. (2)
y_pred = dt.predict(X_test)
print('\nAccuracy on test partition:', accuracy_score(y_test, y_pred))

# Plot the confusion matrix. (3)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

# Plot the decision tree. (4) Hint: cn = list(map(str, model.classes_.tolist())
plt.figure(figsize=(9,8))
fn = X.columns
cn = list(map(str, dt.classes_.tolist()))
loan_tree = plot_tree(dt, feature_names=fn, class_names=cn, filled=True)

plt.show()
