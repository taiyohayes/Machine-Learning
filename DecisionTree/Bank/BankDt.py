import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
bank_df = pd.read_csv('UniversalBank.csv')

y = bank_df['Personal Loan']
print('The target variable is \"Personal Loan\"')

X = bank_df.drop(columns=['Row', 'Personal Loan', 'ZIP Code'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2023, stratify=y)

print('\nNumber of accepted cases in training partition:', np.count_nonzero(y_train))

dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=2023)
dt.fit(X_train, y_train)

plt.figure()
fn = X.columns
cn = ['not accepted', 'accepted']
loan_tree = plot_tree(dt, feature_names=fn, class_names=cn, filled=True)

trainAcc = dt.score(X_train, y_train)
y_pred = dt.predict(X_test)
testAcc = dt.score(X_test, y_test)

cm = confusion_matrix(y_test, y_pred)

print('\nAcceptors misclassified as non-acceptors:', cm[1][0])
print('Non-acceptors misclassified as acceptors:', cm[0][1])

print('\nTraining accuracy:', trainAcc)
print('Testing accuracy:', testAcc)

plt.show()
