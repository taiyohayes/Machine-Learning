import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

mush_df = pd.read_csv('mushrooms.csv')
y = mush_df['class']
X = mush_df.drop(columns=['class'])

# add the sample mushroom to the dataframe
sample = np.array(['x','s','n','t','y','f','c','n','k','e','e','s','s','w','w','p','w','o','p','r','s','u'])
X.loc[len(X)] = sample

# change to numerical using dummy variables
X = pd.get_dummies(X, drop_first=True)

# extract the sample row, and drop it from the dataframe
sample = X.loc[len(X)-1]
sample = pd.DataFrame([sample])
X.drop([len(X)-1], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2023, test_size=0.25, stratify=y)
dt = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=2023)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
print('Confusion Matrix:')
print(cm)
ConfusionMatrixDisplay(cm).plot()

print('\nTraining Accuracy', dt.score(X_train, y_train))
print('Testing Accuracy', dt.score(X_test, y_test))

plt.figure()
fn = X.columns
cn = y.unique()
mush_tree = plot_tree(dt, feature_names=fn, class_names=cn, filled=True)

importance = pd.DataFrame(dt.feature_importances_, index=X.columns).sort_values(by=0, ascending=False)
print('\n3 most important features:', importance.iloc[:3].index.tolist())

sample_pred = dt.predict(sample)
print('\nThe sample mushroom is classified as', sample_pred[0])

plt.show()
