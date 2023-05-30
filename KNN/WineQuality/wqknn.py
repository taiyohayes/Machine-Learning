import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns', None)

wine_df = pd.read_csv('winequality.csv')

y = wine_df['Quality']
X = wine_df.drop(columns=['Quality'], axis=1)
scaler = StandardScaler()
scaler.fit(X)
X_stan = pd.DataFrame(scaler.transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_stan, y, random_state=2023, test_size=0.2, stratify=y)
X_trainA, X_trainB, y_trainA, y_trainB = train_test_split(X_train, y_train, random_state=2023, test_size=0.25,
                                                          stratify=y_train)

ks = np.arange(1, 31)
trainA_acc = []
trainB_acc = []
for k in ks:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_trainA, y_trainA)
    y_predA = model.predict(X_trainA)
    y_predB = model.predict(X_trainB)
    trainA_acc.append(accuracy_score(y_trainA, y_predA))
    trainB_acc.append(accuracy_score(y_trainB, y_predB))

plt.plot(ks, trainA_acc, 'o-', label='Train A')
plt.plot(ks, trainB_acc, 'o-', label='Train B')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(ks)

print('Best K based on validation (train B) accuracy is 6, with 27 as a close second ')

model = KNeighborsClassifier(n_neighbors=6)
model.fit(X_trainA, y_trainA)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:')
print(cm)
ConfusionMatrixDisplay(cm).plot()

X_test['Quality'] = y_test
X_test['Pred Quality'] = y_pred
print('\nUpdated Test DataFrame:')
print(X_test)

print('\nModel Accuracy:', accuracy_score(y_test, y_pred))

plt.show()






