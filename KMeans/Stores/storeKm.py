import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sb
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns', None)

store_df = pd.read_csv('Stores.csv')
y = store_df['Store']
X = store_df.drop(columns=['Store'], axis=1)
scaler = StandardScaler()
scaler.fit(X)
X_stan = pd.DataFrame(scaler.transform(X), columns=X.columns)

ks = np.arange(1, 11)
inertias = []
for k in ks:
    model = KMeans(n_clusters=k, random_state=2023, n_init='auto')
    model.fit(X_stan)
    inertias.append(model.inertia_)
plt.figure()
plt.plot(ks, inertias, 'o-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.xticks(ks)

print("Best k is 3")

model = KMeans(n_clusters=3, random_state=2023, n_init='auto')
model.fit(X_stan)

sample = np.array([[6.3, 3.5, 2.4, 0.5]])
scaler.fit(sample)
sample_stan = scaler.transform(sample)
print("\nCluster for the sample store:", model.predict(sample))

X_stan['Store'] = y
X_stan['Cluster'] = model.labels_
print('\nUpdated DataFrame:')
print(X_stan)

plt.figure()
sb.countplot(X_stan, x='Cluster')
plt.show()