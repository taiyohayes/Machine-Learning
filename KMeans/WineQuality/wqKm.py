# Taiyo Hayes
# ITP 449
# HW8

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Read the dataset into a dataframe. Be sure to import the header. (2)
wineData = pd.read_csv('wineQualityReds.csv', header=0)

# 2. Drop Wine from the dataframe. (1)
wineData.drop(columns=['Wine'], axis=1, inplace=True)

# 3. Extract Quality and store it in a separate variable. (1)
labels = wineData['quality']

# 4. Drop Quality from dataframe. (1)
wineData.drop(columns=['quality'], axis=1, inplace=True)

# 5. Print the dataframe and Quality. (1)
print("Dataframe:\n", wineData)
print("\nQuality:\n", labels)

# 6. Normalize all columns of the dataframe. Use the MinMaxScaler class from sklearn.preprocessing. (2)
scaler = MinMaxScaler()
scaler.fit(wineData)
wineDataNorm = pd.DataFrame(scaler.transform(wineData), columns=wineData.columns)

# 7. Print the normalized dataframe. (1)
print("\nNormalized Dataframe:\n", wineDataNorm)

# 8. Create a range of k values from 1-20 for k-means clustering. Iterate on the k values and store the
# inertia for each clustering in a list. Pass random_state = 2023 and n_init='auto' to KMeans() (2)
ks = range(1, 21)
inertia = []
for k in ks:
    model = KMeans(n_clusters=k, random_state=2023, n_init='auto')
    model.fit(wineDataNorm)
    inertia.append(model.inertia_)

# 9. Plot the chart of inertia vs number of clusters. (2)
plt.plot(ks, inertia, 'o-')
plt.xlabel('Number of Clusters, k')
plt.ylabel('Inertia')

# 10. What K (number of clusters) would you pick for k-means? (1)
print('\nI would choose k = 8')
# I would choose k = 8

# 11. Now cluster the wines into K=6 clusters. Use random_state = 2023 and n_init='auto' when you instantiate the
# k-means model. Assign the respective cluster number to each wine.
# Print the dataframe showing the cluster number for each wine. (2)
model = KMeans(n_clusters=6, random_state=2023, n_init='auto')
model.fit(wineDataNorm)
wineDataNorm['Cluster'] = model.labels_
print("\nDataframe with cluster number:\n", wineDataNorm)

# 12. Add the quality back to the dataframe. (1)
wineDataNorm['quality'] = labels

# 13. Now print a crosstab (from Pandas) of cluster number vs quality. Comment if the clusters
# represent the quality of wine. (3)
print("\nCrosstab:\n", pd.crosstab(wineDataNorm['quality'], wineDataNorm['Cluster']))
# The clusters do not really represent the quality of the wine

plt.show()
