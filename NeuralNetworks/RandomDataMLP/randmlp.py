# Taiyo Hayes
# ITP 259 Spring 2023
# HW3 Part B

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier

np.random.seed(0)

# 1. Generate x, y coordinates of spirally distributed blobs in two colors. (2)
N = 400
theta = np.sqrt(np.random.rand(N))*2*np.pi # np.linspace(0,2*pi,100)

r_a = 2*theta + np.pi
data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
x_a = data_a + np.random.randn(N,2)

r_b = -2*theta - np.pi
data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
x_b = data_b + np.random.randn(N,2)

res_a = np.append(x_a, np.zeros((N,1)), axis=1)
res_b = np.append(x_b, np.ones((N,1)), axis=1)

res = np.append(res_a, res_b, axis=0)
np.random.shuffle(res)

# 2. Display a scatter plot of the x and y coordinates using the label as color. Label is the spiral number
#    such as 0 and 1. You may use any color map i.e., the colors corresponding to 0 and 1. (2)
plt.scatter(res[:,0],res[:,1], c=res[:,2], s=8)
plt.title("Spirals")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 3. Create partitions with 70% train dataset. Stratify the split. Use random state of 2023. (2)
X_train, X_test, y_train, y_test = train_test_split(res[:, [0, 1]], res[:, 2], stratify=res[:,2], test_size=0.3, random_state=2023)

# 4. Now train the network using MLP Classifier from scikit learn. The parameters are your choice. (2)
model = MLPClassifier(hidden_layer_sizes=(7, 7), activation='relu', max_iter=100, alpha=1e-3,
                      solver="adam", random_state=2023, learning_rate_init=0.01)
model.fit(X_train, y_train)

# 5. Plot the loss curve. Label X and Y axis.  Add a title. (2)
plt.plot(model.loss_curve_)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# 6. Print the accuracy of the test partition (2)
print("Accuracy is", model.score(X_test, y_test))

# 7. Display the confusion matrix (2)
y_pred = model.predict(X_test)
cf = confusion_matrix(y_test, y_pred, labels=model.classes_)
labels = ['purple', 'yellow']
ConfusionMatrixDisplay(cf, display_labels=labels).plot()
plt.show()

# 1.8.	Plot the decision boundary (along with the original spirals). The decision boundary is the line where
#       samples of one class are on one side and samples of another class are on the other side. (6)

# a.	To plot the decision boundary, create a mesh of x and y coordinates that cover the entire field
#       (e.g., -20 to 20 for both x and y coordinates).
# b.	You can make the mesh points 0.1 apart. So, you will have a 400x400 mesh grid.
X1 = np.arange(-20, 20, 0.1)
X2 = np.arange(-20, 20, 0.1)
X1, X2 = np.meshgrid(X1, X2)

# c.	Then reshape the meshgrid to a dataframe that has two columns and 160000 rows (each row is a mesh point).
X_decision = pd.DataFrame({'A': np.reshape(X1, 160000), 'B': np.reshape(X2, 160000)})

# d.	Then classify each point using the trained model (model.predict)
Z = model.predict(X_decision)

# e.	Then plot both the original data points (spirals) and the mesh data points. This generates the decision
#       boundary as shown below (green vs light blue). Use color maps of your choice.
plt.scatter(X_decision['A'], X_decision['B'], c=Z, cmap='BuGn')
plt.scatter(res[:,0],res[:,1], c=res[:,2], s=8)
plt.title("Decision boundary and input dataset")
plt.show()
