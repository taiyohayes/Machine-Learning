# Taiyo Hayes
# ITP 259 Spring 2023
# HW5

from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import random

# 1. Set random seed to 2023, for reproducibility of results.
random.seed(2023)

# 2. Load the dataset.
fashion = keras.datasets.fashion_mnist

# 3. Separate the dataset into feature set and target variable. Also separate the train and test partitions.
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()

# 4. Print the shapes of the train and test sets for the features and target.
print("Shape of features for train set:", train_images.shape)
print("Shape of features for test set:", test_images.shape)
print("Shape of target for train set:", train_labels.shape)
print("Shape of target for test set:", test_labels.shape)

# 5. Is the target variable values clothing or numbers?
# The target variable is numbers (int), not clothing (strings)

# 6. If it is numbers, then how would you map numbers to clothing? Hint: Use a data dictionary
apparel_dict = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt",
                7: "Sneaker", 8: "Bag", 9: "Ankle boot"}
train_labels_mapped = np.array([apparel_dict[label] for label in train_labels])

# 7. Show a histogram (count) of the apparel.
plt.figure(1)
sb.countplot(x=train_labels_mapped)

# 8. Display 25 random apparel from the train dataset
sample_idxs = random.sample(range(60000), 25)
idx = 0
fig, ax = plt.subplots(5, 5)
for n in range(5):
    for i in range(5):
        ax[n, i].imshow(np.array(train_images[sample_idxs[idx]]).reshape(28, 28), cmap='gray')
        ax[n, i].set_xlabel(train_labels_mapped[sample_idxs[idx]], size=5, labelpad=0.5)
        ax[n, i].set(xticks=[], yticks=[])
        idx += 1
plt.subplots_adjust(hspace=0.35)

# 9. Scale the train and test features
train_images = train_images/255
test_images = test_images/255

# 10. Create a keras model of sequence of layers.
model = keras.models.Sequential()
    # a. One Flatten layer and two dense layers.
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(units=100, activation='relu'))
model.add(keras.layers.Dense(units=100, activation='relu'))
    # b. Experiment with number of neurons and activation functions.

# 11. Add a dense layer as output layer. Choose the appropriate number of neurons and activation function
model.add(keras.layers.Dense(units=10, activation='softmax'))

# 12. Display the model summary
print("Model summary:")
model.summary()

# 13. Set the model loss function as sparse_categorical_crossentropy. Set the optimizer as sgd.
# Set the metrics as accuracy
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 14. Fit to train the model. Use at least 100 epochs
h = model.fit(train_images, train_labels, epochs=100, verbose=True)

# 15. Plot the loss curve
pd.DataFrame(h.history).plot()
plt.show()

# 16. Display the accuracy of your model
print("\nAccuracy:", model.evaluate(test_images, test_labels)[1])

# 17. Now, display the predicted apparel of the first row in the test dataset. Also display the actual apparel.
# Show both actual and predicted letters (as title) on the image of the apparel.
test_pred = model.predict(test_images)
test_pred = np.argmax(test_pred, axis=1)
plt.figure(4)
plt.imshow(np.array(test_images[0]).reshape(28, 28), cmap='gray')
plt.title("The actual apparel is " + apparel_dict[test_labels[0]])
plt.xlabel("The predicted apparel is " + apparel_dict[test_pred[0]])

# 18. Finally, display the actual and predicted label of a misclassified apparel.
test_images = pd.DataFrame(test_images.reshape(10000, -1))
failed = test_images[test_pred != test_labels]
failed_index = failed.sample(n=1).index
plt.figure(5)
plt.imshow(np.array(test_images.iloc[failed_index]).reshape(28, 28), cmap='gray')
plt.title("The actual apparel is " + apparel_dict[test_labels[failed_index][0]] + " whereas the predicted apparel is "
          + apparel_dict[test_pred[failed_index][0]])

plt.show()





