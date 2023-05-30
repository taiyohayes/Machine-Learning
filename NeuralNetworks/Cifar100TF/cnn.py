from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
import random

# 1. Load the dataset from keras
# 2. Partition the dataset into train and test sets.
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')

labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
          'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee',
          'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
          'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower',
          'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
          'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
          'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
          'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
          'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
          'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

# Print the shapes of the train and test data sets. (screenshot)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# 3. Visualize the first 30 images from the train dataset (screenshot)
plt.figure()
for i in range(30):
    plt.subplot(5, 6, i+1)
    plt.imshow(X_train[i])
    plt.xlabel(labels[y_train[i, 0]])
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()

# 4. Scale the pixel values
X_train = X_train/255
X_test = X_test/255

# 5. One-hot encode the classes to use the categorical cross-entropy loss function
n_classes = len(labels)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test,n_classes)

# 6. Build a CNN sequence of layers. Must contain the following layers. Hyper parameters are up to you.
# a. At least 1 convolutional layer
# b. At least 1 dropout layer
# c. At least 1 maxpool layer
# d. At least 1 flatten layer
# e. At least 1 dense layer
print(X_train.shape)
model = Sequential()
model.add(Conv2D(35, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32,32,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(45, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(n_classes, activation='softmax'))

# 7. Use the loss function categorical_crossentropy when compiling the model
# model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# 8. Train the model with at least 10 epochs
h = model.fit(X_train, Y_train, batch_size=64, epochs=15, validation_data=(X_test, Y_test))


# 9. Plot the loss and accuracy curves for both train and validation sets.
plt.figure()
plt.plot(h.history['loss'], 'black')
plt.plot(h.history['val_loss'], 'green')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.figure()
plt.plot(h.history['accuracy'], 'black')
plt.plot(h.history['val_accuracy'], 'blue')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# 10. Visualize the predicted and actual image labels for the first 30 images in the dataset.
y_pred = model.predict(X_test)
pred_labels = np.argmax(y_pred, axis=1)
plt.figure()
for i in range(30):
    plt.subplot(5, 6, i+1)
    plt.imshow(X_test[i])
    plt.subplot(5,6,i+1).set_title("True:" + str(labels[y_test[i, 0]]) + "\nPredicted:" + str(labels[pred_labels[i]]),
                                   fontsize=7)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()

# 11. Visualize 30 random misclassified images.
failed_indices = []
idx = 0
for i in y_test:
    if i[0] != pred_labels[idx]:
        failed_indices.append(idx)
    idx = idx + 1
rand_failed = random.sample(failed_indices, 30)
plt.figure()
for n, i in enumerate(rand_failed):
    plt.subplot(5, 6, n+1)
    plt.imshow(X_test[i])
    plt.subplot(5,6,n+1).set_title("True:" + str(labels[y_test[i, 0]]) + "\nPredicted:" + str(labels[pred_labels[i]]),
                                   fontsize=7)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

