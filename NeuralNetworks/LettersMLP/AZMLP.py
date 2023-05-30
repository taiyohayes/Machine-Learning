# Taiyo Hayes
# ITP 259 Spring 2023
# HW4

import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Read the dataset into a dataframe.
df = pd.read_csv('A_Z Handwritten Data.csv')

# 2. Explore the dataset and determine what is the target variable.
# 3. Separate the dataframe into feature set and target variable. (1)
targ = df['label']
df_feat = df.drop(columns=['label'])

# 4. Print the shape of feature set and target variable. (1)
# print("Features:\n", targ.shape)
# print("Targets:\n", df_feat.shape)

# 5. Is the target variable values letters or numbers? (1)
# The target value is numbers

# 6. If the target variable is numbers, then how would you map numbers to letters
word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
             12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
             23: 'X', 24: 'Y', 25: 'Z'}
targ.replace(word_dict, inplace=True)

# 7. Show a histogram (count) of the letters
# sb.countplot(df, x='label')
# plt.show()

# 8. Display 64 random letters from the dataset. Display their labels as shown below.
# Hint: Plot a pyplot figure. Use plt.subplot to make the 64 subplots. Use a for loop to iterate through each one.
# sample_idxs = random.sample(range(372451), 64)
# idx = 0
# fig, ax = plt.subplots(8, 8)
# for n in range(8):
#     for i in range(8):
#         ax[n, i].imshow(np.array(df_feat.iloc[sample_idxs[idx]]).reshape(28, 28), cmap='gray')
#         ax[n, i].set_title(targ[sample_idxs[idx]], pad=2, size=6)
#         ax[n, i].set(xticks=[], yticks=[])
#         idx += 1
# plt.subplots_adjust(wspace=0.05, hspace=0.5)
# plt.show()

# 9. Partition the data into train and test sets (70/30). Use random_state = 2023. Stratify it.
feat_train, feat_test, targ_train, targ_test = train_test_split(df_feat, targ, test_size=0.3, random_state=2023, stratify=targ)

# 10. Scale the train and test features.
feat_train /= 255
feat_test /= 255

# 11. Create an MLPClassifier. Experiment with various parameters. random_state = 2023.
model = MLPClassifier(hidden_layer_sizes=(100,100,100), activation="relu", max_iter=25, alpha=1e-3, solver="adam",
                      random_state=2023, learning_rate_init=0.01, verbose=True)

# 12. Fit to train the model.
model.fit(feat_train, targ_train)

# 13. Plot the loss curve.
# plt.plot(model.loss_curve_)
# plt.show()

# 14. Display the accuracy of your model.
print("Accuracy is", model.score(feat_test, targ_test))

# 15. Plot the confusion matrix along with the letters.
targ_pred = model.predict(feat_test)
cm = confusion_matrix(targ_pred, targ_test)
plt.figure(figsize=(50,50))
ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot()
plt.show()

# 16. Now, display the predicted letter of the first row in the test dataset. Also display the actual letter.
# Show both actual and predicted letters (as title) on the image of the letter
# feat_test.reset_index(drop=True, inplace=True)
# targ_test.reset_index(drop=True, inplace=True)
# test_sample = np.array(feat_test.iloc[0]).reshape(28, 28)
# plt.imshow(test_sample, cmap="gray")
# plt.title("The predicted letter is " + str(targ_pred[0]) + " and the actual letter is " + str(targ_test.iloc[0]))
# plt.show()

# 17. Finally, display the actual and predicted letter of a misclassified letter.
# failed_df = feat_test[targ_pred != targ_test]
# failed_index = failed_df.sample(n=1).index
# failed_sample = np.array(feat_test.iloc[failed_index]).reshape(28, 28)
# plt.imshow(failed_sample, cmap="gray")
# plt.title("The failed predicted letter is " + str(targ_pred[failed_index]) + " whereas the actual letter is " +
#           str(targ_test.iloc[failed_index].values))
# plt.show()
