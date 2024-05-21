import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def euclidian_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance


class KNN_mia:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # passagli the tresting data
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # qua calcolo la diustanza della x che gli passo from all the points e ritorna le lable delle 3 nearest neighborought
        # compute the distances
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]

        # compute the closest k
        k_indices = np.argsort(distances)[: self.k]  # prende i primi k delle distanze
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # determine the labels with majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]


##################################################################
train_data = np.load("5_Labo/dataset1/train-data-012.npy")
train_labels = np.load("5_Labo/dataset1/train-labels-012.npy")  # lables are 0, 1, 2
test_data = np.load("5_Labo/dataset1/test-data-012.npy")
test_labels = np.load("5_Labo/dataset1/test-labels-012.npy")


train_data_0 = train_data[train_labels == 0]
train_data_1 = train_data[train_labels == 1]
train_data_2 = train_data[train_labels == 2]


plt.figure(1)
plt.scatter(train_data_0[:, 0], train_data_0[:, 1])
plt.scatter(train_data_1[:, 0], train_data_1[:, 1])
plt.scatter(train_data_2[:, 0], train_data_2[:, 1])
# plt.show()

clf = KNN_mia(k=5)
clf.fit(train_data[0:100], train_labels[0:100])
predictions = clf.predict(test_data)

print(predictions)

acc = np.sum(predictions == test_labels) / len(test_labels)
print(acc)

##################################################################