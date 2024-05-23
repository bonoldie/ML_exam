import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from sklearn import datasets
import sklearn.discriminant_analysis
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.manifold

# import cupy as cp


os.system("cls")


class LDA_mia:

    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = (
            None  # where i will store the e.vectors that I compute
        )

    def fit(
        self, X, y
    ):  # remember "y",  because LDA is a supervised technique, so it wants the labels. PCA ws unsupervised
        n_feature = X.shape[
            1
        ]  # (150,4); index zero is the number of samples, so index 1 in the number of features
        class_labels = np.unique(y)  # return the unique value in the labels in a list

        # Now I calculate the 2 scatter matrices, intra-class and between class
        mean_overall = np.mean(X, axis=0)
        S_I = np.zeros(
            (n_feature, n_feature)
        )  # (4,4); initialize matrix for the intra-class
        S_B = np.zeros((n_feature, n_feature))  # (4,4);

        # Now I apply the formulas. see the photo in the folder
        for c in class_labels:
            X_class = X[y == c]
            mean_class = np.mean(X_class, axis=0)

            # I do the transpose in the first term, despite in the formula the transpose is in the second term because I need to adjust the dimension of the first element, because the results must be a 4 by 4 matrix.
            # The dimension until now are:
            # dimension of:(X_class - mean_class) --> (n_samples, 4) * (n_samples, 4) --> I want a 4 by 4 as a results and so I need to transpose the first element.
            S_I += np.dot((X_class - mean_class).T, (X_class - mean_class))

            # Now I pass at the between class scatter matrix
            n_samples_current_class = X_class.shape[
                0
            ]  # index zero because I want the number of samples
            # Now in the formula I have the difference between the mean of this class and the total mean
            mean_diff = mean_class - mean_overall  # shape (4,1)
            mean_diff = (mean_class - mean_overall).reshape(
                n_feature, 1
            )  # reshape because I need a matrix S_B 4 by 4 and now I have the multiplication of a (4,1)*(4,1), so I reshape the second matrix
            S_B += n_samples_current_class * np.dot((mean_diff), mean_diff.T)

            # now I multiply the inverse of the intra class scatter with the between class scatter
            print("I'm doing the inverse... It takes a lot of time :( ")
            S_I_inv = scipy.linalg.inv(S_I)
            A = np.dot(S_I_inv, S_B)

            # now e.values and e.vectors and then sort them, as in PCA
            eigenvalues, eigenvectors = scipy.linalg.eigh(A)
            eigenvectors = (
                eigenvectors.T
            )  # eigenvector v = [:,i] column vector, transpose for easier calculations
            idxs = np.argsort(abs(eigenvalues))[::-1]  # decreasing order
            eigenvalues = eigenvalues[idxs]
            eigenvectors = eigenvectors[idxs]
            self.linear_discriminants = eigenvectors[
                0 : self.n_components
            ]  # from the element zero (larger) to the number that I specify

    def transform(
        self, X
    ):  # same as PCA, I want to get the new features that I want to project
        # project data
        projection = np.dot(X, self.linear_discriminants.T)
        return projection


class PCA_mia:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance, function needs samples as columns
        cov = np.cov(X.T)

        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(cov)

        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # store first n eigenvectors
        self.components = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)


############# TESTING PHASE #############

img_list = []

for i in range(1, 10):
    img_list.append(plt.imread(f"6_Labo/dataset2/00{i}.bmp"))

for i in range(10, 61):
    img_list.append(plt.imread(f"6_Labo/dataset2/0{i}.bmp"))

labels = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
]

img_list = np.reshape(img_list, [60, 112 * 92])  # 50 elements x 10k features


###### PCA Luca ######
pca = PCA_mia(n_components=3)
pca.fit(img_list)
luca_PCA_projection = pca.transform(img_list)
print("PCA Luca:", np.shape(luca_PCA_projection))

###### PCA sklearn ######
pca_sk = sklearn.decomposition.PCA(n_components=3)
sklearn_PCA_projection = pca_sk.fit_transform(img_list)
print("PCA sklearn:", np.shape(luca_PCA_projection))


###### LDA Luca ######
lda = LDA_mia(n_components=2)
lda.fit(img_list, labels)
luca_LDA_projection = lda.transform(img_list)
print("LDA Luca:", np.shape(luca_LDA_projection))

###### LDA sklearn ######
lda_sk = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=3)
sklearn_LDA_projection = lda_sk.fit_transform(img_list, labels)
print("LDA sklearn:", np.shape(sklearn_LDA_projection))


fig = plt.figure()
ax1 = fig.add_subplot(141, projection="3d")
plt.title("Luca's PCA")
ax2 = fig.add_subplot(142, projection="3d")
plt.title("sklearn's PCA")
ax3 = fig.add_subplot(143, projection="3d")
plt.title("Luca's LDA")
ax4 = fig.add_subplot(144, projection="3d")
plt.title("sklearn's LDA")


ax1.scatter(
    luca_PCA_projection[:, 0],
    luca_PCA_projection[:, 1],
    luca_PCA_projection[:, 2],
    c=labels,
    edgecolor="none",
    alpha=0.8,
    cmap=plt.colormaps.get_cmap("viridis"),
)

ax2.scatter(
    sklearn_PCA_projection[:, 0],
    sklearn_PCA_projection[:, 1],
    sklearn_PCA_projection[:, 2],
    c=labels,
    edgecolor="none",
    alpha=0.8,
    cmap=plt.colormaps.get_cmap("viridis"),
)

ax3.scatter(
    luca_LDA_projection[:, 0],
    luca_LDA_projection[:, 1],
    luca_LDA_projection[:, 2],
    c=labels,
    edgecolor="none",
    alpha=0.8,
    cmap=plt.colormaps.get_cmap("viridis"),
)

ax4.scatter(
    sklearn_LDA_projection[:, 0],
    sklearn_LDA_projection[:, 1],
    sklearn_LDA_projection[:, 2],
    c=labels,
    edgecolor="none",
    alpha=0.8,
    cmap=plt.colormaps.get_cmap("viridis"),
)

plt.show()
