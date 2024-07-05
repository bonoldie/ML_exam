import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from sklearn import datasets
import sklearn.discriminant_analysis
import os


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
            S_I_inv = np.linalg.inv(S_I)
            A = np.dot(S_I_inv, S_B)

            # now e.values and e.vectors and then sort them, as in PCA
            eigenvalues, eigenvectors = np.linalg.eigh(A)
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


###### TESTING ######

data = datasets.load_iris()
X, y = data.data, data.target

# Project the data onto the 2 primary linear discriminants
lda = LDA_mia(n_components=2)
lda.fit(X, y)
X_projected = lda.transform(X)
print("Shape of X MIA:", X.shape)
print("Shape of transformed X MIA:", X_projected.shape)


clf = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
LDA_reduced_images = clf.fit(X, y).transform(X)
print("Shape of X SKLEARN:", X.shape)
print("Shape of transformed X SKLEARN:", LDA_reduced_images.shape)


fig = plt.figure()

plt.subplot(2, 2, 1)
plt.title("row data")
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    edgecolor="none",
    alpha=0.8,
    cmap=plt.colormaps.get_cmap("viridis"),
)

plt.subplot(2, 2, 2)
plt.title("row data")
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    edgecolor="none",
    alpha=0.8,
    cmap=plt.colormaps.get_cmap("viridis"),
)


plt.subplot(2, 2, 3)
plt.title("LDA mia")
plt.scatter(
    X_projected[:, 0],
    X_projected[:, 1],
    c=y,
    edgecolor="none",
    alpha=0.8,
    cmap=plt.colormaps.get_cmap("viridis"),
)

plt.subplot(2, 2, 4)
plt.title("LDA sklearn")
plt.scatter(
    LDA_reduced_images[:, 0],
    LDA_reduced_images[:, 1],
    c=y,
    edgecolor="none",
    alpha=0.8,
    cmap=plt.colormaps.get_cmap("viridis"),
)

plt.show()
