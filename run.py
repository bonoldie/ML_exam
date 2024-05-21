import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.manifold
from package.utils import logger
from package.reduction.pca import PCA
import h5py

# 10099 (32x32) images, 1 channel
ds_10099_r32x32x1 = 'food_c101_n10099_r32x32x1'

# 10099 (64x64) images, 1 channel
ds_110099_r64x64x1 = 'food_c101_n10099_r64x64x1'

dsH5 = h5py.File(f'.ds/{ds_10099_r32x32x1}.h5', 'r')

# Load images and vectorization
images = dsH5['images']
images = np.reshape(images, (images.shape[0], images.shape[1]*images.shape[2] ))

unique_labels = dsH5['category_names']

# 10099x101 array
labels = dsH5['category']
# 10099 vector
images_labels = np.array(np.transpose(np.nonzero(labels[:]))[:, 1])

logger.get().info(f'images: {images.shape}')
logger.get().info(f'unique_labels: {unique_labels.shape}')
logger.get().info(f'images_labels: { images_labels.shape}')

PCA_fit = PCA(images,2)
PCA_reduced_images = PCA_fit.extract_features(images)
logger.get().info(f'PCA_reduced_images: { PCA_reduced_images.shape}')

# sklearnPCA = sklearn.decomposition.PCA(2).fit(images)
# sklearnPCA_reduced_images =  sklearnPCA.transform(images)
# logger.get().info(f'sklearnPCA_reduced_images: { sklearnPCA_reduced_images.shape}')

tsne = sklearn.manifold.TSNE()
tsne_reduced_images = tsne.fit_transform(images)

# TODO: Fisher's LDA
# borrowing from sklearn for now
clf = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
LDA_reduced_images = clf.fit(images, images_labels).transform(images)
logger.get().info(f'LDA_reduced_images: { LDA_reduced_images.shape}')

plt.subplot(1,3,1)
plt.title('PCA')
for i in range(4):
    idx = i * 20
    plt.scatter(PCA_reduced_images[images_labels == idx, 0], PCA_reduced_images[images_labels == idx, 1])

plt.subplot(1,3,2)
plt.title('LDA')
for i in range(4):
    idx = i * 20
    plt.scatter(LDA_reduced_images[images_labels == idx, 0], LDA_reduced_images[images_labels == idx, 1])

plt.subplot(1,3,3)
plt.title('t-SNE')
for i in range(4):
    idx = i * 20
    plt.scatter(tsne_reduced_images[images_labels == idx, 0], tsne_reduced_images[images_labels == idx, 1])


plt.show()

