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

# 10099 (64x64) images, 1 channel
ds_000_r384x384x3 = 'food_c101_n1000_r384x384x3'

dsH5 = h5py.File(f'.ds/{ds_110099_r64x64x1}.h5', 'r')

# Load images and vectorization
images = dsH5['images']
images = np.reshape(images, (images.shape[0], images.shape[1]*images.shape[2] ))

unique_labels = np.asarray(dsH5['category_names'])

labels =  np.asarray(dsH5['category'])

images_labels = np.argwhere(labels)[:, 1]
images_labels_text = unique_labels[images_labels]

logger.get().info(f'images: {images.shape}')
logger.get().info(f'unique_labels: {unique_labels.shape}')
logger.get().info(f'images_labels: { images_labels_text.shape}')

images_1 = images[images_labels == 23, :]
images_2 = images[images_labels == 10, :]
images_3 = images[images_labels == 58, :]

fig = plt.figure()
plt.subplot(2,2,1)
plt.imshow(np.reshape(images_1[0, :],[64,64]))

plt.subplot(2,2,2)
plt.imshow(np.reshape(images_1[1, :],[64,64]))

plt.subplot(2,2,3)
plt.imshow(np.reshape(images_1[2, :],[64,64]))

plt.subplot(2,2,4)
plt.imshow(np.reshape(images_1[4, :],[64,64]))


selected_labels = np.repeat([[23],[10],[58]],[images_1.shape[0],images_2.shape[0],images_3.shape[0]])
selected_images = np.concatenate((images_1, images_2, images_3), axis=0)

PCA_fit = PCA(selected_images,3)
PCA_reduced_images = PCA_fit.extract_features(selected_images)
logger.get().info(f'PCA_reduced_images: { PCA_reduced_images.shape}')

# sklearnPCA = sklearn.decomposition.PCA(2).fit(images)
# sklearnPCA_reduced_images =  sklearnPCA.transform(images)
# logger.get().info(f'sklearnPCA_reduced_images: { sklearnPCA_reduced_images.shape}')

# tsne = sklearn.manifold.TSNE(n_components=2)
# tsne_reduced_images = tsne.fit_transform(images)

# TODO: Fisher's LDA
# borrowing from sklearn for now
# clf = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=3)
# LDA_reduced_images = clf.fit(images, images_labels).transform(images)
# logger.get().info(f'LDA_reduced_images: { LDA_reduced_images.shape}')

fig = plt.figure()
ax1 = fig.add_subplot(131, projection='3d')
# ax2 = fig.add_subplot(132, projection='3d')
# ax3 = fig.add_subplot(133, projection='3d')

mean_1 = np.mean(PCA_reduced_images[selected_labels==23])
cov_1 = np.cov(np.transpose(PCA_reduced_images[selected_labels==23]))

mean_2 = np.mean(PCA_reduced_images[selected_labels==10])
cov_2 = np.cov(np.transpose(PCA_reduced_images[selected_labels==10]))

mean_3 = np.mean(PCA_reduced_images[selected_labels==58])
cov_3 = np.cov(np.transpose(PCA_reduced_images[selected_labels==58]))

print([mean_1, mean_2, mean_3])
print([cov_1, cov_2, cov_3])

plt.title('PCA')
ax1.scatter(PCA_reduced_images[selected_labels==23, 0], PCA_reduced_images[selected_labels==23, 1], PCA_reduced_images[selected_labels==23, 2], c='b')
ax1.scatter(PCA_reduced_images[selected_labels==10, 0], PCA_reduced_images[selected_labels==10, 1], PCA_reduced_images[selected_labels==10, 2],c='r')
ax1.scatter(PCA_reduced_images[selected_labels==58, 0], PCA_reduced_images[selected_labels==58, 1], PCA_reduced_images[selected_labels==58, 2],c='g')
#ax1.scatter(PCA_reduced_images[images_labels_text == b"nachos", 0], PCA_reduced_images[images_labels_text == b"nachos", 1], PCA_reduced_images[images_labels_text == b"nachos", 2])
#ax1.scatter(PCA_reduced_images[images_labels_text == b"cheesecake", 0], PCA_reduced_images[images_labels_text == b"cheesecake", 1], PCA_reduced_images[images_labels_text == b"cheesecake", 2])
plt.show()

# plt.subplot(1,3,2)
# plt.title('LDA')
# for i in range():
#     idx = i*10
    # ax2.scatter(LDA_reduced_images[images_labels == idx, 0], LDA_reduced_images[images_labels == idx, 1], LDA_reduced_images[images_labels == idx, 2])

# plt.subplot(1,3,3)
# plt.title('t-SNE')
# for i in range(101):
#     idx = i
#     ax2.scatter(tsne_reduced_images[images_labels == idx, 0], tsne_reduced_images[images_labels == idx, 1])

# plt.show()

