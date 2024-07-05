import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.manifold
import scipy
from package.utils import logger
from package.reduction.pca import PCA
import h5py

# 10099 (32x32) images, 1 channel
ds_10099_r32x32x1 = 'food_c101_n10099_r32x32x1'

# 10099 (64x64) images, 1 channel
ds_110099_r64x64x1 = 'food_c101_n10099_r64x64x1'

# 10099 (64x64) images, 1 channel
ds_000_r384x384x3 = 'food_c101_n1000_r384x384x3'

ds_test_c101_n1000_r64x64x1 = 'food_test_c101_n1000_r64x64x1'

dsH5 = h5py.File(f'.ds/{ds_110099_r64x64x1}.h5', 'r')

dsTestH5 = h5py.File(f'.ds/{ds_test_c101_n1000_r64x64x1}.h5', 'r')

# Load images and vectorization
images = dsH5['images']
images = np.reshape(images, (images.shape[0], images.shape[1]*images.shape[2] ))

test_images = dsTestH5['images']
test_images =  np.reshape(test_images, (test_images.shape[0], test_images.shape[1]*test_images.shape[2] ))

unique_labels = np.asarray(dsH5['category_names'])
test_unique_labels = np.asarray(dsTestH5['category_names'])

labels =  np.asarray(dsH5['category'])
test_labels = np.asarray(dsTestH5['category'])

images_labels = np.argwhere(labels)[:, 1]
test_images_labels = np.argwhere(test_labels)[:, 1]
images_labels_text = unique_labels[images_labels]

logger.get().info(f'images: {images.shape}')
logger.get().info(f'unique_labels: {unique_labels.shape}')
logger.get().info(f'images_labels: { images_labels_text.shape}')

PCA_fit = PCA(images,25)
PCA_reduced_images = PCA_fit.extract_features(images)
PCA_test_images = PCA_fit.extract_features(test_images)
logger.get().info(f'PCA_reduced_images: { PCA_reduced_images.shape}')

# densities = np.empty(test_images.shape[0])
estimators = list()

pdfs = None

for i in range(unique_labels.shape[0]):
    #label_idx = i
    #label = images_labels[label_idx]
    PCA_images = PCA_reduced_images[images_labels == i, :]

    mean = np.mean(PCA_images, axis=0)

    cov = np.cov(np.transpose(PCA_images))

    mean_cond = np.any(mean > 10, axis=0)

    pdf = scipy.stats.multivariate_normal.pdf(PCA_test_images, mean=mean, cov=cov, allow_singular=True)

    # if not mean_cond:
    #     pdf = np.zeros([test_images.shape[0]])

    if pdfs is None:
        pdfs = pdf
    else:
        pdfs = np.vstack((pdfs, pdf))

preds = np.argmax(pdfs, axis=0)

print((test_images_labels.shape, preds.shape))

print([np.argwhere(test_images_labels == preds),test_images_labels[test_images_labels == preds]])

print(test_unique_labels[41])
print(test_unique_labels[80])

plt.imshow(np.reshape(test_images[100], [64,64]))
plt.show()

precision = np.count_nonzero(test_images_labels == preds) / test_images_labels.shape[0]

print(precision)

