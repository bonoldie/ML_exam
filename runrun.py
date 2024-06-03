import numpy as np
from PIL import Image
import io
from matplotlib import pyplot as plt
from package.utils.logger import logger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import multivariate_normal

# Bootstrap
ds = np.load('.ds.tiny/dataset.zip')

test_image_names = []
test_images = []
test_labels = []
unique_test_labels = [] 

train_image_names = []
train_images = []
train_labels = []
unique_train_labels = []

for dsKey in ds.keys():
    splittedKey = dsKey.split('/')
    img = Image.open(io.BytesIO(ds[dsKey])).resize((128,128), Image.Resampling.LANCZOS)

    if(splittedKey[2] == 'train'):
        train_images.append(np.asarray(img))
        train_labels.append(splittedKey[3])
        train_image_names.append(splittedKey[4])
    else:
        test_images.append(np.asarray(img))
        test_labels.append(splittedKey[3])
        test_image_names.append(splittedKey[4])

train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)

test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)

unique_test_labels = np.unique(test_labels)
unique_train_labels = np.unique(train_labels)

# print(train_labels)
# print(unique_train_labels)

logger.info(['Traing images size:', train_images.shape])
logger.info(['Unique training labels size:', unique_train_labels.shape])
# plt.imshow(train_images[0])
# plt.show()

#train_images = np.mean(train_images)
#test_images = np.mean(test_images)

# plt.imshow(train_images[0])
train_images = train_images[:,:,:,:].reshape((train_images.shape[0], train_images.shape[1]*train_images.shape[2], 3))
test_images = test_images[:,:,:,:].reshape((test_images.shape[0], test_images.shape[1]*test_images.shape[2], 3))

pca = [PCA(n_components=10), PCA(n_components=10), PCA(n_components=10)]

pca_train_images = []
pca_test_images = []

for i in range(3):
    pca[i].fit(train_images[:,:,i]) 
    pca_train_images.append(pca[i].transform(train_images[:,:,i])) 
    pca_test_images.append(pca[i].transform(test_images[:,:,i])) 

    logger.info([f'PCA{i} variance ratio: ', np.sum(pca[i].explained_variance_ratio_)])

estimators = []

for labelIdx in range(unique_train_labels.shape[0]):
    estimators.append([])
    label = unique_train_labels[labelIdx]

    for i in range(3):
        estimators[labelIdx].append(multivariate_normal(np.mean(pca_train_images[i][train_labels == label], axis=0), np.cov(np.transpose(pca_train_images[i][train_labels == label])), allow_singular=True))

estims = np.zeros((len(estimators), 3, test_images.shape[0]))

for estimator_idx in range(len(estimators)):
    for color_idx in range(3):
        estims[estimator_idx,color_idx] = estimators[estimator_idx][color_idx].pdf(pca_test_images[color_idx])

max_estims_idx = np.argmax(estims.sum(1),0)

max_estims = unique_test_labels[max_estims_idx]

print(np.count_nonzero(max_estims == test_labels)/test_images.shape[0])

ConfusionMatrixDisplay(confusion_matrix(test_labels, max_estims),display_labels=unique_test_labels).plot()

plt.show()