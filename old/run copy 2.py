import numpy as np
from PIL import Image
import io
from matplotlib import pyplot as plt
from package.utils.logger import logger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
train_images_1 = train_images[:,:,:,0].reshape((train_images.shape[0], train_images.shape[1]*train_images.shape[2]))
train_images_2 = train_images[:,:,:,1].reshape((train_images.shape[0], train_images.shape[1]*train_images.shape[2]))
train_images_3 = train_images[:,:,:,2].reshape((train_images.shape[0], train_images.shape[1]*train_images.shape[2]))

test_images_1 = test_images[:,:,:,0].reshape((test_images.shape[0], test_images.shape[1]*test_images.shape[2]))
test_images_2 = test_images[:,:,:,1].reshape((test_images.shape[0], test_images.shape[1]*test_images.shape[2]))
test_images_3 = test_images[:,:,:,2].reshape((test_images.shape[0], test_images.shape[1]*test_images.shape[2]))

pca_1 = PCA(n_components=100)
pca_2 = PCA(n_components=100)
pca_3 = PCA(n_components=100)

PCA_train_images_1 = pca_1.fit_transform(train_images_1)
PCA_train_images_2 = pca_2.fit_transform(train_images_2)
PCA_train_images_3 = pca_3.fit_transform(train_images_3)


logger.info(['PCA1 variance ratio: ', np.sum(pca_1.explained_variance_ratio_)])
logger.info(['PCA2 variance ratio: ', np.sum(pca_2.explained_variance_ratio_)])
logger.info(['PCA3 variance ratio: ', np.sum(pca_3.explained_variance_ratio_)])

PCA_test_images_1 = pca_1.transform(test_images_1)
PCA_test_images_2 = pca_2.transform(test_images_2)
PCA_test_images_3 = pca_3.transform(test_images_3)

#tnse = TSNE(n_components=3)
#TNSE_train_images = tnse.fit_transform(train_images)

# fig = plt.figure()
# ax1 = fig.add_subplot(121, projection='3d')
#ax2 = fig.add_subplot(122, projection='3d')

# ax1.scatter(PCA_train_images[train_labels=='apple_pie', 0], PCA_train_images[train_labels=='apple_pie', 1], PCA_train_images[train_labels=='apple_pie', 2], c='g')
# ax1.scatter(PCA_train_images[train_labels=='cannoli', 0], PCA_train_images[train_labels=='cannoli', 1], PCA_train_images[train_labels=='cannoli', 2], c='r')
# ax1.scatter(PCA_train_images[train_labels=='edamame', 0], PCA_train_images[train_labels=='edamame', 1], PCA_train_images[train_labels=='edamame', 2], c='b')

# ax2.scatter(TNSE_train_images[train_labels=='apple_pie', 0], TNSE_train_images[train_labels=='apple_pie', 1], TNSE_train_images[train_labels=='apple_pie', 2], c='g')
# ax2.scatter(TNSE_train_images[train_labels=='cannoli', 0], TNSE_train_images[train_labels=='cannoli', 1], TNSE_train_images[train_labels=='cannoli', 2], c='r')
# ax2.scatter(TNSE_train_images[train_labels=='edamame', 0], TNSE_train_images[train_labels=='edamame', 1], TNSE_train_images[train_labels=='edamame', 2], c='b')

# plt.show()

svm_1 = SVC(kernel='poly', max_iter=100000).fit(PCA_train_images_1, train_labels)
svm_2 = SVC(kernel='poly', max_iter=100000).fit(PCA_train_images_2, train_labels)
svm_3 = SVC(kernel='poly', max_iter=100000).fit(PCA_train_images_3, train_labels)

preds_1 = svm_1.predict(PCA_test_images_1)
preds_2 = svm_2.predict(PCA_test_images_2)
preds_3 = svm_3.predict(PCA_test_images_3)

cm_1 = confusion_matrix(test_labels, preds_1, normalize='all')
cm_2 = confusion_matrix(test_labels, preds_2, normalize='all')
cm_3 = confusion_matrix(test_labels, preds_3, normalize='all')

fig, axes = plt.subplots(1,3)

ConfusionMatrixDisplay(cm_1,display_labels=svm_1.classes_).plot(ax=axes[0])
ConfusionMatrixDisplay(cm_2,display_labels=svm_2.classes_).plot(ax=axes[1])
ConfusionMatrixDisplay(cm_3,display_labels=svm_3.classes_).plot(ax=axes[2])

plt.show()


