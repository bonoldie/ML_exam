import numpy as np
import h5py
from package.utils.logger import logger
import matplotlib.pyplot as plt
import cv2

# Load dataset 

# 10099 (64x64) images, 1 channel
ds_110099_r64x64x1 = 'food_c101_n10099_r64x64x1'

ds = h5py.File(f'.ds/{ds_110099_r64x64x1}.h5', 'r')

logger.info(f'Datasets : ${ds.keys()}')

categories = ds['category']
categories_names = ds['category_names']
images = ds['images']

logger.debug(categories)
logger.debug(categories_names)
logger.debug(images)

categories = np.asarray(ds['category'])
categories_names = np.asarray(ds['category_names'])
images = np.asarray(ds['images'])

labels = np.argwhere(categories)[:,1]
categories_labels = categories_names[labels]

if False:
    images_to_plot = 10

    for i in range(images_to_plot): 
        plt.subplot(int( np.round(np.sqrt(images_to_plot))),int(np.ceil(np.sqrt(images_to_plot))), i+1)
        plt.imshow(images[i+100], cmap='gray')
        plt.title(categories_labels[i+100])

    plt.show()

img = np.interp(images[0],[-1,1], [0,255]).astype(int)

im_out = np.tile(img, [1,1,3])


img=cv2.cvtColor(im_out,cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

