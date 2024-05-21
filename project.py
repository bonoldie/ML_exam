#from scripts.dataset import downloadFromUrl
import numpy as np
from scripts.utils.logger import logger
import h5py

datasetH5File = h5py.File('.ds/food_c101_n10099_r32x32x1.h5', 'r')

print(datasetH5File)


