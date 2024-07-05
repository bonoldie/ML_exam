import numpy as np
import scipy
import scipy.linalg 
from package.utils.logger import logger

class PCA:
    d = 6

    eigen_vectors = []
    eigen_values = []
    
    def __init__(self, data, d = 6):

        if not isinstance(data, np.ndarray):
            logger.error('PCA data must be a NxM matrix(N records with M features)')
            raise Exception('PCA data must be a NxM matrix(N records with M features)')    

        self.data = data - np.mean(data)
        self.d = d

        logger.debug('Calculating covariance matrix')
        cov = np.cov(data.T)

        logger.debug('Eigenstuff extraction')
        self.eigen_values, self.eigen_vectors = scipy.linalg.eigh(cov)

        sortIdxs = np.argsort(self.eigen_values)[::-1]

        self.eigen_vectors = self.eigen_vectors[:, sortIdxs]
        self.eigen_values = self.eigen_values[sortIdxs]

    def extract_features(self, data):
        # TODO: check data shape
        return data @ self.eigen_vectors[:, :self.d]
    
