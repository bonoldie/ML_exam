import numpy as np
from scipy.sparse.linalg import eigs
from scripts.utils.logger import logger

class PCA:
    data = np.array()
    d = 6

    eigen_vectors = []
    eigen_values = []
    
    def __init__(self, data, d = 6):

        if not isinstance(data, np.ndarray):
            logger.error('PCA data must be a NxM matrix(N records with M features)')
            raise Exception('PCA data must be a NxM matrix(N records with M features)')    

        self.data = data
        self.d = d

        cov = np.cov(data)
        self.eigen_values, self.eigen_vectors = eigs(cov, k=d)

        print(self.eigen_values)

    def extract():
        return
    
