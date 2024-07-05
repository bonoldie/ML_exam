import numpy as np
from package.utils.logger import logger

class KNN:
    data = np.array()
    labels = np.array()
    k = 5

    def __init__(self, train_data, train_labels, k = 5):
        if isinstance(train_data, list):
            self.data = np.array(train_data) 

        if not isinstance(train_labels, list):
            self.labels = np.array(train_labels) 
    
        self.k = k

    def predict(self, test_data):
        distances = [[np.linalg.norm(record - record) for record in self.data] for test_record in test_data]

        print(distances)
        pass