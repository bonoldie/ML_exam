import zipfile
from package.utils.dataset import downloadFromUrl
from package.utils import logger
from pathlib import Path

ds_path = downloadFromUrl("https://www.kaggle.com/datasets/kmader/food41/download?datasetVersionNumber=5", "archive.zip", ".ds")

with zipfile.ZipFile(ds_path, 'r') as zip_ref:
    zip_ref.extractall(Path(ds_path).parent)
    logger.get().info('Dataset extracted')
