import zipfile
from scripts.dataset import downloadFromUrl
from pathlib import Path

ds_path = downloadFromUrl("https://www.kaggle.com/datasets/kmader/food41/download?datasetVersionNumber=5", "dataset.zip", ".ds")

with zipfile.ZipFile(ds_path, 'r') as zip_ref:
    zip_ref.extractall(Path(ds_path).parent)