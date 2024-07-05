#!/usr/bin/env python
import os
import requests
from pathlib import Path

def downloadFromUrl(url, filename, ds_dir = ".ds", force = False):
    ds_dir = os.getcwd() + f"/{ds_dir}"
    ds_path = os.path.join(ds_dir, filename)

    if os.path.exists(ds_path) and not force:
        print('Dataset file available')
        return ds_path
    
    
    if(os.path.exists(ds_dir) == False):
        os.makedirs(ds_dir)
    
    print('Dataset is downloading...please wait')
    
    req = requests.get(url, allow_redirects=True)

    print(f"Dataset downloaded, available at {ds_dir}")

    open(ds_path, 'wb').write(req.content)

    return ds_path

ds_path = downloadFromUrl("https://www.kaggle.com/datasets/msarmi9/food101tiny/download?datasetVersionNumber=1", "dataset.zip", ".ds.tiny")
