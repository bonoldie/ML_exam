import os
import sys
import requests
from scripts.utils.logger import logger

def downloadFromUrl(url, filename, ds_dir = ".ds", force = False):
    ds_dir = os.getcwd() + f"/{ds_dir}"
    ds_path = os.path.join(ds_dir, filename)

    if os.path.exists(ds_path) and not force:
        logger.info('Dataset file is downloaded already')
        return
    
    
    if(os.path.exists(ds_dir) == False):
        os.makedirs(ds_dir)
    
    logger.info('Dataset is downloading...please wait')
    
    req = requests.get(url, allow_redirects=True)

    logger.info(f"Dataset downloaded, available at {ds_dir}")

    open(ds_path, 'wb').write(req.content)

    return ds_path
    

def printDatasetInfo(ds):
    pass

