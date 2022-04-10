import requests
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import tarfile
import os
import re
import numpy as np
import cv2


def download_tar(db_url):
    db_path = Path(db_url)
    tar = db_path.name
    folder = db_path.name.split('.')[0]

    if Path(folder).exists:
        print(f"{folder} exists. Skipping")
        return folder

    print(f"Downloading {db_url} to {db_path.name}")

    response = requests.get(db_url, stream=True)

    if response.status_code == 200:
        with open(db_path.name, "wb") as f:
            f.write(response.raw.read())

    print(f"Extracting {tar} to {folder}")

    f = tarfile.open(tar)
    f.extractall(folder)
    f.close()

    print(f"Removing {tar}")
    os.remove(tar)

    return folder

def load_db(db_folder):
    fileClassExpr = re.compile(r"^(\D+)(\d+)")

    db = []
    for file in Path(db_folder).glob("**/*.pgm"):
        className, num = fileClassExpr.match(file.stem).groups()

        db.append(dict(clas=className.lower(), num=num, image=cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)))
    
    return pd.DataFrame(data=db)