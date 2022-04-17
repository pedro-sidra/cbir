import base64
from IPython.display import display, HTML
from IPython.core.display import HTML
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

    if Path(folder).exists():
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

        db.append(dict(clas=className.lower(), num=num, image=cv2.imread(str(file), cv2.IMREAD_GRAYSCALE), filepath=file))
    
    return pd.DataFrame(data=db).reset_index(drop=True)

def image_to_html(img:np.array) -> str:
    """Convert a numpy image array to an html <img/> tag"""
    retval, buffer = cv2.imencode('.png', img)
    b64 = base64.b64encode(buffer).decode()
    return '<img src="'+ "data:image/png;base64, "+ b64 + '" width="100" >'

def render_df(df):
    """Render df as a table and map images to <img/> tags"""
    return HTML(df.to_html(escape=False,formatters=dict(image=image_to_html)))

def displayImages(df):
    """Assume that the whole df contains only image and render it as an html table"""
    return display(HTML(df.to_html(escape=False,formatters={c:image_to_html for c in df.columns})))

def displayDf(df):
    """Render the df as an html table only with class, number and image"""
    display(render_df(df[["clas","num","image"]]))

def get_largest_contour(im):
    contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cmax = sorted(contours, key=cv2.contourArea)[-1]
    return cmax