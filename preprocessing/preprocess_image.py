
import os
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import model_selection
from sklearn import preprocessing
import IPython.display as ipd
from io import BytesIO
from PIL import Image
import cv2
import re
import unicodedata
import librosa
import librosa.display
from keras import utils 

# define directories
base_dir = os.getcwd()
oc256_dir = os.path.join(base_dir, "data_raw","Caltech256", "256_ObjectCategories")
oc256_dirs = [os.path.join(oc256_dir, x) for x in os.listdir(oc256_dir)]
oc256_files = [[os.path.join(x,y) for y in os.listdir(x)] for x in oc256_dirs]
# make a dict of label num to category name
oc_class_dict = {int(x.split("/")[-1].split(".")[0]): x.split("/")[-1].split(".")[1] for x in oc256_dirs}

def convert2dgray_to_3dgray(img_array):
    return np.array([[[y,y,y] for y in x] for x in img_array])

# 入力: 2次元のグレースケール画像（形状が (高さ, 幅) の numpy.array）
# 出力: 3チャンネルのグレースケール画像（形状が (高さ, 幅, 3)）
# 処理の中身は、各ピクセルの値 y を [y, y, y] にコピーして「R=G=B」の3チャンネルにしている。
# つまり モノクロ1チャンネルを擬似的にカラー画像っぽくする

def resize_image_array(img_array, image_size=(299, 299)):
    if img_array is None:
        return None
    return cv2.resize(img_array, image_size)

# save image data in npz
def save_np_256_oc_data(x, data_type="train"):
    data_dir = "256_" + data_type
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for _,c in enumerate(x):
        for _,f in enumerate(c):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                print("スキップ:", f)
                continue
            image_name = os.path.splitext(os.path.basename(f))[0]
            image_path = os.path.join(data_dir, image_name)
            print("image_path:", image_path)
            print("image_name:", image_name)
            if not os.path.exists(image_path):
                y = f.split("/")[2].split(".")[0]
                img = Image.open(os.path.join(f))
                img = np.asarray(img)
                if len(img.shape) == 2:
                    img = convert2dgray_to_3dgray(img)
                img = resize_image_array(img, image_size=(299, 299))
                np.savez(image_path, img=img, y=y)

save_np_256_oc_data(oc256_files,  data_type="all")

