import os
import random
import numpy as np
import pandas as pd
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications import Xception
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

# ====================
# 1. データ読み込み
# ====================
FULL = True
if FULL:
    dataset_train = pd.read_csv("../dataset_full/dataset_train.csv", index_col=0)
    dataset_test = pd.read_csv("../dataset_full/dataset_test.csv", index_col=0)
else:
    dataset_train = pd.read_csv("../dataset_missing/dataset_train.csv", index_col=0)
    dataset_test = pd.read_csv("../dataset_missing/dataset_test.csv", index_col=0)

x_test_img_path = dataset_test["img"].values
x_test_snd_path = dataset_test["snd"].values
x_test_text = dataset_test["text"].values
y_test = to_categorical(dataset_test["target"].values)

# --- Image
x_test_img = np.zeros((len(x_test_img_path), 299, 299, 3))
for i, p in enumerate(x_test_img_path):
    if p is not None:
        x_test_img[i] = np.load(p)["img"]
x_test_img = x_test_img.astype("float32") / 255.

# --- Sound
def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=128)
    return melsp

def load_wave_data(filepath):
    x, _ = librosa.load(filepath, sr=44100)
    return x, _

freq, time = 128, 1723
x_test_snd = np.zeros((len(x_test_snd_path), freq, time))
for i, p in enumerate(x_test_snd_path):
    if p is not None:
        s, _ = load_wave_data(p)
        x_test_snd[i] = calculate_melsp(s)
x_test_snd = x_test_snd.reshape(len(x_test_snd), freq, time, 1)

# --- Text
txt_length = 200
def convert_text_to_unicode(s, del_rate=0.001):
    return [ord(x) for x in str(s).strip() if random.random() > del_rate] if s != 0 else [0]

def reshape_text(s, max_length=200, del_rate=0.001):
    s_ = convert_text_to_unicode(s, del_rate)
    s_ = s_[:max_length] + [0] * (max_length - len(s_))
    return s_

x_test_text = np.array([reshape_text(t, max_length=txt_length, del_rate=0) for t in x_test_text])

# ====================
# 2. モデル構築（特徴抽出器）
# ====================
input_text = Input(shape=(txt_length,), name='input_text')
def clcnn(input_text):
    filter_sizes = (2,3,4,5)
    clx = Embedding(0xffff, 256)(input_text)
    convs = []
    for i, f in enumerate(filter_sizes):
        _x = Conv1D(256, f, strides=(f//2), padding="same")(clx)
        _x = Activation("tanh")(_x)
        _x = GlobalMaxPooling1D()(_x)
        convs.append(_x)
    x = Concatenate()(convs)
    x = Dense(1024, activation="selu", kernel_initializer="lecun_normal")(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation="selu", kernel_initializer="lecun_normal")(x)
    return x

input_img = Input(shape=(299, 299, 3), name="input_img")
def xception(input_img):
    cnn = Xception(input_tensor=input_img, include_top=False, weights='imagenet')
    x = cnn.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    return x

input_snd = Input(shape=(freq, time, 1), name="input_snd")
def snd_cnn(input_snd):
    def cba(x, f, k, s):
        x = Conv2D(f, k, strides=s, padding='same')(x)
        x = BatchNormalization()(x)
        return Activation("relu")(x)
    x_1 = cba(input_snd, 32, (1,8), (1,2))
    x_1 = cba(x_1, 32, (8,1), (2,1))
    x_1 = cba(x_1, 64, (1,8), (1,2))
    x_1 = cba(x_1, 64, (8,1), (2,1))
    x = GlobalAveragePooling2D()(x_1)
    x = Dense(256, activation='relu')(x)
    return x

# モデル構成
clx = clcnn(input_text)
xcp = xception(input_img)
snd = snd_cnn(input_snd)
merged = Concatenate()([clx, xcp, snd])
last_dense = Dense(256, activation='relu', name="last_dense")(merged)
model = Model(inputs=[input_text, input_img, input_snd], outputs=last_dense)

# ====================
# 3. 特徴ベクトル抽出
# ====================
features_test = model.predict([x_test_text, x_test_img, x_test_snd])

# ====================
# 4. GMMクラスタリング
# ====================
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_test)

gmm = GaussianMixture(n_components=y_test.shape[1], covariance_type='full', random_state=0)
gmm.fit(features_scaled)
gmm_preds = gmm.predict(features_scaled)

y_true = np.argmax(y_test, axis=1)
ari = adjusted_rand_score(y_true, gmm_preds)
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
