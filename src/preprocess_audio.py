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

# ESC-50 データセットの音声ファイルを一括で前処理し、メルスペクトログラムを .npz で保存する

# define directories
base_dir = "./"
esc_dir = os.path.join(base_dir, "data_raw", "ESC-50")
meta_file = os.path.join(esc_dir, "meta/esc50.csv")
audio_dir = os.path.join(esc_dir, "audio/")

# 出力先ディレクトリ
output_dir = os.path.join(base_dir, "data_processed", "audio")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# load metadata
meta_data = pd.read_csv(meta_file)

# get data size
data_size = meta_data.shape
print(data_size)

# arrange target label and its name
class_dict = {}
for i in range(data_size[0]):
    if meta_data.loc[i,"target"] not in class_dict.keys():
        class_dict[meta_data.loc[i,"target"]] = meta_data.loc[i,"category"]

# load a wave data
def load_wave_data(audio_dir, file_name):
    file_path = os.path.join(audio_dir, file_name)
    x, fs = librosa.load(file_path, sr=44100)
    return x,fs

# change wave data to mel-stft
def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    melsp = librosa.feature.melspectrogram(S=stft, sr=44100, n_mels=128)
    melsp_db = librosa.power_to_db(melsp)
    return melsp_db

# display wave in plots
def show_wave(x):
    plt.plot(x)
    plt.show()

# display wave in heatmap
def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs)
    plt.colorbar()
    plt.show()

# 全ファイルを処理
for idx, row in meta_data.iterrows():
    filename = row["filename"]
    label = row["target"]
    file_path = os.path.join(audio_dir, filename)
    try:
        x, _ = librosa.load(file_path, sr=44100)
        melsp = calculate_melsp(x)
        out_path = os.path.join(output_dir, filename.replace(".wav", ".npz"))
        np.savez(out_path, melsp=melsp, label=label)
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# example data
x, fs = load_wave_data(audio_dir, meta_data.loc[0,"filename"])
melsp = calculate_melsp(x)
print("wave size:{0}\nmelsp size:{1}\nsamping rate:{2}".format(x.shape, melsp.shape, fs))

