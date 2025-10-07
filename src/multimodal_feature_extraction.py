import os
import random
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 前提変数（これらはすでに得ているものと仮定）
# dataset_test: pandas DataFrame（元データ, imgパス, sndパス, text欄などを含む）
# gmm_preds: 各テストサンプルのクラスタ所属ラベル（配列長 N）
# n_clusters = gmm.n_components

import os
import random
import numpy as np
import pandas as pd
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications import Xception
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from pathlib import Path

# ====================
# 1. データ読み込み
# ====================
# 現在のスクリプトがあるフォルダの一つ上のディレクトリを取得
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FULL = True
if FULL:
    dataset_train = pd.read_csv(os.path.join(base_dir, "dataset_full", "dataset_train_full.csv"), index_col=0)
    dataset_test  = pd.read_csv(os.path.join(base_dir, "dataset_full", "dataset_test_full.csv"), index_col=0)
else:
    dataset_train = pd.read_csv(os.path.join(base_dir, "dataset_missing", "dataset_train_missing.csv"), index_col=0)
    dataset_test  = pd.read_csv(os.path.join(base_dir, "dataset_missing", "dataset_test_missing.csv"), index_col=0)

x_test_img_path = dataset_test["img"].values
x_test_snd_path = dataset_test["snd"].values
x_test_text = dataset_test["text"].values
# y_test = to_categorical(dataset_test["target"].values)

# --- Image
x_test_img = np.zeros((len(x_test_img_path), 299, 299, 3))
for i, p in enumerate(x_test_img_path):
    if p is not None and isinstance(p, str) and os.path.exists(p):
        x_test_img[i] = np.load(p)["img"]
x_test_img = x_test_img.astype("float32") / 255.

# --- Sound
# mel スペクトログラム算出（preprocess_audio.py と同様の処理に合わせる）
def calculate_melsp(x, n_fft=1024, hop_length=128, sr=44100):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    melsp = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=128)
    melsp_db = librosa.power_to_db(melsp)
    return melsp_db

# 汎用ローダ: .npz の場合は melsp をそのまま読み込み、波形ファイルなら計算
def load_melsp_from_path(filepath):
    if filepath is None or not isinstance(filepath, str) or not os.path.exists(filepath):
        return None
    try:
        if filepath.lower().endswith(".npz"):
            d = np.load(filepath)
            # preprocess_audio.py で "melsp" キーで保存している想定
            if "melsp" in d:
                return d["melsp"]
            # フォールバック: 他のキー名の可能性
            for k in d.files:
                if k.lower().startswith("mel"):
                    return d[k]
            return None
        else:
            # wav 等のときは波形から mel を作成
            x, _sr = librosa.load(filepath, sr=44100)
            return calculate_melsp(x, sr=_sr)
    except Exception as e:
        print(f"[WARN] 音声読み込みに失敗: {filepath} ({e})")
        return None

# 全テスト音声の mel 長を調べて入力長を決定（パディング/切り詰め）
mel_list = []
time_max = 0
freq = 128
for p in x_test_snd_path:
    mel = load_melsp_from_path(p)
    mel_list.append(mel)
    if mel is not None:
        # mel shape: (freq, time)
        if mel.shape[0] != freq:
            # 周波数次元が異なる場合はスキップ/調整（ここではスキップして後段でゼロ詰め）
            pass
        time_max = max(time_max, mel.shape[1])

# 時間長が取得できなかった場合のデフォルト（元コードの値に合わせる）
if time_max <= 0:
    time_max = 1723

# 固定長テンソルに整形（右側ゼロパディング、長い場合は右側切り落とし）
x_test_snd = np.zeros((len(x_test_snd_path), freq, time_max), dtype=np.float32)
for i, mel in enumerate(mel_list):
    if mel is None:
        continue
    # 周波数次元の整合（不足時は切り詰め、超過時は先頭freqチャネル採用）
    mel_use = mel
    if mel_use.shape[0] != freq:
        if mel_use.shape[0] > freq:
            mel_use = mel_use[:freq, :]
        else:
            tmp = np.zeros((freq, mel_use.shape[1]), dtype=mel_use.dtype)
            tmp[:mel_use.shape[0], :] = mel_use
            mel_use = tmp
    # 時間次元の整合
    T = min(time_max, mel_use.shape[1])
    x_test_snd[i, :, :T] = mel_use[:, :T]

x_test_snd = x_test_snd.reshape(len(x_test_snd), freq, time_max, 1)

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

# 音声入力の形状を動的に設定
input_snd = Input(shape=(freq, time_max, 1), name="input_snd")
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
# 4. GMMクラスタリング（教師なし, 成分数はBICで自動選択）
# ====================
scaler = StandardScaler()
# 数値安定のため float64 で標準化
features_scaled = scaler.fit_transform(features_test.astype(np.float64))

n_samples = features_scaled.shape[0]
if n_samples < 2:
    raise RuntimeError("クラスタリングできるだけのサンプル数がありません。")

k_min = 2
k_max = max(k_min, min(20, n_samples - 1))

best = {"bic": np.inf, "gmm": None, "k": None, "cov": None, "reg": None}
for cov in ("diag", "spherical"):  # 安定寄り
    for reg in (1e-4, 1e-3, 1e-2):
        for k in range(k_min, k_max + 1):
            try:
                gm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov,
                    reg_covar=reg,
                    random_state=0,
                    init_params="kmeans",
                    max_iter=200,
                )
                gm.fit(features_scaled)
                bic = gm.bic(features_scaled)
                if bic < best["bic"]:
                    best = {"bic": bic, "gmm": gm, "k": k, "cov": cov, "reg": reg}
            except Exception:
                continue

if best["gmm"] is None:
    raise RuntimeError("GMM の学習に失敗しました（全設定でエラー）。")

gmm = best["gmm"]
gmm_preds = gmm.predict(features_scaled)
print(f"Selected GMM -> k={best['k']}, cov={best['cov']}, reg={best['reg']}, BIC={best['bic']:.2f}")

# 責務とCSVを保存
ARTIFACTS = Path(base_dir) / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
resp = gmm.predict_proba(features_scaled)
np.save(ARTIFACTS / "resp.npy", resp)

def abs_or_none(p):
    return os.path.abspath(p) if isinstance(p, str) and len(p) > 0 else p

dataset_to_save = dataset_test.copy()
for col in ["img", "snd"]:
    if col in dataset_to_save.columns:
        dataset_to_save[col] = dataset_to_save[col].apply(abs_or_none)
dataset_to_save.to_csv(ARTIFACTS / "dataset_test.csv")
print(f"Saved responsibilities and dataset CSV to: {ARTIFACTS}")

# ===== 可視化: クラスタサイズ分布 =====
try:
    counts = np.bincount(gmm_preds, minlength=gmm.n_components)
    plt.figure(figsize=(max(6, 0.8*len(counts)), 4))
    plt.bar(np.arange(len(counts)), counts, color="#4e79a7")
    plt.xlabel("Cluster ID")
    plt.ylabel("Count")
    plt.title("Cluster sizes")
    plt.tight_layout()
    out = ARTIFACTS / "cluster_sizes.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
except Exception as e:
    print(f"[WARN] Cluster size 可視化失敗: {e}")

# ===== Silhouette スコアと分布（教師なし評価）=====
try:
    sil_overall = silhouette_score(features_scaled, gmm_preds)
    sil_samples = silhouette_samples(features_scaled, gmm_preds)
    print(f"Silhouette score (overall): {sil_overall:.4f}")
    df_sil = pd.DataFrame({"cluster": gmm_preds, "sil": sil_samples})
    plt.figure(figsize=(max(6, 0.8*gmm.n_components), 4))
    sns.boxplot(data=df_sil, x="cluster", y="sil", color="#59a14f")
    plt.axhline(sil_overall, ls="--", c="red", lw=1, label=f"overall={sil_overall:.3f}")
    plt.legend()
    plt.title("Silhouette by cluster")
    plt.tight_layout()
    out = ARTIFACTS / "silhouette_box.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
except Exception as e:
    print(f"[WARN] Silhouette 可視化失敗: {e}")

# ===== t-SNE 2D 可視化（クラスタのみを色分け）=====
try:
    n = len(features_scaled)
    if n >= 5:
        perplexity = max(5, min(30, n // 10))
        tsne = TSNE(n_components=2, init="pca", random_state=0, perplexity=perplexity)
        emb = tsne.fit_transform(features_scaled)

        plt.figure(figsize=(6,5))
        plt.scatter(emb[:,0], emb[:,1], c=gmm_preds, cmap="tab20", s=18, alpha=0.9)
        plt.title("t-SNE colored by predicted cluster")
        plt.xticks([]); plt.yticks([])
        plt.tight_layout()
        out = ARTIFACTS / "tsne_pred.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close()
except Exception as e:
    print(f"[WARN] t-SNE 可視化失敗: {e}")
