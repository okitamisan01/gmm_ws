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
y_test = to_categorical(dataset_test["target"].values)

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
# 4. GMMクラスタリング
# ====================
scaler = StandardScaler()
# 数値安定のため float64 で標準化
features_scaled = scaler.fit_transform(features_test.astype(np.float64))

# サンプル数より多い成分数は不安定になるため抑制
orig_components = y_test.shape[1]
n_samples = features_scaled.shape[0]
n_components = min(orig_components, max(2, n_samples - 1))

# 安定化のためのフォールバック付きフィッタ
last_err = None
for cov in ("diag", "spherical"):
    for reg in (1e-4, 1e-3, 1e-2):
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=cov,
                reg_covar=reg,
                random_state=0,
                init_params="kmeans",
                max_iter=200,
            )
            gmm.fit(features_scaled)
            last_err = None
            break
        except Exception as e:
            last_err = e
    if last_err is None:
        break

if last_err is not None:
    raise last_err

gmm_preds = gmm.predict(features_scaled)

y_true = np.argmax(y_test, axis=1)
ari = adjusted_rand_score(y_true, gmm_preds)
print(f"Adjusted Rand Index (ARI): {ari:.4f}")

# ===== 可視化: 真のラベル vs 予測クラスタの対応（行正規化） =====
try:
    ct = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(gmm_preds, name="cluster"), normalize="index")
    plt.figure(figsize=(1.2*ct.shape[1]+3, 1.2*ct.shape[0]+3))
    sns.heatmap(ct, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f"Contingency (row-normalized)\nARI={ari:.3f}")
    plt.xlabel("Predicted cluster")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"[WARN] Heatmap 可視化失敗: {e}")

# ===== 可視化: クラスタサイズ分布 =====
try:
    counts = np.bincount(gmm_preds, minlength=gmm.n_components)
    plt.figure(figsize=(max(6, 0.8*len(counts)), 4))
    plt.bar(np.arange(len(counts)), counts, color="#4e79a7")
    plt.xlabel("Cluster ID")
    plt.ylabel("Count")
    plt.title("Cluster sizes")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"[WARN] Cluster size 可視化失敗: {e}")

# ===== Silhouette スコアと分布 =====
try:
    sil_overall = silhouette_score(features_scaled, gmm_preds)
    sil_samples = silhouette_samples(features_scaled, gmm_preds)
    print(f"Silhouette score (overall): {sil_overall:.4f}")
    df_sil = pd.DataFrame({"cluster": gmm_preds, "sil": sil_samples})
    plt.figure(figsize=(max(6, 0.8*len(counts)), 4))
    sns.boxplot(data=df_sil, x="cluster", y="sil", color="#59a14f")
    plt.axhline(sil_overall, ls="--", c="red", lw=1, label=f"overall={sil_overall:.3f}")
    plt.legend()
    plt.title("Silhouette by cluster")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"[WARN] Silhouette 可視化失敗: {e}")

# ===== t-SNE 2D 可視化（重い場合あり） =====
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
        plt.tight_layout(); plt.show()

        plt.figure(figsize=(6,5))
        plt.scatter(emb[:,0], emb[:,1], c=y_true, cmap="tab20", s=18, alpha=0.9)
        plt.title("t-SNE colored by true class")
        plt.xticks([]); plt.yticks([])
        plt.tight_layout(); plt.show()
except Exception as e:
    print(f"[WARN] t-SNE 可視化失敗: {e}")

# ====================
# 5. 結果の確認
# ====================

n_clusters = gmm.n_components

def display_sample_for_cluster(cluster_id):
    # cluster に属するインデックスを取得
    idxs = np.where(gmm_preds == cluster_id)[0]
    if len(idxs) == 0:
        print(f"クラスタ {cluster_id} に属するサンプルなし")
        return
    # ランダムに1つ選択
    sel = random.choice(idxs)
    print(f"クラスタ {cluster_id} — 選択サンプル index: {sel}")
    
    # 画像表示
    img_path = dataset_test.iloc[sel]["img"]
    if img_path is not None and os.path.exists(img_path):
        arr = np.load(img_path)["img"]  # .npz に保存している前提
        # 正規化など逆変換があればここでやる
        plt.figure(figsize=(4,4))
        plt.imshow(arr.astype(np.uint8))
        plt.axis("off")
        plt.title(f"クラスタ{cluster_id} の画像 (idx={sel})")
        plt.show()
    else:
        print("画像データなし")
    
    # 音声: .wav なら再生、.npz ならメルスペクトログラムを可視化
    snd_path = dataset_test.iloc[sel]["snd"]
    if snd_path is not None and os.path.exists(snd_path):
        if snd_path.lower().endswith(".npz"):
            try:
                mel = np.load(snd_path)["melsp"]
                plt.figure(figsize=(5, 3))
                plt.imshow(mel, origin="lower", aspect="auto", cmap="magma")
                plt.colorbar(label="dB")
                plt.title("Mel spectrogram (from .npz)")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"音声メルスペクトログラム表示失敗: {e}")
        else:
            display_audio = ipd.Audio(snd_path)
            print("🔊 音声再生：")
            ipd.display(display_audio)
    else:
        print("音声データなし")
    
    # テキスト出力
    text = dataset_test.iloc[sel]["text"]
    print("📝 テキスト:", repr(text))
    print()  # 改行

# 全クラスタに対して表示する
for c in range(n_clusters):
    display_sample_for_cluster(c)
