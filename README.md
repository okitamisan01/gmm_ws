# gmm_ws

## 説明

参考サイト：https://qiita.com/cvusk/items/bdf51354c171631b554c
＋
GMMでマルチモーダル概念モデル作成

## 手順
- データセットを入手する
- データセットを組み合わせる
- データの前処理（ここまではqiitaにお世話になる）
- モデルを定義する
- 学習する
- 評価する


## データセットを入手する

画像：Caltech256
https://www.kaggle.com/datasets/jessicali9530/caltech256
Caltech256は名前のとおり、256種類（ラベル）の画像データが揃っています。
各ラベルの画像数はまちまちで、数十から数百枚ずつあります。

音声：ESC-50
https://github.com/karolpiczak/ESC-50
中身は音声データ（.wav）とメタデータ（.csv）です。
メタデータにはファイル名とクラス（0-49）、クラス名（上記テーブル参照）が入っています。
音声データ（.wav）はlibrosaというライブラリでロードすることでnumpy.arrayとして扱えるようになります。

テキストはWikipedia（日本語）から適当にコピペして集めてきます

こういう感じで配置しているよ：
su-laptop-02-okitani@su-laptop-02:~/gmm_ws$ ls data_raw/
Caltech256  ESC-50
su-laptop-02-okitani@su-laptop-02:~/gmm_ws$ ls data_raw/Caltech256/
256_ObjectCategories  256_objectcategories
su-laptop-02-okitani@su-laptop-02:~/gmm_ws$ 

## データセットを組み合わせる

データセットを組み合わせる
各カテゴリのデータは入手したので、次は組み合わせを作ります。
Caltech256、ESC-50、Wikipediaはまったく違う目的で作られているデータです。
理由のある組み合わせ方は考えつかないので、同一カテゴリのデータを適当にランダムに組み合わせます。
ここでオリジナリティのため、少し検証をします。
普通にマルチモーダルなデータを作るのではつまらないので、今回は一部のモードが存在しないデータを作ります。


## データの前処理

Caltech256(画像)
ダウンロードして解凍した.jpgデータをnumpy.arrayに変換して.npzで保存します。
https://qiita.com/cvusk/items/bdf51354c171631b554c

.npzとは？
＝＞Numpyの圧縮形式で、複数の配列をまとめて保存できる。

```
# 例: 複数の配列
X = np.random.rand(100, 32, 32, 3)   # 画像データ
y = np.random.randint(0, 256, 100)   # ラベル
meta = np.array(["cat", "dog", "car"])  # クラス名とか
# 複数の配列をまとめて保存
np.savez("dataset.npz", images=X, labels=y, classes=meta)
```

1. データ準備
必要なライブラリをインポート
データセット（Caltech256）のディレクトリ構造を取得
クラス番号とカテゴリ名の辞書を作成
2. 画像処理関数の定義
convert2dgray_to_3dgray: 2次元グレースケール画像を3チャンネル画像に変換
resize_image_array: 画像を指定サイズ（299x299）にリサイズ
3. データ保存処理
save_np_256_oc_data:
各画像ファイルを読み込み
グレースケールなら3チャンネル化
リサイズ
.npz形式で保存（画像データとラベル）
4. 実行
すべての画像に対して保存処理を実行

ESC50（音声）
メタデータをロードし、データセットの一覧を取得します。
wavデータをロードして波形データおよびメルスペクトログラムを描画
https://qiita.com/cvusk/items/61cdbce80785eaf28349

データは以下からダウンロードできます。
https://github.com/karoldvl/ESC-50

中身は音声データ（.wav）とメタデータ（.csv）です。
メタデータにはファイル名とクラス（0-49）、クラス名（上記テーブル参照）が入っています。

音声データ（.wav）はlibrosaというライブラリでロードすることでnumpy.arrayとして扱えるようになります。





## モデルを定義する
## 学習する
## 評価する