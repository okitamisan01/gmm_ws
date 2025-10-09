# gmm_ws

以下をクリック
` https://htmlpreview.github.io/?https://github.com/okitamisan01/gmm_ws/blob/main/artifacts/cluster_gallery.html `


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

テキストはWikipedia（日本語）から適当にコピペして集めてきます、とおもったけど想像以上にだめだったのでLLMにおまかせ

初期化（未実施なら）
python3 preprocess_text.py --init
生成（Geminiを明示）
export GEMINI_API_KEY=AIzaSyBetTy3ustseM_-gbaR0zX3HHBzve_xWXM
curl -s "https://generativelanguage.googleapis.com/v1/models?key=${GEMINI_API_KEY}" | jq '.'

python3 src/preprocess_text.py --auto-llm --llm-samples 5 \
  --chunk-mode time --chunk-seconds 30 --reading-cpm 500
読字速度（--reading-cpm）は好みに合わせて調整（450〜600あたり）。
句点優先で切るので、文の途中で切れにくいです。

JSON作成
python3 preprocess_text.py --build


######################

/home/su-laptop-02-okitani/gmm_ws/src/preprocess_image.py

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



###########

/home/su-laptop-02-okitani/gmm_ws/src/preprocess_audio.py

ESC50（音声）
メタデータをロードし、データセットの一覧を取得します。
wavデータをロードして波形データおよびメルスペクトログラムを描画
https://qiita.com/cvusk/items/61cdbce80785eaf28349

データは以下からダウンロードできます。
https://github.com/karoldvl/ESC-50

中身は音声データ（.wav）とメタデータ（.csv）です。
メタデータにはファイル名とクラス（0-49）、クラス名（上記テーブル参照）が入っています。

音声データ（.wav）はlibrosaというライブラリでロードすることでnumpy.arrayとして扱えるようになります。





＃＃＃＃＃＃＃＃＃
src/making_category.py

音と画像が両方存在しているのどれかな〜〜

# output

"""
[228, 'triceratops', 13, 'crickets', 0.631578947368421]
[20, 'brain-101', 10, 'rain', 0.6153846153846154]
[152, 'owl', 3, 'cow', 0.6666666666666666]
[58, 'doorknob', 30, 'door_wood_knock', 0.6086956521739131]
[251, 'airplanes-101', 47, 'airplane', 0.7619047619047619]
[89, 'goose', 1, 'rooster', 0.6666666666666666]
[113, 'hummingbird', 14, 'chirping_birds', 0.64]
[210, 'syringe', 28, 'snoring', 0.7142857142857143]
[102, 'helicopter-101', 40, 'helicopter', 0.8333333333333334]
[170, 'rainbow', 45, 'train', 0.6666666666666666]
[170, 'rainbow', 10, 'rain', 0.7272727272727273]
[56, 'dog', 0, 'dog', 1.0]
[7, 'bat', 5, 'cat', 0.6666666666666666]
[142, 'microwave', 9, 'crow', 0.6153846153846154]
[72, 'fire-truck', 48, 'fireworks', 0.631578947368421]
[245, 'windmill', 16, 'wind', 0.6666666666666666]
[43, 'coin', 24, 'coughing', 0.6666666666666666]
[158, 'penguin', 44, 'engine', 0.7692307692307693]
[133, 'lightning', 26, 'laughing', 0.7058823529411765]
[239, 'washing-machine', 35, 'washing_machine', 0.9333333333333333]
[80, 'frog', 4, 'frog', 1.0]
[220, 'toaster', 1, 'rooster', 0.7142857142857143]
[73, 'fireworks', 48, 'fireworks', 1.0]
[25, 'cactus', 5, 'cat', 0.6666666666666666]
[30, 'canoe', 34, 'can_opening', 0.625]
"""

# from the above, we can make following combinations
"""
chosen_oc_esc = {
    58:30, #[58, 'doorknob', 30, 'door_wood_knock', 0.6086956521739131]
    102:40, #[102, 'helicopter-101', 40, 'helicopter', 0.8333333333333334]
    239:35, #[239, 'washing-machine', 35, 'washing_machine', 0.9333333333333333] 
    245:16, #[245, 'windmill', 16, 'wind', 0.6666666666666666]
    113:14, #[113, 'hummingbird', 14, 'chirping_birds', 0.64]
    170:10, #[170, 'rainbow', 10, 'rain', 0.7272727272727273]
    89:1, #[89, 'goose', 1, 'rooster', 0.6666666666666666]
    73:48, #[73, 'fireworks', 48, 'fireworks', 1.0]
    251:47, #[251, 'airplanes-101', 47, 'airplane', 0.7619047619047619]
    56:0, #[56, 'dog', 0, 'dog', 1.0]
    80:4 #[80, 'frog', 4, 'frog', 1.0]
}
"""

自動収集用に、Wikipedia日本語版APIから各カテゴリの概要文を取得し、data_processed/text/<cid>-<name>/ にサンプルを自動保存する機能を preprocess_text.py に追加


## データセットを組み合わせる

データセットを組み合わせる
各カテゴリのデータは入手したので、次は組み合わせを作ります。
Caltech256、ESC-50、Wikipediaはまったく違う目的で作られているデータです。
理由のある組み合わせ方は考えつかないので、同一カテゴリのデータを適当にランダムに組み合わせます。
ここでオリジナリティのため、少し検証をします。
普通にマルチモーダルなデータを作るのではつまらないので、今回は一部のモードが存在しないデータを作ります。

フルデータ：python combine_dataset_full.py 
欠損あり：python combine_dataset_withlack.py 

## データの前処理
multimodal_feature_extraction.py

## モデルを定義する
## 学習する
## 評価する


すること
- preprocess_txt

python3 src/preprocess_text.py --init

export GEMINI_API_KEY="AIzaSyBetTy3ustseM_-gbaR0zX3HHBzve_xWXM"
export GEMINI_API_KEY="AIzaSyCodUkKhybTqMOdBdfw1tyGXiwlvv7F5Dk"

python3 src/preprocess_text.py --auto-llm --llm-samples 5 \
  --chunk-mode time --chunk-seconds 60 --reading-cpm 500

- preprocess_txt --build
python3 src/preprocess_text.py --build

- combine 2つ

フルデータ：python combine_dataset_full.py 
欠損あり：python combine_dataset_withlack.py 

- multimodal完成＋動かす

# 基本（各クラスタ1件ずつ）
python sample_concept.py --artifacts ~/gmm_ws/artifacts

# 特定クラスタだけ、3件ずつ
python sample_concept.py --artifacts ~/gmm_ws/artifacts --cluster 2 --per-cluster 3

# HTMLタイトル変更
python sample_concept.py --title "GMM Cluster Samples" 


結果：
file:///home/su-laptop-02-okitani/gmm_ws/artifacts/cluster_gallery.html
