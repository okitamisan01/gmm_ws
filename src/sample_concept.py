#!/usr/bin/env python3
import argparse, os, random, shutil
from pathlib import Path
import numpy as np
import pandas as pd

# 画像系
import matplotlib.pyplot as plt

# 音声系（wav保存に必要）
try:
    import librosa
    import soundfile as sf
except Exception:
    librosa = None
    sf = None

# =========================
# ユーティリティ
# =========================
WAV_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")

def ensure_librosa():
    if librosa is None or sf is None:
        raise RuntimeError(
            "音声の合成/保存に librosa と soundfile が必要です。\n"
            "  pip install librosa soundfile"
        )

def load_npz_image(npz_path):
    d = np.load(npz_path)
    if "img" in d:
        arr = d["img"]
    else:
        arr = None
        for k in d.files:
            if k.lower().startswith("img"):
                arr = d[k]; break
        if arr is None:
            raise KeyError("NPZに画像キーがありません（例: 'img'）")

    # 正規化戻し
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    # CHW→HWC の可能性に対応
    if arr.ndim == 3 and arr.shape[0] in (1,3) and arr.shape[2] not in (1,3):
        arr = np.transpose(arr, (1,2,0))
    return arr

def save_image_png(arr, out_png):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(3.6,3.6))
    if arr.ndim == 2:
        plt.imshow(arr, cmap="gray")
    else:
        plt.imshow(arr)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, dpi=160)
    plt.close()
    return str(out_png)

def load_npz_mel(npz_path):
    d = np.load(npz_path)
    if "melsp" in d:
        return d["melsp"]
    for k in d.files:
        if k.lower().startswith("mel"):
            return d[k]
    raise KeyError("NPZにメルスペクトログラムがありません（例: 'melsp'）")

def synthesize_wav_from_mel(mel_db,
                            sr=44100, n_fft=1024, hop_length=128, n_mels=128):
    """mel(dB)→近似波形（Griffin–Lim）。元波形の厳密復元ではありません。"""
    ensure_librosa()
    # dB→power
    mel_power = librosa.db_to_power(mel_db)
    # mel→波形（Griffin–Lim 内部で反復）。n_mels は入力から自動推定されるため渡さない
    y = librosa.feature.inverse.mel_to_audio(
        mel_power, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=48
    )
    return y, sr

def save_wav(y, sr, out_wav):
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, y, sr)
    return str(out_wav)

def copy_if_small(src, dst):
    """wav等が既にある場合、 artifacts配下へコピー（同名重複は上書き）"""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)

def html_escape(s: str) -> str:
    return (str(s)
            .replace("&","&amp;")
            .replace("<","&lt;")
            .replace(">","&gt;")
            .replace('"',"&quot;")
            .replace("'","&#39;"))

# =========================
# メイン
# =========================
def main():
    ap = argparse.ArgumentParser(
        description="画像+テキスト+（クリックで）音声を同一画面に並べるギャラリーHTMLを生成"
    )
    ap.add_argument("--artifacts", default=str(Path.home() / "gmm_ws" / "artifacts"),
                    help="resp.npy と dataset_test.csv のあるディレクトリ（出力もここに作る）")
    ap.add_argument("--cluster", type=int, default=None,
                    help="特定クラスタのみ表示（省略時は全クラスタから1件ずつ）")
    ap.add_argument("--per-cluster", type=int, default=1,
                    help="各クラスタから何件表示するか（デフォルト1）")
    ap.add_argument("--seed", type=int, default=0, help="乱数シード")
    ap.add_argument("--title", default="Cluster Gallery", help="HTMLタイトル")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    artifacts = Path(args.artifacts).expanduser().resolve()
    resp_path = artifacts / "resp.npy"
    csv_path  = artifacts / "dataset_test.csv"
    out_dir   = artifacts / "gallery_assets"
    html_path = artifacts / "cluster_gallery.html"

    if not resp_path.exists() or not csv_path.exists():
        raise FileNotFoundError(f"必要ファイルが見つかりません: {resp_path} / {csv_path}")

    resp = np.load(resp_path)                    # (N, K)
    df = pd.read_csv(csv_path, index_col=0)      # 必須列: img, snd, text
    if len(df) != resp.shape[0]:
        print(f"[WARN] dataset行数({len(df)}) と resp件数({resp.shape[0]}) が不一致。小さい方に合わせます。")
        N = min(len(df), resp.shape[0])
        df = df.iloc[:N].copy()
        resp = resp[:N]

    clusters = np.argmax(resp, axis=1)
    n_clusters = resp.shape[1]

    # 抜き出し対象のインデックス集合を作る
    targets = []
    if args.cluster is None:
        for c in range(n_clusters):
            idxs = np.where(clusters == c)[0]
            if len(idxs) == 0: 
                continue
            # ランダムに per-cluster 件
            take = min(args.per_cluster, len(idxs))
            chosen = random.sample(list(idxs), take)
            for i in chosen:
                targets.append((c, i))
    else:
        c = int(args.cluster)
        if not (0 <= c < n_clusters):
            raise ValueError(f"cluster_id は 0〜{n_clusters-1}")
        idxs = np.where(clusters == c)[0]
        if len(idxs) == 0:
            print(f"cluster {c}: サンプルなし")
        else:
            take = min(args.per_cluster, len(idxs))
            chosen = random.sample(list(idxs), take)
            for i in chosen:
                targets.append((c, i))

    # 各サンプルのアセット（PNG/WAV）を artifacts/gallery_assets/ に保存し、HTMLカードを作る
    cards_html = []
    for (c, i) in targets:
        row = df.iloc[i]
        # ---------- 画像 ----------
        img_disp_path = None
        img_src = row.get("img", None)
        try:
            if isinstance(img_src, str) and os.path.exists(img_src):
                if img_src.lower().endswith(".npz"):
                    arr = load_npz_image(img_src)
                    img_disp_path = out_dir / f"img_c{c}_i{i}.png"
                    save_image_png(arr, img_disp_path)
                    img_disp_path = img_disp_path.name  # 相対指定
                else:
                    # 画像ファイルならそのままコピー
                    img_disp_path = out_dir / f"img_c{c}_i{i}{Path(img_src).suffix}"
                    copy_if_small(img_src, img_disp_path)
                    img_disp_path = img_disp_path.name
        except Exception as e:
            print(f"[WARN] 画像処理失敗 c={c} i={i}: {e}")

        # ---------- 音声 ----------
        audio_disp_path = None
        snd_src = row.get("snd", None)
        try:
            if isinstance(snd_src, str) and os.path.exists(snd_src):
                if snd_src.lower().endswith(WAV_EXTS):
                    # 既存の音源をコピー
                    audio_disp_path = out_dir / f"aud_c{c}_i{i}{Path(snd_src).suffix}"
                    copy_if_small(snd_src, audio_disp_path)
                    audio_disp_path = audio_disp_path.name
                elif snd_src.lower().endswith(".npz"):
                    # メルから近似合成
                    mel = load_npz_mel(snd_src)
                    y, sr = synthesize_wav_from_mel(mel, sr=44100, n_fft=1024, hop_length=128, n_mels=128)
                    audio_disp_path = out_dir / f"aud_c{c}_i{i}.wav"
                    save_wav(y, sr, audio_disp_path)
                    audio_disp_path = audio_disp_path.name
        except Exception as e:
            print(f"[WARN] 音声処理失敗 c={c} i={i}: {e}")

        # ---------- テキスト ----------
        text = row.get("text", "")
        text_safe = html_escape(text)

        # ---------- HTMLカード ----------
        card = f"""
        <div class="card">
          <div class="tag">cluster {c} · idx {i}</div>
          <div class="img">{('<img src="gallery_assets/'+img_disp_path+'" alt="image">') if img_disp_path else '<div class="noimg">no image</div>'}</div>
          <div class="text">{text_safe if text_safe else '<span class="muted">[no text]</span>'}</div>
          <div class="audio">{('<audio controls src="gallery_assets/'+audio_disp_path+'"></audio>') if audio_disp_path else '<div class="muted">[no audio]</div>'}</div>
        </div>
        """
        cards_html.append(card)

    # =========================
    # HTML 生成
    # =========================
    out_dir.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{html_escape(args.title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{
  --bg: #0b0d10;
  --card: #14181d;
  --fg: #e9eef5;
  --muted: #9aa7b4;
  --accent: #4da3ff;
  --border: #21262d;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 24px;
  font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Hiragino Kaku Gothic ProN', 'Noto Sans JP', sans-serif;
  background: var(--bg); color: var(--fg);
}}
h1 {{ margin: 0 0 16px; font-size: 20px; font-weight: 600; }}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
}}
.card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 12px;
  display: flex; flex-direction: column; gap: 8px;
  box-shadow: 0 2px 10px rgba(0,0,0,.25);
}}
.card .tag {{
  font-size: 12px; color: var(--muted);
}}
.card .img {{
  width: 100%; aspect-ratio: 1/1; 
  background: #0f1318; border: 1px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  border-radius: 12px; overflow: hidden;
}}
.card .img img {{ width: 100%; height: 100%; object-fit: contain; }}
.card .noimg {{ color: var(--muted); font-size: 12px; }}
.card .text {{ font-size: 14px; line-height: 1.5; }}
.card .muted {{ color: var(--muted); }}
.card .audio audio {{ width: 100%; }}
footer {{ margin-top: 16px; color: var(--muted); font-size: 12px; }}
.note {{ color: var(--muted); font-size: 13px; margin: 8px 0 16px; }}
</style>
</head>
<body>
<h1>{html_escape(args.title)}</h1>
<div class="note">クリックで音声を再生できます。NPZのメルしか無い場合は近似再生（Griffin–Lim）です。</div>
<div class="grid">
{''.join(cards_html) if cards_html else '<p>表示対象がありません。</p>'}
</div>
<footer>Generated in {html_escape(str(artifacts))}</footer>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    print(f"[DONE] ギャラリーを出力しました → {html_path}")
    print("ローカルで開く： file://" + str(html_path))

if __name__ == "__main__":
    main()
