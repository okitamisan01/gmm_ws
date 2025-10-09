#!/usr/bin/env python3
import argparse, os, random, shutil
from pathlib import Path
import numpy as np
import pandas as pd

# 画像系
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

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
    ap.add_argument("--mosaic-max-width", type=int, default=640,
                    help="モザイク画像の最大表示幅(px)。画面いっぱいにならないように制限します。")
    ap.add_argument("--tsne-max-width", type=int, default=720,
                    help="t-SNE画像の最大表示幅(px)。画面いっぱいにならないように制限します。")
    ap.add_argument("--text-wall-count", type=int, default=40,
                    help="クラスタごとに上位から並べるテキスト数")
    ap.add_argument("--audio-wall-count", type=int, default=12,
                    help="クラスタごとに上位から並べる音声数（npzはwav復元）")
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
    # CSVは先頭列に category があり、これをインデックスにせず列として保持する
    df = pd.read_csv(csv_path)      # 必須列: category, target, img, snd, text
    # pandasが保存時に付与した自動インデックス列があれば除去
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"]) 
    if len(df) != resp.shape[0]:
        print(f"[WARN] dataset行数({len(df)}) と resp件数({resp.shape[0]}) が不一致。小さい方に合わせます。")
        N = min(len(df), resp.shape[0])
        df = df.iloc[:N].copy()
        resp = resp[:N]

    clusters = np.argmax(resp, axis=1)
    n_clusters = resp.shape[1]
    counts = np.bincount(clusters, minlength=n_clusters)

    # t-SNE と同じ tab20 カラーマップでクラスタ色を定義
    cmap = plt.cm.get_cmap('tab20', n_clusters)
    palette = [to_hex(cmap(i)) for i in range(n_clusters)]

    # クラスタ凡例（表）をHTMLとして作成
    legend_rows = []
    for i in range(n_clusters):
        col = palette[i]
        cnt = int(counts[i])
        legend_rows.append(
            f"<tr><td>Cluster {i}</td><td><span class=\"swatch\" style=\"background:{col}\"></span>{col}</td><td class=\"num\">{cnt}</td></tr>"
        )
    legend_html = (
        "<div class=\"legend\">"
        "<table class=\"cl-legend\">"
        "<thead><tr><th>クラスタ</th><th>色</th><th>件数</th></tr></thead>"
        f"<tbody>{''.join(legend_rows)}</tbody>"
        "</table>"
        "</div>"
    )

    # 既存の t-SNE 画像があればヘッダーに載せる
    tsne_img = None
    for name in ("tsne_pred.png", "tsne_true.png"):
        p = artifacts / name
        if p.exists():
            tsne_img = p.name  # 後で同フォルダにコピー不要（同一dir参照）
            break

    # 画像モザイク作成
    def build_cluster_mosaic(c, top=16):
        try:
            idxs = np.where(clusters == c)[0]
            if len(idxs) == 0:
                return None
            # 責務の高い順に並べる
            idxs_sorted = idxs[np.argsort(resp[idxs, c])[::-1]]
            pick = idxs_sorted[:min(top, len(idxs_sorted))]
            # 画像読み込み + メタ(カテゴリ/確率)
            imgs = []
            metas = []  # [(cat_str, p_float), ...]
            for i in pick:
                p_img = df.iloc[i]["img"]
                try:
                    if isinstance(p_img, str) and os.path.exists(p_img):
                        if p_img.lower().endswith('.npz'):
                            arr = load_npz_image(p_img)
                        else:
                            arr = plt.imread(p_img)
                        if arr.ndim == 2:
                            arr = np.stack([arr]*3, axis=-1)
                        # メタ情報
                        cat = df.iloc[i].get("category", "")
                        cat_str = str(cat).strip() if isinstance(cat, str) else ""
                        p = float(resp[i, c])
                        imgs.append(arr)
                        metas.append((cat_str, p))
                except Exception:
                    pass
            if not imgs:
                return None
            # グリッド配置
            import math
            n = len(imgs)
            cols = int(math.ceil(math.sqrt(n)))
            rows = int(math.ceil(n/cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
            axes = np.atleast_2d(axes)
            for ax in axes.flat:
                ax.axis('off')
            for k, arr in enumerate(imgs):
                r, c2 = divmod(k, cols)
                ax = axes[r, c2]
                ax.imshow(arr.astype(np.uint8))
                ax.axis('off')
                # タイル左上に 元カテゴリのみ をオーバーレイ（確率は表示しない）
                cat_str, p = metas[k]
                if cat_str:
                    label = f"{cat_str}"
                    ax.text(
                        4, 12, label, color='white', fontsize=8, ha='left', va='top',
                        bbox=dict(facecolor='black', alpha=0.4, pad=2, edgecolor='none')
                    )
            fig.tight_layout(pad=0)
            out_path = out_dir / f"mosaic_c{c}.png"
            fig.savefig(out_path, dpi=160, bbox_inches='tight')
            plt.close(fig)
            return out_path.name
        except Exception as e:
            print(f"[WARN] モザイク作成失敗 c={c}: {e}")
            return None

    # テキスト壁（クラスタ上位テキストを多めに一覧表示）
    def build_text_wall_html(c, idxs_sorted, top_n):
        items = []
        count = 0
        for i in idxs_sorted:
            txt = df.iloc[i].get("text", "")
            # 値が0/"0"/空は非表示
            if txt is None:
                continue
            txt_str = str(txt).strip()
            if txt_str == "" or txt_str == "0" or txt_str == "０":
                continue
            # 元カテゴリ名のみ表示
            cat = df.iloc[i].get("category", "")
            cat_str = str(cat).strip() if isinstance(cat, str) else ""
            cat_html = f"<span class=\"cat\">({html_escape(cat_str)})</span>" if cat_str else ""

            txt_short = txt_str
            if len(txt_short) > 160:
                txt_short = txt_short[:160] + "…"
            items.append(f"<div class=\"titem\">{cat_html} {html_escape(txt_short)}</div>")
            count += 1
            if count >= top_n:
                break
        if not items:
            return ""
        return f"<div class=\"wall-text\">{''.join(items)}</div>"

    # 音声壁（クラスタ上位の音声プレイヤーを並べる）
    def build_audio_wall_html(c, idxs_sorted, top_n):
        items = []
        count = 0
        for i in idxs_sorted:
            snd_src = df.iloc[i].get("snd", None)
            if not (isinstance(snd_src, str) and os.path.exists(snd_src)):
                continue
            disp = None
            try:
                if snd_src.lower().endswith(WAV_EXTS):
                    disp = out_dir / f"audw_c{c}_i{i}{Path(snd_src).suffix}"
                    copy_if_small(snd_src, disp)
                    disp = disp.name
                elif snd_src.lower().endswith('.npz'):
                    mel = load_npz_mel(snd_src)
                    y, sr = synthesize_wav_from_mel(mel, sr=44100, n_fft=1024, hop_length=128, n_mels=128)
                    disp = out_dir / f"audw_c{c}_i{i}.wav"
                    save_wav(y, sr, disp)
                    disp = disp.name
            except Exception as e:
                print(f"[WARN] 音声壁作成失敗 c={c} i={i}: {e}")
                disp = None
            if disp:
                # 元カテゴリ名のみ表示
                cat = df.iloc[i].get("category", "")
                cat_str = str(cat).strip() if isinstance(cat, str) else ""
                cat_html = f"<span class=\"cat\">({html_escape(cat_str)})</span>" if cat_str else ""
                items.append(
                    f"<div class=\"aitem\"><div class=\"meta\">{cat_html}</div>"
                    f"<audio controls src=\"gallery_assets/{disp}\"></audio></div>"
                )
                count += 1
                if count >= top_n:
                    break
        if not items:
            return ""
        return f"<div class=\"wall-audio\">{''.join(items)}</div>"

    # 抜き出し対象のインデックス集合を作る（責務で降順）
    targets = []
    if args.cluster is None:
        for c in range(n_clusters):
            idxs = np.where(clusters == c)[0]
            if len(idxs) == 0:
                continue
            idxs_sorted = idxs[np.argsort(resp[idxs, c])[::-1]]
            take = min(args.per_cluster, len(idxs_sorted))
            chosen = list(idxs_sorted[:take])
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
            idxs_sorted = idxs[np.argsort(resp[idxs, c])[::-1]]
            take = min(args.per_cluster, len(idxs_sorted))
            chosen = list(idxs_sorted[:take])
            for i in chosen:
                targets.append((c, i))

    # 以前のカード生成は不要のため削除（画像・テキスト・音声のウォールのみ表示）

    # クラスタごとにモザイク＋テキスト/音声ウォールのセクションを作る
    sections = []
    for c in range(n_clusters if args.cluster is None else args.cluster+1):
        if args.cluster is not None and c != args.cluster:
            continue
        mosaic = build_cluster_mosaic(c, top=max(9, args.per_cluster))
        # クラスタ内の上位順インデックス（責務降順）
        idxs_all = np.where(clusters == c)[0]
        wall_html = ""
        if len(idxs_all) > 0:
            idxs_sorted_all = idxs_all[np.argsort(resp[idxs_all, c])[::-1]]
            wall_text = build_text_wall_html(c, idxs_sorted_all, args.text_wall_count)
            wall_audio = build_audio_wall_html(c, idxs_sorted_all, args.audio_wall_count)
            wall_html = (('<h3>Texts</h3>'+wall_text) if wall_text else '') + \
                        (('<h3>Audio</h3>'+wall_audio) if wall_audio else '')
        section = f"""
        <section class="cluster">
          <h2>Cluster {c}</h2>
          {('<div class="mosaic"><img src="gallery_assets/'+mosaic+'" alt="mosaic"></div>') if mosaic else ''}
          {wall_html}
        </section>
        """
        sections.append(section)

    # =========================
    # HTML 生成
    # =========================
    out_dir.mkdir(parents=True, exist_ok=True)
    header_img_html = f'<div class="tsne"><img src="{tsne_img}" alt="t-SNE"></div>' if tsne_img else ''
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
.card .tag {{ font-size: 12px; color: var(--muted); }}
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
section.cluster {{ margin-top: 24px; }}
section.cluster h2 {{ margin: 0 0 8px; font-size: 18px; }}
section.cluster h3 {{ margin: 8px 0 8px; font-size: 14px; color: var(--muted); font-weight: 600; }}
.mosaic {{ display:flex; justify-content:center; }}
.mosaic img {{ width: 100%; max-width: {args.mosaic_max_width}px; height: auto; border-radius: 12px; border: 1px solid var(--border); margin-bottom: 8px; }}
.tsne {{ display:flex; justify-content:center; }}
.tsne img {{ width: 100%; max-width: {args.tsne_max_width}px; height: auto; border-radius: 12px; border: 1px solid var(--border); margin: 8px 0 16px; }}
.legend {{ display:flex; justify-content:center; margin: 0 0 16px; }}
.cl-legend {{ border-collapse: collapse; background: #0f1318; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; min-width: 360px; }}
.cl-legend th, .cl-legend td {{ border-bottom: 1px solid var(--border); padding: 6px 10px; font-size: 13px; }}
.cl-legend thead th {{ background: #12171d; color: var(--muted); font-weight: 600; }}
.cl-legend tbody tr:last-child td {{ border-bottom: none; }}
.cl-legend td.num {{ text-align: right; width: 72px; }}
.swatch {{ display:inline-block; width: 12px; height: 12px; border-radius: 3px; border: 1px solid var(--border); margin-right: 6px; vertical-align: -1px; }}
.wall-text {{
  display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 8px; margin: 8px 0 16px;
}}
.wall-text .titem {{ background: #0f1318; border: 1px solid var(--border); border-radius: 10px; padding: 8px; font-size: 13px; line-height: 1.5; }}
.wall-audio {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 8px; margin: 8px 0 16px; }}
.wall-audio .aitem {{ background: #0f1318; border: 1px solid var(--border); border-radius: 10px; padding: 8px; font-size: 12px; }}
.cat {{ color: var(--muted); margin-left: 6px; font-size: 12px; }}
footer {{ margin-top: 16px; color: var(--muted); font-size: 12px; }}
.note {{ color: var(--muted); font-size: 13px; margin: 8px 0 16px; }}
</style>
</head>
<body>
<h1>{html_escape(args.title)}</h1>
<div class="note">上: t-SNE（クラスタ色・最大幅 {args.tsne_max_width}px）。下にクラスタ色の凡例を表示。各クラスタはモザイク（最大幅 {args.mosaic_max_width}px）、続いてテキスト一覧と音声一覧を表示します。</div>
{header_img_html}
{legend_html}
{''.join(sections) if sections else '<p>表示対象がありません。</p>'}
<footer>Generated in {html_escape(str(artifacts))}</footer>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    print(f"[DONE] ギャラリーを出力しました → {html_path}")
    print("ローカルで開く： file://" + str(html_path))

if __name__ == "__main__":
    main()
