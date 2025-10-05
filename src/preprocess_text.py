import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional
from urllib.parse import quote
import datetime

try:
    import requests  # lightweight HTTP client
except Exception:  # pragma: no cover
    requests = None

# Add a proper User-Agent per Wikipedia API etiquette
HEADERS = {
    "User-Agent": "gmm-ws-textbuilder/0.1 (Linux; +https://www.mediawiki.org/wiki/API:Etiquette)",
    "Accept": "application/json",
}

# Chosen category mapping (Caltech256 class ID -> ESC-50 target ID)
# Keep in sync with preprocessing/making_category.py selection
chosen_oc_esc = {
    58: 30,
    102: 40,
    239: 35,
    245: 16,
    113: 14,
    170: 10,
    89: 1,
    73: 48,
    251: 47,
    56: 0,
    80: 4,
}

# Japanese title overrides for Caltech256 class names -> canonical ja.wikipedia titles
# 追加：曖昧/英語名に対する日本語タイトルの手動オーバーライド
JAPANESE_TITLE_OVERRIDES = {
    "dog": "イヌ",
    "frog": "カエル",
    "helicopter": "ヘリコプター",
    "wind": "風",
    "hummingbird": "ハチドリ",
    "goose": "ガン (鳥)",
    "fireworks": "花火",
    "airplane": "飛行機",
    "washing-machine": "洗濯機",
    "windmill": "風車",
    "doorknob": "ドアノブ",
    "rainbow": "虹",
}

# Concept-specific keywords to validate on-topic outputs
CONCEPT_KEYWORDS: Dict[str, List[str]] = {
    "イヌ": ["イヌ", "犬"],
    "カエル": ["カエル", "蛙"],
    "ヘリコプター": ["ヘリコプター", "回転翼機"],
    "風": ["風"],
    "ハチドリ": ["ハチドリ"],
    "ガン (鳥)": ["ガン", "雁"],
    "花火": ["花火"],
    "飛行機": ["飛行機", "航空機", "固定翼機"],
    "洗濯機": ["洗濯機"],
    "風車": ["風車"],
    "ドアノブ": ["ドアノブ", "扉のノブ", "ドアの取っ手", "ドアハンドル"],
    "虹": ["虹", "レインボー"],
}

# Words that strongly indicate biographies/entertainment (off-topic for generic concepts)
OFFTOPIC_TOKENS = [
    "バンド", "ロックバンド", "グループ", "ユニット", "歌手", "ドラマー", "ミュージシャン",
    "アルバム", "シングル", "楽曲", "デビュー", "芸人", "俳優", "監督", "作曲家", "作家",
    "生まれ", "出生", "本名", "活動", "メンバー", "加入", "脱退", "チャート", "ツアー",
]

# Blacklist of Wikidata instance-of QIDs for unrelated concepts (people, bands, works, companies, etc.)
WIKIDATA_P31_BLACKLIST = set(
    [
        "Q5",        # human
        "Q215380",   # band
        "Q43229",    # organization
        "Q4830453",  # business / company
        "Q11424",    # film
        "Q5398426",  # television series
        "Q7889",     # video game
        "Q482994",   # album
        "Q7366",     # song
        "Q134556",   # single
        "Q8274",     # manga
        "Q1107",     # anime
        "Q95074",    # fictional character
        "Q571",      # book
    ]
)

# Category keywords blacklist (Japanese Wikipedia categories)
CATEGORY_KEYWORD_BLACKLIST = [
    "人物", "存命人物", "年没", "生年", "ミュージシャン", "音楽家", "歌手", "俳優", "タレント", "声優",
    "バンド", "楽曲", "シングル", "アルバム", "アーティスト",
    "企業", "会社", "出版社", "放送局",
    "テレビドラマ", "テレビ番組", "アニメ作品", "漫画作品", "映画作品", "小説",
]


def _resolve_ja_concept_name(name: str) -> str:
    """Return a Japanese concept label for prompting. Prefer overrides."""
    base = name.replace("_", " ").replace(".", " ")
    base = re.sub(r"-\d+$", "", base).replace("-", " ").strip()
    if name in JAPANESE_TITLE_OVERRIDES:
        return JAPANESE_TITLE_OVERRIDES[name]
    if base in JAPANESE_TITLE_OVERRIDES:
        return JAPANESE_TITLE_OVERRIDES[base]
    return base or name


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(k for k in keywords if k and k in text)


def _is_offtopic_bio_or_entertainment(text: str) -> bool:
    low = text
    return any(tok for tok in OFFTOPIC_TOKENS if tok in low)


def _looks_like_biography(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    head = t.split("\n", 1)[0]
    # 典型的な人物文パターンを軽く弾く
    return any(
        kw in head
        for kw in ["生まれ", "俳優", "歌手", "ドラマー", "作家", "人物", "バンドの", "メンバー"]
    )


def _wiki_page_metadata(title: str, lang: str = "ja") -> Dict[str, List[str]]:
    """Return page metadata: wikidata QID and categories list for a given title."""
    if requests is None:
        raise RuntimeError("requests が見つかりません。requirements.txt に requests を追加してください。")
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageprops|categories",
        "ppprop": "wikibase_item",
        "cllimit": 200,
        "format": "json",
        "utf8": 1,
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    qid = ""
    cats: List[str] = []
    for _, page in pages.items():
        pp = page.get("pageprops", {})
        if pp.get("wikibase_item"):
            qid = pp.get("wikibase_item")
        for c in page.get("categories", []) or []:
            if c.get("title"):
                # Category: を取り除く
                cats.append(c["title"].split(":", 1)[-1])
    return {"qid": qid, "categories": cats}


def _wikidata_instance_of(qid: str) -> List[str]:
    """Return list of QIDs that the entity is instance-of (P31)."""
    if not qid:
        return []
    if requests is None:
        raise RuntimeError("requests が見つかりません。requirements.txt に requests を追加してください。")
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetclaims",
        "entity": qid,
        "property": "P31",
        "format": "json",
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    claims = data.get("claims", {}).get("P31", [])
    qids: List[str] = []
    for cl in claims:
        try:
            val = cl["mainsnak"]["datavalue"]["value"]["id"]
            if val:
                qids.append(val)
        except Exception:
            continue
    return qids


def repo_root() -> str:
    """Return repository root path (one level up from this file)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def sanitize_name(name: str) -> str:
    """Make a filesystem-friendly name."""
    s = name.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9._\-]", "", s)
    return s


def load_oc256_classes(base_dir: str) -> Dict[int, str]:
    """Scan Caltech256 folder to build mapping {class_id: class_name}."""
    oc256_dir = os.path.join(base_dir, "data_raw", "Caltech256", "256_ObjectCategories")
    if not os.path.isdir(oc256_dir):
        raise FileNotFoundError(f"Not found: {oc256_dir}")
    result: Dict[int, str] = {}
    for entry in os.listdir(oc256_dir):
        # Expect names like '056.dog'
        try:
            label_str, name = entry.split(".", 1)
            cid = int(label_str)
            result[cid] = name
        except Exception:
            # Skip unexpected entries
            continue
    if not result:
        raise RuntimeError("Caltech256 classes not found. Check dataset layout.")
    return result


def get_text_root(base_dir: str) -> str:
    return os.path.join(base_dir, "data_processed", "text")


def init_text_folders(base_dir: str) -> None:
    """Create data_processed/text/<cid>-<name>/ folders and a placeholder file if empty."""
    oc_classes = load_oc256_classes(base_dir)
    root = get_text_root(base_dir)
    os.makedirs(root, exist_ok=True)

    for cid in chosen_oc_esc.keys():
        cname = oc_classes.get(cid, f"class-{cid}")
        folder = os.path.join(root, f"{cid}-{sanitize_name(cname)}")
        os.makedirs(folder, exist_ok=True)

        # Drop a placeholder if the folder is empty
        if not any(os.scandir(folder)):
            placeholder = os.path.join(folder, "00_placeholder.txt")
            with open(placeholder, "w", encoding="utf-8") as f:
                f.write(
                    "このファイルはプレースホルダーです。\n"
                    "同じフォルダーに Wikipedia（日本語）からコピペしたテキストを .txt または .md で保存してください。\n"
                    "1ファイル=1サンプルとして扱います。\n"
                )

    # Write a manifest for convenience
    manifest_path = os.path.join(root, "manifest.json")
    manifest = {
        "text_root": root,
        "categories": {
            str(cid): {
                "name": oc_classes.get(cid, f"class-{cid}"),
                "folder": f"{cid}-{sanitize_name(oc_classes.get(cid, f'class-{cid}'))}",
            }
            for cid in chosen_oc_esc.keys()
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_oc_txt_dict(base_dir: str) -> Dict[int, List[str]]:
    """Read .txt/.md under data_processed/text/<cid>-<name>/ and build {cid: [texts...]}."""
    root = get_text_root(base_dir)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Text root not found: {root}. Run with --init and place text files first."
        )

    oc_txt_dict: Dict[int, List[str]] = {}
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        # Folder format: <cid>-<name>
        try:
            cid_str = entry.name.split("-", 1)[0]
            cid = int(cid_str)
        except Exception:
            continue

        texts: List[str] = []
        for fentry in os.scandir(entry.path):
            if not fentry.is_file():
                continue
            if not (fentry.name.endswith(".txt") or fentry.name.endswith(".md")):
                continue
            if fentry.name.upper().startswith("ATTRIBUTION"):
                continue
            content = read_text_file(fentry.path)
            if content:
                texts.append(content)
        if texts:
            oc_txt_dict[cid] = texts

    if not oc_txt_dict:
        raise RuntimeError(
            "No text samples found. Place .txt/.md files into each category folder and retry."
        )

    # Save as JSON (UTF-8, keep Japanese as-is)
    out_json = os.path.join(root, "oc_txt_dict.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in oc_txt_dict.items()}, f, ensure_ascii=False, indent=2)

    return oc_txt_dict


def _split_into_chunks(text: str, max_chars: int = 600) -> List[str]:
    """従来の文字数ベース分割（互換用）。"""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= max_chars:
                buf = p
            else:
                start = 0
                while start < len(p):
                    chunks.append(p[start : start + max_chars])
                    start += max_chars
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def _split_by_time(text: str, seconds: int = 60, reading_cpm: int = 500) -> List[str]:
    """時間ベースで分割。句点（。！？）優先で自然に切る。
    seconds: 1チャンクの目安秒数
    reading_cpm: 1分あたりの読字文字数（日本語の平均 450–600 目安）
    """
    if seconds <= 0 or reading_cpm <= 0:
        return [text.strip()] if text.strip() else []
    target = max(100, int(reading_cpm * (seconds / 60.0)))
    s = re.sub(r"\s+\n", "\n", text.strip())
    # 一旦「文」に分割（。！？ + 任意の引用符・空白・改行）
    sentences = re.split(r"(?<=[。！？])(?:[”」』】』\"]*)\s*", s)
    chunks, buf = [], ""
    for sent in sentences:
        if not sent:
            continue
        cand = (buf + (" " if buf else "") + sent).strip()
        if len(cand) <= target:
            buf = cand
        else:
            if buf:
                chunks.append(buf)
                buf = sent.strip()
            else:
                # 単独で長い文は強制分割（極端な長文保険）
                start = 0
                while start < len(sent):
                    chunks.append(sent[start:start+target])
                    start += target
                buf = ""
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


def _wiki_search_titles(term: str, lang: str = "ja", limit: int = 5) -> List[str]:
    if requests is None:
        raise RuntimeError("requests が見つかりません。requirements.txt に requests を追加してください。")
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": term,
        "format": "json",
        "srlimit": limit,
        "utf8": 1,
        "srnamespace": 0,  # 記事本文のみ
        "srqiprofile": "classic_noboostlinks",
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    hits = data.get("query", {}).get("search", [])
    titles = [h.get("title") for h in hits if h.get("title")]
    return titles


def _wiki_fetch_summary(title: str, lang: str = "ja") -> Dict[str, str]:
    if requests is None:
        raise RuntimeError("requests が見つかりません。requirements.txt に requests を追加してください。")
    api = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    r = requests.get(api, headers=HEADERS, timeout=20)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    js = r.json()
    # disambiguation/redirect は除外
    if js.get("type") in ("disambiguation", "redirect"):
        return {}
    extract = js.get("extract") or ""
    # タイトル末尾の（…）は原則除外（全角）
    ttl = js.get("title", title) or title
    if re.search(r"（.+）$", ttl):
        return {}
    url = (
        js.get("content_urls", {})
        .get("desktop", {})
        .get("page", f"https://{lang}.wikipedia.org/wiki/{quote(title)}")
    )
    desc = js.get("description") or ""
    return {"title": ttl, "url": url, "extract": extract, "description": desc}


def _wiki_langlink(title: str, from_lang: str = "en", to_lang: str = "ja") -> str:
    """Resolve an interlanguage link from from_lang title to to_lang title. Returns '' if not found."""
    if requests is None:
        raise RuntimeError("requests が見つかりません。requirements.txt に requests を追加してください。")
    url = f"https://{from_lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "langlinks",
        "titles": title,
        "lllang": to_lang,
        "format": "json",
        "utf8": 1,
        "lllimit": 50,
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        lls = page.get("langlinks") or []
        for ll in lls:
            if ll.get("lang") == to_lang and ll.get("*", ll.get("title")):
                # MediaWiki returns the translated title in '*' key; 'title' for some clients
                return ll.get("*", ll.get("title", ""))
    return ""


def _generate_query_variants(name: str) -> List[str]:
    base = name.replace("_", " ").replace(".", " ")
    base = re.sub(r"-\d+$", "", base).replace("-", " ").strip()
    variants = []
    # ① override があれば最優先
    if name in JAPANESE_TITLE_OVERRIDES:
        variants.append(JAPANESE_TITLE_OVERRIDES[name])
    if base in JAPANESE_TITLE_OVERRIDES:
        variants.append(JAPANESE_TITLE_OVERRIDES[base])
    # ② 通常化した候補
    variants += [name, base]
    # ③ 分野ヒントを追加（あいまいさ回避を避けるため）
    HINTS = {
        "dog": "動物", "frog": "両生類", "hummingbird":"鳥",
        "helicopter":"航空機", "airplanes":"航空機",
        "washing machine":"家電", "windmill":"機械", "rainbow":"気象",
        "wind":"気象", "goose":"鳥", "fireworks":"行事",
        "doorknob":"建築金物",
    }
    key = base.lower()
    for k,v in HINTS.items():
        if k in key:
            # override があればそれ、なければ base を使ってヒント付与
            jp = JAPANESE_TITLE_OVERRIDES.get(base, base)
            variants.append(f"{jp} {v}")
            break
    # 重複除去
    seen, uniq = set(), []
    for v in variants:
        if v and v not in seen:
            uniq.append(v); seen.add(v)
    return uniq


def _call_gemini_text(model: str, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """Call Gemini (Google Generative AI) to generate text. Requires GOOGLE_API_KEY or GEMINI_API_KEY."""
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:
        raise RuntimeError("google-generativeai が未インストールです。requirements.txt に追加しインストールしてください。") from e

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY もしくは GEMINI_API_KEY が未設定です。環境変数に API キーを設定してください。")

    genai.configure(api_key=api_key)
    try:
        model_name = (model or "").replace("models/", "") or "gemini-2.5-flash"
        gm = genai.GenerativeModel(model_name=model_name, system_instruction=system_prompt)
        resp = gm.generate_content(
            user_prompt,
            generation_config={"temperature": float(temperature)},
        )
    except Exception as e:
        raise RuntimeError(f"Gemini API エラー: {e}")

    text = getattr(resp, "text", None)
    if not text:
        try:
            parts = []
            for cand in getattr(resp, "candidates", []) or []:
                for part in getattr(cand, "content", {}).get("parts", []) or []:
                    if hasattr(part, "text"):
                        parts.append(part.text)
            text = "\n".join([p for p in parts if p])
        except Exception:
            text = None
    if not text:
        raise RuntimeError("Gemini 応答の解析に失敗しました。")
    return text.strip()


def auto_generate_with_llm(
    base_dir: str,
    samples_per_category: int = 5,
    chunk_chars: int = 600,
    model: Optional[str] = None,
    temperature: float = 0.2,
    delay_sec: float = 0.2,
    max_retries: int = 3,
) -> None:
    """Generate Japanese text samples per category using Gemini only.

    Writes files under data_processed/text/<cid>-<name>/ and appends to ATTRIBUTION.txt.
    """
    oc_classes = load_oc256_classes(base_dir)
    root = get_text_root(base_dir)
    os.makedirs(root, exist_ok=True)

    # Gemini default model: 2.5 系に更新
    model = (model or os.environ.get("GEMINI_MODEL") or os.environ.get("GOOGLE_MODEL") or "gemini-2.5-flash").replace("models/", "")

    sys_prompt = (
        "あなたは百科事典編集者です。指定された一般概念について日本語で中立的に説明します。"
        "同名の固有名詞（バンド・楽曲・人物・企業・作品名など）には触れません。"
        "定義、特徴、用途/仕組み/生態、関連事項を含め、平易な文章で1〜2段落にまとめてください。"
        "最初の文は『<用語>とは、…』で始め、用語名を必ず含めてください。"
        "出力は本文のみ。箇条書き、見出し、出典の記載、注意書きは禁止。"
    )

    for cid in chosen_oc_esc.keys():
        cname = oc_classes.get(cid, f"class-{cid}")
        folder = os.path.join(root, f"{cid}-{sanitize_name(cname)}")
        os.makedirs(folder, exist_ok=True)
        ja_label = _resolve_ja_concept_name(cname)
        concept_keywords = CONCEPT_KEYWORDS.get(ja_label, [ja_label])

        saved = 0
        for i in range(samples_per_category):
            tries = 0
            while tries < max_retries:
                user_prompt = (
                    f"用語: {ja_label}\n"
                    f"長さの目安: {max(220, min(chunk_chars, 800))} 文字前後。\n"
                    "禁止: バンド/楽曲/人物/企業/作品の紹介、年表、ディスコグラフィ。\n"
                )
                try:
                    text = _call_gemini_text(model=model, system_prompt=sys_prompt, user_prompt=user_prompt, temperature=temperature)
                except Exception as e:
                    print(f"[WARN] LLM 生成失敗 cid={cid} name={cname}: {e}")
                    break

                if not text:
                    tries += 1
                    continue
                # On-topic checks
                if _is_offtopic_bio_or_entertainment(text):
                    tries += 1
                    continue
                if not _contains_any(text, concept_keywords):
                    tries += 1
                    continue

                # Save chunks（モードに応じて後段で決定）
                # chunks の決定は呼び出し側で。
                if not chunks:
                    tries += 1
                    continue
                for j, ch in enumerate(chunks, start=1):
                    fname = os.path.join(folder, f"{str(saved+1).zfill(3)}_llm_{i+1}_{j}.txt")
                    with open(fname, "w", encoding="utf-8") as f:
                        f.write(ch)
                    saved += 1
                time.sleep(delay_sec)
                break

        if saved:
            placeholder = os.path.join(folder, "00_placeholder.txt")
            if os.path.exists(placeholder):
                try:
                    os.remove(placeholder)
                except OSError:
                    pass

            attr = os.path.join(folder, "ATTRIBUTION.txt")
            with open(attr, "a", encoding="utf-8") as f:
                f.write("Generated: LLM (Gemini)\n")
                f.write(f"Model: {model}\n")
                f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
                f.write("Notes: 生成テキスト。外部出典なし。\n\n")

        print(f"[INFO] {cid}-{cname}: {saved} サンプル LLM 生成 (gemini/{model})")


def main():
    parser = argparse.ArgumentParser(description="Prepare and build text data per category.")
    parser.add_argument("--init", action="store_true", help="Create folders for text data.")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build oc_txt_dict.json by reading .txt/.md files.",
    )
    parser.add_argument("--auto-llm", action="store_true", help="LLM でテキストを自動生成する (Gemini)")
    parser.add_argument("--llm-samples", type=int, default=5, help="各カテゴリの生成サンプル数")
    parser.add_argument("--llm-model", type=str, default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"), help="Gemini のモデル名")
    parser.add_argument("--llm-temperature", type=float, default=0.2, help="LLM の温度パラメータ")
    parser.add_argument("--max-pages", type=int, default=5, help="Max search pages per category.")
    parser.add_argument("--chunk-chars", type=int, default=600, help="（charsモード時）1チャンクの最大文字数")
    parser.add_argument("--chunk-mode", choices=["none","chars","time"], default="chars", help="分割方式: none=分割しない, chars=文字数, time=時間ベース")
    parser.add_argument("--chunk-seconds", type=int, default=60, help="（timeモード時）1チャンクの目安秒数")
    parser.add_argument("--reading-cpm", type=int, default=500, help="（timeモード時）1分あたりの読字文字数")
    parser.add_argument("--lang", type=str, default="ja", help="Wikipedia language code (default: ja)")
    args = parser.parse_args()

    base = repo_root()

    if args.init:
        init_text_folders(base)
        print("Initialized data_processed/text structure and manifest.json.")

    if getattr(args, "auto_llm", False):
        # 実行しつつ、分割モードをここで反映
        def _run():
            oc_classes = load_oc256_classes(base)
            root = get_text_root(base)
            os.makedirs(root, exist_ok=True)
            model = (args.llm_model or os.environ.get("GEMINI_MODEL") or os.environ.get("GOOGLE_MODEL") or "gemini-2.5-flash").replace("models/", "")
            sys_prompt = (
                "あなたは百科事典編集者です。指定された一般概念について日本語で中立的に説明します。"
                "同名の固有名詞（バンド・楽曲・人物・企業・作品名など）には触れません。"
                "定義、特徴、用途/仕組み/生態、関連事項を含め、平易な文章で1〜2段落にまとめてください。"
                "最初の文は『<用語>とは、…』で始め、用語名を必ず含めてください。"
                "出力は本文のみ。箇条書き、見出し、出典の記載、注意書きは禁止。"
            )
            for cid in chosen_oc_esc.keys():
                cname = oc_classes.get(cid, f"class-{cid}")
                folder = os.path.join(root, f"{cid}-{sanitize_name(cname)}")
                os.makedirs(folder, exist_ok=True)
                ja_label = _resolve_ja_concept_name(cname)
                concept_keywords = CONCEPT_KEYWORDS.get(ja_label, [ja_label])
                saved = 0
                for i in range(args.llm_samples):
                    tries = 0
                    while tries < 3:
                        user_prompt = (
                            f"用語: {ja_label}\n"
                            f"長さの目安: {max(220, min(args.chunk_chars, 800))} 文字前後。\n"
                            "禁止: バンド/楽曲/人物/企業/作品の紹介、年表、ディスコグラフィ。\n"
                        )
                        try:
                            text = _call_gemini_text(model=model, system_prompt=sys_prompt, user_prompt=user_prompt, temperature=args.llm_temperature)
                        except Exception as e:
                            print(f"[WARN] LLM 生成失敗 cid={cid} name={cname}: {e}")
                            break
                        if not text:
                            tries += 1; continue
                        if _is_offtopic_bio_or_entertainment(text):
                            tries += 1; continue
                        if not _contains_any(text, concept_keywords):
                            tries += 1; continue
                        # ---- 分割モードに応じた保存 ----
                        if args.chunk_mode == "none":
                            chunks = [text]
                        elif args.chunk_mode == "time":
                            chunks = _split_by_time(text, seconds=args.chunk_seconds, reading_cpm=args.reading_cpm)
                        else:
                            chunks = _split_into_chunks(text, max_chars=args.chunk_chars)
                        if not chunks:
                            tries += 1; continue
                        for j, ch in enumerate(chunks, start=1):
                            fname = os.path.join(folder, f"{str(saved+1).zfill(3)}_llm_{i+1}_{j}.txt")
                            with open(fname, "w", encoding="utf-8") as f:
                                f.write(ch.strip() + "\n")
                            saved += 1
                        time.sleep(0.2)
                        break
                if saved:
                    placeholder = os.path.join(folder, "00_placeholder.txt")
                    if os.path.exists(placeholder):
                        try: os.remove(placeholder)
                        except OSError: pass
                    attr = os.path.join(folder, "ATTRIBUTION.txt")
                    with open(attr, "a", encoding="utf-8") as f:
                        f.write("Generated: LLM (Gemini)\n")
                        f.write(f"Model: {model}\n")
                        f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
                        f.write("Notes: 生成テキスト。外部出典なし。\n\n")
                print(f"[INFO] {cid}-{cname}: {saved} サンプル LLM 生成 (gemini/{model})")
        _run()

    if args.build:
        oc_txt = build_oc_txt_dict(base)
        counts = {k: len(v) for k, v in oc_txt.items()}
        print("Built oc_txt_dict.json with counts:")
        for k in sorted(counts.keys()):
            print(f"  {k}: {counts[k]}")

    if not args.init and not getattr(args, "auto_llm", False) and not args.build:
        print(
            "Nothing to do. Use --init to scaffold folders, --auto-llm (Gemini) to generate, and --build to generate oc_txt_dict.json."
        )


if __name__ == "__main__":
    main()