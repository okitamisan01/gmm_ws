import os
import random
import pandas as pd
import json
import re

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)

esc_np_dir = os.path.join(root_dir, "data_processed", "audio")
oc_np_dir = os.path.join(root_dir, "data_processed", "image")

# 使用済みの画像/音声は「分割内」で二度と使わない（train/test間の重複は許容）
used_img_test = set()
used_snd_test = set()
used_img_train = set()
used_snd_train = set()

def avail_unique(seq, used):
    return [x for x in seq if x not in used]

def take_unique(seq, used, k):
    cand = avail_unique(seq, used)
    if not cand:
        return []
    k = min(k, len(cand))
    picks = random.sample(cand, k)
    used.update(picks)
    return picks

texts_data_train = []
img_data_train = []
snd_data_train = []
target_train = []
category_train = []

texts_data_test = []
img_data_test = []
snd_data_test = []
target_test = []
category_test = []
t = 0

chosen_oc_esc = {
    58:30, 102:40, 239:35, 245:16, 113:14,
    170:10, 89:1, 73:48, 251:47, 56:0, 80:4
}

chosen_kv = {
    58: "doorknob", 102: "helicopter-101", 239: "washing-machine",
    245: "windmill", 113: "hummingbird", 170: "rainbow",
    89: "goose", 73: "fireworks", 251: "airplanes-101",
    56: "dog", 80: "frog"
}

with open(os.path.join(root_dir, "data_processed", "text", "oc_txt_dict.json"), "r", encoding="utf-8") as f:
    oc_txt_dict = {int(k): v for k, v in json.load(f).items()}

for ok, c in chosen_kv.items():
    oc_list = []
    for x in os.listdir(oc_np_dir):
        p = os.path.join(oc_np_dir, x)
        if not os.path.isfile(p): continue
        m = re.match(r"^(\d+)[_-]", x)
        if not m: continue
        try:
            cid = int(m.group(1))
        except ValueError:
            continue
        if cid == ok:
            oc_list.append(p)

    esc_list = []
    for x in os.listdir(esc_np_dir):
        p = os.path.join(esc_np_dir, x)
        if not os.path.isfile(p): continue
        try:
            esc_id = int(x.split("-")[-1].split(".")[0])
        except Exception:
            continue
        if esc_id == chosen_oc_esc[ok]:
            esc_list.append(p)

    txt_pool = oc_txt_dict.get(ok, [])

    # ========= Test: Full triplets（重複禁止：テスト分割内のみ） =========
    avail_img = avail_unique(oc_list, used_img_test)
    avail_snd = avail_unique(esc_list, used_snd_test)
    n_test = min(8, len(txt_pool), len(avail_img), len(avail_snd))
    if n_test == 0:
        print(f"[WARN] Skip class {ok} ({c}): insufficient data (unique img/snd in test)")
        t += 1
        continue

    pick_img = take_unique(oc_list, used_img_test, n_test)
    pick_snd = take_unique(esc_list, used_snd_test, n_test)
    pick_txt_idx = random.sample(range(len(txt_pool)), n_test)

    for i in range(n_test):
        texts_data_test.append(txt_pool[pick_txt_idx[i]])
        img_data_test.append(pick_img[i])
        snd_data_test.append(pick_snd[i])
        target_test.append(t)
        category_test.append(c)

    # ========= Test: One missing =========
    # 最大8件めやす、在庫に応じて減少
    attempts = 0
    made = 0
    while made < 8 and attempts < 8*4:
        attempts += 1
        m = random.choice([0, 1, 2])  # 0:missing text, 1:missing img, 2:missing snd
        if m == 0:
            # need unique img + snd (within test)
            ai = avail_unique(oc_list, used_img_test)
            asn = avail_unique(esc_list, used_snd_test)
            if ai and asn:
                img = take_unique(oc_list, used_img_test, 1)[0]
                snd = take_unique(esc_list, used_snd_test, 1)[0]
                texts_data_test.append(0)
                img_data_test.append(img)
                snd_data_test.append(snd)
                target_test.append(t); category_test.append(c)
                made += 1
        elif m == 1:
            # need unique snd only (within test)
            asn = avail_unique(esc_list, used_snd_test)
            if asn:
                txt = random.choice(txt_pool) if txt_pool else 0
                snd = take_unique(esc_list, used_snd_test, 1)[0]
                texts_data_test.append(txt)
                img_data_test.append(None)
                snd_data_test.append(snd)
                target_test.append(t); category_test.append(c)
                made += 1
        else:
            # need unique img only (within test)
            ai = avail_unique(oc_list, used_img_test)
            if ai:
                txt = random.choice(txt_pool) if txt_pool else 0
                img = take_unique(oc_list, used_img_test, 1)[0]
                texts_data_test.append(txt)
                img_data_test.append(img)
                snd_data_test.append(None)
                target_test.append(t); category_test.append(c)
                made += 1
        # 在庫が完全に尽きたら中断（within test）
        if not avail_unique(oc_list, used_img_test) and not avail_unique(esc_list, used_snd_test):
            break

    # ========= Test: Two missing =========
    attempts = 0
    made = 0
    while made < 8 and attempts < 8*4:
        attempts += 1
        h = random.choice([0, 1, 2])  # 0:text only, 1:img only, 2:snd only
        if h == 0:
            txt = random.choice(txt_pool) if txt_pool else 0
            texts_data_test.append(txt); img_data_test.append(None); snd_data_test.append(None)
            target_test.append(t); category_test.append(c)
            made += 1
        elif h == 1:
            ai = avail_unique(oc_list, used_img_test)
            if ai:
                img = take_unique(oc_list, used_img_test, 1)[0]
                texts_data_test.append(0); img_data_test.append(img); snd_data_test.append(None)
                target_test.append(t); category_test.append(c)
                made += 1
        else:
            asn = avail_unique(esc_list, used_snd_test)
            if asn:
                snd = take_unique(esc_list, used_snd_test, 1)[0]
                texts_data_test.append(0); img_data_test.append(None); snd_data_test.append(snd)
                target_test.append(t); category_test.append(c)
                made += 1
        if not avail_unique(oc_list, used_img_test) and not avail_unique(esc_list, used_snd_test):
            break

    # ========= Train pools（テストで使ったものとは独立。学習分割内で重複禁止） =========
    train_text = txt_pool[:]  # テキストは重複許容
    train_img = avail_unique(oc_list, used_img_train)
    train_snd = avail_unique(esc_list, used_snd_train)

    if not train_img or not train_snd:
        print(f"[WARN] Not enough unique training pool for class {ok} ({c}) (img={len(train_img)}, snd={len(train_snd)})")
        t += 1
        continue

    # Train: Full triplets（最大60件、在庫でクリップ）
    n_train_full = min(60, len(train_img), len(train_snd))
    pick_img = take_unique(oc_list, used_img_train, n_train_full)
    pick_snd = take_unique(esc_list, used_snd_train, n_train_full)
    for i in range(n_train_full):
        texts_data_train.append(random.choice(train_text) if train_text else 0)
        img_data_train.append(pick_img[i])
        snd_data_train.append(pick_snd[i])
        target_train.append(t); category_train.append(c)

    # Train: One missing（最大60件を目安、在庫が尽きるまで）
    attempts = 0
    made = 0
    while made < 60 and attempts < 60*4:
        attempts += 1
        m = random.choice([0, 1, 2])
        if m == 0:
            ai = avail_unique(oc_list, used_img_train)
            asn = avail_unique(esc_list, used_snd_train)
            if ai and asn:
                img = take_unique(oc_list, used_img_train, 1)[0]
                snd = take_unique(esc_list, used_snd_train, 1)[0]
                texts_data_train.append(0); img_data_train.append(img); snd_data_train.append(snd)
                target_train.append(t); category_train.append(c)
                made += 1
        elif m == 1:
            asn = avail_unique(esc_list, used_snd_train)
            if asn:
                txt = random.choice(train_text) if train_text else 0
                snd = take_unique(esc_list, used_snd_train, 1)[0]
                texts_data_train.append(txt); img_data_train.append(None); snd_data_train.append(snd)
                target_train.append(t); category_train.append(c)
                made += 1
        else:
            ai = avail_unique(oc_list, used_img_train)
            if ai:
                txt = random.choice(train_text) if train_text else 0
                img = take_unique(oc_list, used_img_train, 1)[0]
                texts_data_train.append(txt); img_data_train.append(img); snd_data_train.append(None)
                target_train.append(t); category_train.append(c)
                made += 1
        if not avail_unique(oc_list, used_img_train) and not avail_unique(esc_list, used_snd_train):
            break

    # Train: Two missing（最大60件を目安、在庫が尽きるまで）
    attempts = 0
    made = 0
    while made < 60 and attempts < 60*4:
        attempts += 1
        h = random.choice([0, 1, 2])
        if h == 0:
            txt = random.choice(train_text) if train_text else 0
            texts_data_train.append(txt); img_data_train.append(None); snd_data_train.append(None)
            target_train.append(t); category_train.append(c)
            made += 1
        elif h == 1:
            ai = avail_unique(oc_list, used_img_train)
            if ai:
                img = take_unique(oc_list, used_img_train, 1)[0]
                texts_data_train.append(0); img_data_train.append(img); snd_data_train.append(None)
                target_train.append(t); category_train.append(c)
                made += 1
        else:
            asn = avail_unique(esc_list, used_snd_train)
            if asn:
                snd = take_unique(esc_list, used_snd_train, 1)[0]
                texts_data_train.append(0); img_data_train.append(None); snd_data_train.append(snd)
                target_train.append(t); category_train.append(c)
                made += 1
        if not avail_unique(oc_list, used_img_train) and not avail_unique(esc_list, used_snd_train):
            break

    t += 1

# 保存先ディレクトリを確保
output_dir = os.path.join(root_dir, "dataset_missing")
os.makedirs(output_dir, exist_ok=True)

# DataFrame にして保存
dataset_train = pd.DataFrame({
    "category": category_train,
    "target": target_train,
    "img": img_data_train,
    "snd": snd_data_train,
    "text": texts_data_train
})

dataset_test = pd.DataFrame({
    "category": category_test,
    "target": target_test,
    "img": img_data_test,
    "snd": snd_data_test,
    "text": texts_data_test
})

# ===== バランス調整: クラス数を最小件数に揃える（ダウンサンプリング） =====
def balance_by_category(df, cat_col="category", seed=0, target_per_class=None):
    if df.empty:
        return df
    g = df.groupby(cat_col, dropna=False)
    if target_per_class is None:
        target_per_class = g.size().min()
    parts = []
    for cat, sub in g:
        n = len(sub)
        if n <= target_per_class:
            parts.append(sub)
        else:
            parts.append(sub.sample(n=target_per_class, random_state=seed))
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out

# テスト/学習をそれぞれ揃える（最小件数に合わせる）
dataset_test_bal = balance_by_category(dataset_test, seed=0)
dataset_train_bal = balance_by_category(dataset_train, seed=0)

# 保存
dataset_train_bal.to_csv(os.path.join(output_dir, "dataset_train_missing.csv"), index=False)
dataset_test_bal.to_csv(os.path.join(output_dir, "dataset_test_missing.csv"), index=False)

print("Balanced per-class counts:")
print("train:", dataset_train_bal["category"].value_counts().to_dict())
print("test:", dataset_test_bal["category"].value_counts().to_dict())
