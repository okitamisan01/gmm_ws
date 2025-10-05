import os
import random
import pandas as pd
import json
import re

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)

esc_np_dir = os.path.join(root_dir, "data_processed", "audio")
oc_np_dir = os.path.join(root_dir, "data_processed", "image")

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
    n_test = min(8, len(txt_pool), len(oc_list), len(esc_list))
    if n_test == 0:
        print(f"[WARN] Skip class {ok} ({c}): insufficient data")
        t += 1
        continue

    test_text_idx = random.sample(range(len(txt_pool)), n_test)
    test_img_idx = random.sample(range(len(oc_list)), n_test)
    test_snd_idx = random.sample(range(len(esc_list)), n_test)

    # Full test samples
    for i in range(n_test):
        texts_data_test.append(txt_pool[test_text_idx[i]])
        img_data_test.append(oc_list[test_img_idx[i]])
        snd_data_test.append(esc_list[test_snd_idx[i]])
        target_test.append(t)
        category_test.append(c)

    # One missing modality in test
    for _ in range(8):
        m = random.choice([0, 1, 2])
        tt = [0] if m == 0 else [random.choice([txt_pool[i] for i in test_text_idx])]
        it = [None] if m == 1 else [random.choice([oc_list[i] for i in test_img_idx])]
        st = [None] if m == 2 else [random.choice([esc_list[i] for i in test_snd_idx])]
        texts_data_test.extend(tt)
        img_data_test.extend(it)
        snd_data_test.extend(st)
        target_test.append(t)
        category_test.append(c)

    # Two missing modalities in test
    for _ in range(8):
        h = random.choice([0, 1, 2])
        tt = [random.choice([txt_pool[i] for i in test_text_idx])] if h == 0 else [0]
        it = [random.choice([oc_list[i] for i in test_img_idx])] if h == 1 else [None]
        st = [random.choice([esc_list[i] for i in test_snd_idx])] if h == 2 else [None]
        texts_data_test.extend(tt)
        img_data_test.extend(it)
        snd_data_test.extend(st)
        target_test.append(t)
        category_test.append(c)

    # Training full triplets
    train_text = [txt_pool[i] for i in range(len(txt_pool)) if i not in test_text_idx]
    train_img = [oc_list[i] for i in range(len(oc_list)) if i not in test_img_idx]
    train_snd = [esc_list[i] for i in range(len(esc_list)) if i not in test_snd_idx]

    if not train_text or not train_img or not train_snd:
        print(f"[WARN] Not enough training pool for class {ok} ({c}) (txt={len(train_text)}, img={len(train_img)}, snd={len(train_snd)})")
        t += 1
        continue  # このクラスはスキップ
    
    for _ in range(60):
        texts_data_train.append(random.choice(train_text))
        img_data_train.append(random.choice(train_img))
        snd_data_train.append(random.choice(train_snd))
        target_train.append(t)
        category_train.append(c)

    # One missing modality in training
    for _ in range(60):
        m = random.choice([0, 1, 2])
        tt = [0] if m == 0 else [random.choice(train_text)]
        it = [None] if m == 1 else [random.choice(train_img)]
        st = [None] if m == 2 else [random.choice(train_snd)]
        texts_data_train.extend(tt)
        img_data_train.extend(it)
        snd_data_train.extend(st)
        target_train.append(t)
        category_train.append(c)

    # Two missing modalities in training
    for _ in range(60):
        h = random.choice([0, 1, 2])
        tt = [random.choice(train_text)] if h == 0 else [0]
        it = [random.choice(train_img)] if h == 1 else [None]
        st = [random.choice(train_snd)] if h == 2 else [None]
        texts_data_train.extend(tt)
        img_data_train.extend(it)
        snd_data_train.extend(st)
        target_train.append(t)
        category_train.append(c)

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
dataset_train.to_csv(os.path.join(output_dir, "dataset_train_missing.csv"), index=False)

dataset_test = pd.DataFrame({
    "category": category_test,
    "target": target_test,
    "img": img_data_test,
    "snd": snd_data_test,
    "text": texts_data_test
})
dataset_test.to_csv(os.path.join(output_dir, "dataset_test_missing.csv"), index=False)
