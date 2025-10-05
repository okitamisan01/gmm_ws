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
        if not os.path.isfile(p):
            continue
        m = re.match(r"^(\d+)[_-]", x)
        if not m:
            continue
        try:
            cid = int(m.group(1))
        except ValueError:
            continue
        if cid == ok:
            oc_list.append(p)
    
    esc_list = []
    for x in os.listdir(esc_np_dir):
        p = os.path.join(esc_np_dir, x)
        if not os.path.isfile(p):
            continue
        try:
            esc_id = int(x.split("-")[-1].split(".")[0])
        except Exception:
            continue
        if esc_id == chosen_oc_esc[ok]:
            esc_list.append(p)

    txt_pool = oc_txt_dict.get(ok, [])
    n_test = min(8, len(txt_pool), len(oc_list), len(esc_list))
    if n_test == 0:
        print(f"[WARN] Skip class {ok} ({c}): insufficient data (txt={len(txt_pool)}, img={len(oc_list)}, snd={len(esc_list)})")
        t += 1
        continue

    test_text_idx = random.sample(range(len(txt_pool)), n_test)
    test_img_idx = random.sample(range(len(oc_list)), n_test)
    test_snd_idx = random.sample(range(len(esc_list)), n_test)

    test_text = [txt_pool[i] for i in test_text_idx]
    test_img = [oc_list[i] for i in test_img_idx]
    test_snd = [esc_list[i] for i in test_snd_idx]

    # make compound test data (only full modality)
    for tx, im, sd in zip(test_text, test_img, test_snd):
        texts_data_test.append(tx)
        img_data_test.append(im)
        snd_data_test.append(sd)
        target_test.append(t)
        category_test.append(c)

    # training data (remaining data not in test)
    train_text = [v for i, v in enumerate(txt_pool) if i not in test_text_idx]
    train_img = [v for i, v in enumerate(oc_list) if i not in test_img_idx]
    train_snd = [v for i, v in enumerate(esc_list) if i not in test_snd_idx]

    # randomly sample full triplets
    if train_text and train_img and train_snd:
        for _ in range(60):
            texts_data_train.append(random.choice(train_text))
            img_data_train.append(random.choice(train_img))
            snd_data_train.append(random.choice(train_snd))
            target_train.append(t)
            category_train.append(c)
    else:
        print(f"[WARN] Not enough training pool for class {ok} ({c})")

    t += 1

# 保存先ディレクトリを確保
output_dir = os.path.join(root_dir, "dataset_full")
os.makedirs(output_dir, exist_ok=True)


# save as pandas dataframe
dataset_train = pd.DataFrame({
    "category": category_train,
    "target": target_train,
    "img": img_data_train,
    "snd": snd_data_train,
    "text": texts_data_train
})
dataset_train.to_csv(os.path.join(output_dir, "dataset_train_full.csv"), index=False)

dataset_test = pd.DataFrame({
    "category": category_test,
    "target": target_test,
    "img": img_data_test,
    "snd": snd_data_test,
    "text": texts_data_test
})
dataset_test.to_csv(os.path.join(output_dir, "dataset_test_full.csv"), index=False)
