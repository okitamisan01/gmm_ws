import difflib
import os
import pandas as pd

base_dir = os.getcwd()
oc256_dir = os.path.join(base_dir, "data_raw","Caltech256", "256_ObjectCategories")
oc256_dirs = [os.path.join(oc256_dir, x) for x in os.listdir(oc256_dir)]
oc256_files = [[os.path.join(x,y) for y in os.listdir(x)] for x in oc256_dirs]
# make a dict of label num to category name
oc_class_dict = {int(x.split("/")[-1].split(".")[0]): x.split("/")[-1].split(".")[1] for x in oc256_dirs}
esc_dir = os.path.join(base_dir, "data_raw", "ESC-50")
meta_file = os.path.join(esc_dir, "meta/esc50.csv")
audio_dir = os.path.join(esc_dir, "audio/")
# load metadata
meta_data = pd.read_csv(meta_file)

# get data size
data_size = meta_data.shape
print(data_size)

# make a dict of label num to category name
esc_class_dict = {}
for i in range(data_size[0]):
    if meta_data.loc[i,"target"] not in esc_class_dict.keys():
        esc_class_dict[meta_data.loc[i,"target"]] = meta_data.loc[i,"category"]


# look for label that spells similar from the image and sound to make combinations
close_values = []
for ok,ov in oc_class_dict.items():
    for ek,ev in esc_class_dict.items():
        r = difflib.SequenceMatcher(None, ov, ev).ratio()
        if r > 0.6: # category names as close as 60% is considered close
            close_values.append([ok,ov,ek,ev,r])
for l in close_values:
    print(l)

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