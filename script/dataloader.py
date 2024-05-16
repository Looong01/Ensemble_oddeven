# %%
import pandas as pd
import numpy as np
import os
from glob import glob
from collections import Counter
from sklearn.utils import shuffle

# %%
train_path = "../data/train"

# %%
odd = glob(os.path.join(train_path, "odd", '*jpg'))
even = glob(os.path.join(train_path, "even", '*jpg'))
same = glob(os.path.join(train_path, "same", '*jpg'))

# %%
n_trails = 8000
meta_labels = np.concatenate([[0]*int(n_trails/2), [1]*int(n_trails/2)])
meta_labels = shuffle(meta_labels)

info_paths = shuffle(odd + even)
img1 = []
img2 = []

for idx_trial in range(n_trails):
    meta_label = meta_labels[idx_trial]
    if meta_label == 0:#informative
        img1.append(np.random.choice(info_paths, size=1)[0])
        img2.append(np.random.choice(same, size=1)[0])
    else:
        img2.append(np.random.choice(info_paths, size=1)[0])
        img1.append(np.random.choice(same, size=1)[0])
        


# %%

df = pd.DataFrame({"meta_labels" : meta_labels, "img1_path" : img1, "img2_path" : img2})

# %%
df

# %%
label_map = {"odd": 0, "even": 1, "same" : 0.5}
df['img1_labels'] = df['img1_path'].apply(lambda x: label_map[x.split('/')[-2]])

df['img2_labels'] = df['img2_path'].apply(lambda x: label_map[x.split('/')[-2]])
# %%
def extract_even_num(path):
    return path.split('/')[-1].split('_')[0]

df['even_num'] = df['img2_path'].apply(extract_even_num)
# %%
# img1, img2, label1, label2, meta_label