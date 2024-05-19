
import os
from glob import glob
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils import shuffle
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, root, n_trials, transform=None):
        self.root = root
        self.n_trials = n_trials
        self.transform = transform

        odd = glob(os.path.join(root, "odd", '*.jpg'))
        even = glob(os.path.join(root, "even", '*.jpg'))
        same = glob(os.path.join(root, "same", '*.jpg'))

        meta_labels = np.concatenate([[0]*int(self.n_trials/2), [1]*int(self.n_trials/2)])
        meta_labels = shuffle(meta_labels)
        info_paths = shuffle(odd + even)
        img1 = []
        img2 = []

        for idx_trial in range(self.n_trials):
            meta_label = meta_labels[idx_trial]
            if meta_label == 0:  # informative
                img1.append(np.random.choice(info_paths, size=1)[0])
                img2.append(np.random.choice(same, size=1)[0])
            else:
                img2.append(np.random.choice(info_paths, size=1)[0])
                img1.append(np.random.choice(same, size=1)[0])

        df = pd.DataFrame({"meta_labels": meta_labels, "img1_path": img1, "img2_path": img2})
        label_map = {"odd": 0, "even": 1, "same": 0.5}
        def extract_even_num(path):
            return path.split('/')[-1].split('_')[0]

        df['even_num1'] = df['img1_path'].apply(extract_even_num)
        df['even_num2'] = df['img2_path'].apply(extract_even_num)
        df['img1_labels'] = df['img1_path'].apply(lambda x: label_map[x.split(os.path.sep)[-2]])
        df['img2_labels'] = df['img2_path'].apply(lambda x: label_map[x.split(os.path.sep)[-2]])

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img1_path = self.df.iloc[idx]['img1_path']
        img2_path = self.df.iloc[idx]['img2_path']
        img1_label = self.df.iloc[idx]['img1_labels']
        img2_label = self.df.iloc[idx]['img2_labels']
        meta_label = self.df.iloc[idx]['meta_labels']
        even_num1 = self.df.iloc[idx]['even_num1']
        even_num2 = self.df.iloc[idx]['even_num2']

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        img1 = transforms.ToTensor()(img1)
        img2 = transforms.ToTensor()(img2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, img1_label, img2_label, meta_label, even_num1, even_num2


