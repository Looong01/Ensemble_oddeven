import os
import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, Dataset,TensorDataset
from utils import *

def data_loader(image_dir, batch_size, drop_last=False, shuffle=True):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()])
    data = datasets.ImageFolder(image_dir, transform=transform)
    # 创建数据加载器
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    num_data = len(data)
    
    return loader, num_data



class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['same', 'odd', 'even']
        self.num_classes = len(self.classes)

        # Create a list to store image paths for each class
        self.image_paths = {cls: [] for cls in self.classes}

        # Populate the image paths list
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for filename in os.listdir(cls_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    self.image_paths[cls].append(os.path.join(cls_path, filename))

    def __len__(self):
        return min(len(self.image_paths['same']), len(self.image_paths['odd']), len(self.image_paths['even']))

    def __getitem__(self, idx):
        # Randomly select an odd or even image
        info_category = random.choice(['odd', 'even'])
        info_category_image_path = random.choice(self.image_paths[info_category])

        # Select a same image
        same_image_path = random.choice(self.image_paths['same'])

        # Read images
        info_category_image = Image.open(info_category_image_path).convert("RGB")
        same_image = Image.open(same_image_path).convert("RGB")

        info_category_image = transforms.ToTensor()(info_category_image)
        same_image = transforms.ToTensor()(same_image)

        if self.transform:
            info_category_image = self.transform(info_category_image)
            same_image = self.transform(same_image)

        # Create labels
        category_label = 0 if info_category == "odd" else 1  # 0: odd, 1: even
        info_label = 1 if category_label == 0 or category_label == 1 else 0  # 0: noninfo (same), 1: info (odd/even)

        # Randomly choose the order of images
        idx_info = np.random.randint(0, 2)
        images_all = [info_category_image, same_image] if idx_info == 0 else [same_image, info_category_image]
        category_label = [category_label, 2] if idx_info == 0 else [2, category_label]

        return images_all, category_label, info_label, idx_info
    """
    图片的一共有两个，idx_info与列表的图片顺序对应，idx_info为1则是same放在前面，odd/even在后面
    idx为0就informative放在前面，为1就放在后面
    图片也是一样的
    
    """
