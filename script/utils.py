import torch
import numpy as np
import matplotlib.pyplot as plt

def classify_sigmoid_output(preds):
    # 处理张量输入
    if isinstance(preds, torch.Tensor):
        # 将概率值映射到分类标签
        result = torch.where(preds < 1/3, 0, torch.where(preds > 2/3, 2, 1))
    # 处理标量输入
    else:
        result = 0 if preds < 1/3 else (2 if preds > 2/3 else 1)
    return result


def target_to_oh(target, class_num = 3):
    NUM_CLASS = class_num  
    one_hot = torch.eye(NUM_CLASS)[target]
    return one_hot


def binary_label_to_onehot(binary_labels):
    """
    将二分类标签转换为one-hot编码的函数
    
    参数:
    binary_labels (Tensor): 包含二分类标签的张量，每个标签应为0或1
    
    返回:
    Tensor: 一个one-hot编码的张量，每行对应于输入张量中的一个标签
    """
    num_samples = binary_labels.size(0)
    num_classes = 2  # 二分类
    onehot_labels = torch.zeros(num_samples, num_classes)
    onehot_labels[:, 0] = 1 - binary_labels
    onehot_labels[:, 1] = binary_labels
    return onehot_labels


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def show_images_labels(images_all, category_label, info_label, idx_info, even_num):
    for i in range(len(idx_info)):
        fig, axs = plt.subplots(1, 2, figsize=(4, 2))  # 创建包含两个子图的图形窗口
        
        if idx_info[i] == 0:
            img_inf = images_all[0][i].permute(1, 2, 0).numpy()
            img_non = images_all[1][i].permute(1, 2, 0).numpy()
            label_inf = category_label[0][i]
            label_non = category_label[1][i]
            info_even_num = even_num[0][i]
            non_even_num = even_num[1][i]
        else:
            img_inf = images_all[1][i].permute(1, 2, 0).numpy()
            img_non = images_all[0][i].permute(1, 2, 0).numpy()
            label_inf = category_label[1][i]
            label_non = category_label[0][i]
            info_even_num = even_num[1][i]
            non_even_num = even_num[0][i]
        info_label_idx = info_label[i]
        
        axs[0 if idx_info[i] == 0 else 1].imshow(img_inf, cmap='gray')  # 使用灰度色彩映射显示信息图像
        axs[0 if idx_info[i] == 0 else 1].set_title(f"Informative\nCategory: {label_inf}, Meta: {info_label_idx}\nEven_n:{info_even_num}")
        axs[0 if idx_info[i] == 0 else 1].axis('off')

        axs[1 if idx_info[i] == 0 else 0].imshow(img_non, cmap='gray')  # 使用灰度色彩映射显示非信息图像
        axs[1 if idx_info[i] == 0 else 0].set_title(f"Non-informative\nCategory: {label_non}, Meta: {info_label_idx}\nEven_n:{non_even_num}")
        axs[1 if idx_info[i] == 0 else 0].axis('off')

        plt.tight_layout()  # 调整子图之间的间距
        # 如果是第一个试验，则保存图像
        plt.show()
