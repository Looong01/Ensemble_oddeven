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

def show_images_labels(img1, img2, img1_label, img2_label, meta_label):
    for i in range(img1.shape[0]):
        fig, axs = plt.subplots(1, 2, figsize=(4, 2))  # 创建包含两个子图的图形窗口
        axs[0].imshow(img1[i].squeeze(), cmap='gray')
        axs[0].set_title(f"Category:{int(img1_label[i])}")
        axs[1].imshow(img2[i].squeeze(), cmap='gray')
        axs[1].set_title(f"Category:{int(img2_label[i])}")

        fig.suptitle(f"Meta: {int(meta_label[i])}")  # 在两个子图中间添加标题
