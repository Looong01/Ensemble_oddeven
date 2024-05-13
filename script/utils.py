import torch
import numpy as np
import matplotlib.pyplot as plt

def classify_sigmoid_output(preds):
    # 处理张量输入
    if isinstance(preds, torch.Tensor):
        # 将概率值映射到分类标签
        result = torch.where(preds < 1.3, 0, torch.where(preds > 2/3, 1, 0.5))
    # 处理标量输入
    else:
        result = 0 if preds < 1.3 else (1 if preds > 2/3 else 0.5)
    return result


def target_to_oh(target, class_num = 3):
    NUM_CLASS = class_num  # hard code here, can do partial
    one_hot = torch.eye(NUM_CLASS)[target]
    return one_hot

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot




def show_images_labels(images_all, category_label, info_label, idx_info):
    for i in range(len(idx_info)):
        fig, axs = plt.subplots(1, 2, figsize=(4, 2))  # 创建包含两个子图的图形窗口
        
        if idx_info[i] == 0:
            img_inf = images_all[0][i].permute(1, 2, 0).numpy()
            img_non = images_all[1][i].permute(1, 2, 0).numpy()
            label_inf = category_label[0][i]
            label_non = category_label[1][i]
            info_label_inf = info_label[0][i]
            info_label_non = info_label[1][i]
        else:
            img_inf = images_all[1][i].permute(1, 2, 0).numpy()
            img_non = images_all[0][i].permute(1, 2, 0).numpy()
            label_inf = category_label[1][i]
            label_non = category_label[0][i]
            info_label_inf = info_label[1][i]
            info_label_non = info_label[0][i]
        
        axs[0 if idx_info[i] == 0 else 1].imshow(img_inf, cmap='gray')  # 使用灰度色彩映射显示信息图像
        axs[0 if idx_info[i] == 0 else 1].set_title(f"Informative\nCategory: {label_inf}, Meta: {info_label_inf}")
        axs[0 if idx_info[i] == 0 else 1].axis('off')

        axs[1 if idx_info[i] == 0 else 0].imshow(img_non, cmap='gray')  # 使用灰度色彩映射显示非信息图像
        axs[1 if idx_info[i] == 0 else 0].set_title(f"Non-informative\nCategory: {label_non}, Meta: {info_label_non}")
        axs[1 if idx_info[i] == 0 else 0].axis('off')

        plt.tight_layout()  # 调整子图之间的间距
        plt.show()