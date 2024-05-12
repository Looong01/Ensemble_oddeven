import torch
import numpy as np

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