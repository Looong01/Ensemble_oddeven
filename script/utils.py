import torch

def classify_sigmoid_output(preds):
    # 处理张量输入
    if isinstance(preds, torch.Tensor):
        # 将概率值映射到分类标签
        result = torch.where(preds < 1.3, 0, torch.where(preds > 2/3, 1, 0.5))
    # 处理标量输入
    else:
        result = 0 if preds < 1.3 else (1 if preds > 2/3 else 0.5)
    return result


def target_to_oh(target):
    NUM_CLASS = 3  # hard code here, can do partial
    one_hot = torch.eye(NUM_CLASS)[target]
    return one_hot