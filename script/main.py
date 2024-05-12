
import torch
import numpy as np
import argparse
import os
import random

from Myloader import *
from Model import *
from Training import *
from utils import *




parser = argparse.ArgumentParser(description='')
#---------------------------------------------------前处理参数---------------------------------------------------
parser.add_argument('--seed',dest='seed', type=int, default=114514, help='Seed')
parser.add_argument('--batch', dest='batch', type=int, default=32, help='# images in batch')
parser.add_argument('--nepoch', dest='nepoch', type=int, default=100, help='# of epoch')


args = parser.parse_args()




if __name__ == '__main__':
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(os.getcwd())

    #设置数据集路径
    train_image_dir = './data/train/'
    val_image_dir = './data/validation/'
    test_image_dir = './data/test/'

    custom_class_to_idx = {'even': 0, 'odd': 1, 'same': 2}
    #读取数据集
    train_dataloader, num_train = data_loader(train_image_dir, args.batch, drop_last=False, shuffle=True)
    val_dataloader, num_val = data_loader(val_image_dir, args.batch, drop_last=False, shuffle=True)
    test_dataloader, num_test = data_loader(test_image_dir, args.batch, drop_last=False, shuffle=False)
    #打印数据集大小
    print('train dataset size：', num_train, 'validatin dataset size：', num_val, 'test dataset size：', num_test)

    #导入模型
    model = MyResNet(dropout_prob=0.5).to(device)
    #优化器
    optimizer = {
        'adam': torch.optim.Adam(model.parameters(), 1e-4, betas = (0.9, 0.999)),
        'sgd': torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, nesterov=True, weight_decay=1e-4)
    }['adam']


    train_loss, val_loss, val_acc, val_roc, model = model_training(epochs=args.nepoch, model=model, train_loader=train_dataloader,
                                                               val_loader=test_dataloader, optimizer=optimizer, device=device)

#torch.save(model.state_dict(), './model/model.pt')

