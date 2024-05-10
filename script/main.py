
import torch
import numpy as np
import argparse
import os
import random

from Myloader import data_loader
from Model import *
from Training import *




parser = argparse.ArgumentParser(description='')
#---------------------------------------------------前处理参数---------------------------------------------------
parser.add_argument('--seed',dest='seed', type=int, default=1, help='Seed')
parser.add_argument('--batch', dest='batch', type=int, default=64, help='# images in batch')
parser.add_argument('--nepoch', dest='nepoch', type=int, default=30, help='# of epoch')


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
        'adam': torch.optim.Adam(model.parameters(), 1e-3, betas = (0.9, 0.999)),
        'sgd': torch.optim.SGD(model.parameters(), 1e-3, momentum=0.9, nesterov=True, weight_decay=1e-4)
    }['adam']


train_loss, val_loss, val_acc, val_roc, model = model_training(epochs=args.nepoch, model=model, train_loader=train_dataloader, 
                                                               val_loader=val_dataloader, optimizer=optimizer, 
                                                               device=device, loss_fun=F.binary_cross_entropy_with_logits)

#torch.save(model.state_dict(), './model/model.pt')

