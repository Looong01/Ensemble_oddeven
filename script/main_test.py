
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

    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()])

    test_dataset = CustomDataset(test_image_dir, transform=transform)
    test_data_loader = DataLoader(test_dataset, args.batch, shuffle=False)
    train_dataset = CustomDataset(train_image_dir, transform=transform)
    train_data_loader = DataLoader(train_dataset, args.batch, shuffle=False)
    val_dataset = CustomDataset(val_image_dir, transform=transform)
    val_data_loader = DataLoader(val_dataset, args.batch, shuffle=False)




    #导入模型
    model = CNN().to(device)
    #优化器
    optimizer = {
        'adam': torch.optim.Adam(model.parameters(), 1e-4, betas = (0.9, 0.999)),
        'sgd': torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, nesterov=True, weight_decay=1e-4)
    }['adam']


    for epoch in range(args.nepoch):
        #训练
        trainloss = train_epoch_meta(epoch, model, train_data_loader, optimizer,device)
        print('----------------------------------------------------------------------------------')
        print('Epoch: {}/{} || train Loss: {:.4}'.format(epoch + 1, args.nepoch, trainloss))
        #验证
        valloss, acc1, roc1, acc2, roc2 = val_epoch_meta(model, val_data_loader, device)
        print('Epoch: {}/{} || val Loss: {:.4} || acc1: {:.4} || roc1: {:.4} || acc2: {:.4} || roc2: {:.4}'.format(epoch + 1, args.nepoch, valloss, acc1, roc1, acc2, roc2))


#torch.save(model.state_dict(), './model/model.pt')

