
import torch
import numpy as np
import argparse
import os
import random

from Mydataloader import *
from Model import *
from Training import *
from utils import *




parser = argparse.ArgumentParser(description='')
#---------------------------------------------------前处理参数---------------------------------------------------
parser.add_argument('--seed',dest='seed', type=int, default=114514, help='Seed')
parser.add_argument('--batch', dest='batch', type=int, default=32, help='# images in batch')
parser.add_argument('--nepoch', dest='nepoch', type=int, default=1000, help='# of epoch')


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
    transforms.ToTensor(), 
    AddGaussianNoise(0., 1.2)])
    transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()])

    # test_dataset = MyDataset(test_image_dir, transform=transform_test)
    # test_data_loader = DataLoader(test_dataset, args.batch, shuffle=False)
    train_dataset = MyDataset(train_image_dir, 7000, transform=transform)
    train_data_loader = DataLoader(train_dataset, args.batch, shuffle=False)
    val_dataset = MyDataset(val_image_dir, 3000, transform=transform_test)
    val_data_loader = DataLoader(val_dataset, args.batch, shuffle=False)




    #导入模型
    model1 = hidden_layer().to(device)
    model2 = DecisionModel().to(device)
    model3 = ConfidenceModel().to(device)
    all_modules = nn.ModuleList([model1, model2, model3])
    #优化器
    optimizer = {
        'adam': torch.optim.Adam(all_modules.parameters(), 1e-4, betas = (0.9, 0.999)),
        'sgd': torch.optim.SGD(all_modules.parameters(), 1e-4, momentum=0.9, nesterov=True, weight_decay=1e-4)
    }['adam']




    for epoch in range(args.nepoch):
        #训练
        trainloss = train_epoch_meta(epoch, model1, model2, model3, train_data_loader, optimizer,  device)
        print('--------------------------------------------------------------------------------------------------------------------------')
        print('Epoch: {}/{} || train Loss: {:.4}'.format(epoch + 1, args.nepoch, trainloss))
        # #验证
        valloss, acc1, roc1, acc2, roc2, acc3, roc3 = val_epoch_meta(model1, model2, model3, val_data_loader, device)
        print('Epoch: {}/{} || val Loss: {:.4} || acc1: {:.3} || roc1: {:.3} || acc2: {:.3} || roc2: {:.3}|| acc3: {:.3} || roc3: {:.3}'.format(epoch + 1, args.nepoch, valloss, acc1, roc1, acc2, roc2, acc3, roc3))
        if epoch+1 == args.nepoch:
            torch.save(model1.state_dict(), './model/hidden.pt')
            torch.save(model2.state_dict(), './model/decision.pt')
            torch.save(model3.state_dict(), './model/confidece.pt')


#torch.save(model.state_dict(), './model/model.pt')

