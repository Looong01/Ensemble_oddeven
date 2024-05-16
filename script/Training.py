import torch
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from utils import *

def train_epoch(epoch, model, train_loader, optimizer,device):
    model.train()
    num_iter = len(train_loader)
    trainloss = 0
    for i, (data, label) in enumerate(train_loader):
        #传入cuda中
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()

        label_pred = model(data).squeeze()
        #二分类损伤函数F.binary_cross_entropy_with_logits
        loss = F.binary_cross_entropy_with_logits(label_pred, label.float())

        loss.backward()
        optimizer.step()
        trainloss += loss
    trainloss = trainloss/num_iter
    return trainloss


def val_epoch(model, val_loader, device):
    model.train()
    num_iter = len(val_loader)
    valloss = 0
    all_labels = []
    all_preds = []
    model.eval()
    for i, (data, label) in enumerate(val_loader):
        # 传入cuda中
        data, label = data.to(device), label.to(device)
        with torch.no_grad():
            label_pred = model(data).squeeze()
            loss = F.binary_cross_entropy_with_logits(label_pred, label.float())
            valloss += loss
            #acc
            preds = torch.sigmoid(label_pred)
            preds = (preds > 0.5).float()
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    roc  = roc_auc_score(all_labels, all_preds)
    valloss = valloss / num_iter
    return valloss , acc, roc



def train_epoch_meta(epoch, model1, model2, model3, train_loader, optimizer,device):
    model1.train()
    model2.train()
    model3.train()

    num_iter = len(train_loader)
    trainloss = 0

    for i, (img1, img2, img1_label, img2_label, meta_label, _, _) in enumerate(train_loader):
        #传入cuda中
        img1, img2, img1_label, img2_label, meta_label = img1.to(device), img2.to(device), img1_label.to(device), img2_label.to(device), meta_label.to(device)

        

        optimizer.zero_grad()
        feature1 = model1(img1)#select features
        feature2 = model1(img2)#select features

        combine_features = torch.cat((feature1, feature2), dim=1)#to model 3


        label_pred1 = model2(feature1).squeeze()
        label_pred2 = model2(feature2).squeeze()

        label_pred3 = model3(combine_features).squeeze()

        #二分类损伤函数F.binary_cross_entropy_with_logits
        loss1 = F.binary_cross_entropy_with_logits(label_pred1, img1_label.float())
        loss2 = F.binary_cross_entropy_with_logits(label_pred2, img2_label.float())
        loss3 = F.binary_cross_entropy_with_logits(label_pred3, meta_label.float())
        img_loss = (loss1+loss2)/2.0
        combine_loss = img_loss + loss3


        combine_loss.backward()
        optimizer.step()
        trainloss += combine_loss
    trainloss = trainloss/num_iter
    return trainloss



def val_epoch_meta(model1, model2, model3, val_loader, device):
    model1.train()
    model2.train()
    model3.train()

    num_iter = len(val_loader)
    valloss = 0
    all_labels1 = []
    all_preds1 = []

    all_labels2 = []
    all_preds2 = []

    all_labels3 = []
    all_preds3 = []

    model1.eval()
    model2.eval()
    model3.eval()

    for i, (img1, img2, img1_label, img2_label, meta_label, _, _) in enumerate(val_loader):
        #传入cuda中
        img1, img2, img1_label, img2_label, meta_label = img1.to(device), img2.to(device), img1_label.to(device), img2_label.to(device), meta_label.to(device)
        with torch.no_grad():
            feature1 = model1(img1)#select features
            feature2 = model1(img2)#select features

            combine_features = torch.cat((feature1, feature2), dim=1)#to model 3


            label_pred1 = model2(feature1).squeeze()
            label_pred2 = model2(feature2).squeeze()

            label_pred3 = model3(combine_features).squeeze()

            #二分类损伤函数F.binary_cross_entropy_with_logits
            loss1 = F.binary_cross_entropy_with_logits(label_pred1, img1_label.float())
            loss2 = F.binary_cross_entropy_with_logits(label_pred2, img2_label.float())
            loss3 = F.binary_cross_entropy_with_logits(label_pred3, meta_label.float())
            img_loss = (loss1+loss2)/2.0
            combine_loss = img_loss + loss3
            valloss += combine_loss
            #acc
            preds1 = torch.sigmoid(label_pred1)
            preds1 = (preds1 > 0.5).float()

            preds2 = torch.sigmoid(label_pred2)
            preds2 = (preds2 > 0.5).float()

            preds3 = torch.sigmoid(label_pred3)
            preds3 = (preds3 > 0.5).float()

            all_labels1.extend(img1_label.cpu().numpy())
            all_preds1.extend(preds1.cpu().numpy())

            all_labels2.extend(img2_label.cpu().numpy())
            all_preds2.extend(preds2.cpu().numpy())

            all_labels3.extend(meta_label.cpu().numpy())
            all_preds3.extend(preds3.cpu().numpy())

    indices_to_remove1 = [i for i, label in enumerate(all_labels1) if label == 0.5]
    indices_to_remove2 = [i for i, label in enumerate(all_labels2) if label == 0.5]
    # 删除对应索引位置的元素
    for index in sorted(indices_to_remove1, reverse=True):
        del all_labels1[index]
        del all_preds1[index]

    for index in sorted(indices_to_remove2, reverse=True):
        del all_labels2[index]
        del all_preds2[index]

    acc1 = accuracy_score(all_labels1, all_preds1)
    roc1  = roc_auc_score(all_labels1, all_preds1)
    acc2 = accuracy_score(all_labels2, all_preds2)
    roc2  = roc_auc_score(all_labels2, all_preds2)
    acc3 = accuracy_score(all_labels3, all_preds3)
    roc3  = roc_auc_score(all_labels3, all_preds3)
    valloss = valloss / num_iter
    return valloss , acc1, roc1, acc2, roc2, acc3, roc3
