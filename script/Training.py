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




import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

def model_training(epochs, model, train_loader, val_loader, optimizer, device):
    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    val_acc = np.zeros(epochs)
    val_roc = np.zeros(epochs)

    for epochi in range(epochs):
        model.train()
        num_train_iter = len(train_loader)
        train_batch_loss = []

        # Create tqdm progress bar for training set
        train_progress = tqdm(train_loader, desc=f'Epoch {epochi+1}/{epochs} (Train)')

        for i, (data, label) in enumerate(train_progress):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            label_pred = model(data)
            # label = torch.argmax(label, dim=1)

            loss = F.cross_entropy(label_pred, label.long())  # 不再对标签进行 float() 转换
            loss.backward()
            optimizer.step()
            train_batch_loss.append(loss.item())

            # Update tqdm progress bar
            train_progress.set_postfix(train_loss=np.mean(train_batch_loss))

        train_progress.close()

        train_loss[epochi] = np.mean(train_batch_loss)

        model.eval()
        val_batch_loss = []
        val_batch_labels = []
        val_batch_target_labels = []
        val_batch_preds = []
        val_batch_target_preds = []

        # Create tqdm progress bar for validation set
        val_progress = tqdm(val_loader, desc=f'Epoch {epochi+1}/{epochs} (Val)')

        with torch.no_grad():  # Ensure no gradient calculation during validation
            for i, (data, label) in enumerate(val_progress):
                data, label = data.to(device), label.to(device)
                label_pred = model(data)
                # label_target_pred = torch.argmax(label_pred, dim=1)

                # target_label = torch.argmax(label, dim=1)


                loss = F.cross_entropy(label_pred, label.long())  # 不再对标签进行 float() 转换
                val_batch_loss.append(loss.item())

                val_batch_labels.extend(label.cpu().numpy())
                val_batch_preds.extend(label_pred.cpu().numpy())

                # Update tqdm progress bar
                val_progress.set_postfix(val_loss=np.mean(val_batch_loss))

        val_progress.close()
        #将val_batch_labels和val_batch_preds转换为numpy数组
        val_batch_labels = np.array(val_batch_labels)
        val_batch_preds = np.array(val_batch_preds)

        val_preds = np.argmax(val_batch_preds, axis=1)
        #将val_batch_labels转换为one-hot编码
        val_label_onehot = dense_to_one_hot(val_batch_labels, 3)

        batch_acc = accuracy_score(val_batch_labels, val_preds)
        batch_roc = roc_auc_score(val_label_onehot, val_batch_preds, multi_class="ovr", average="macro")

        val_acc[epochi] = batch_acc
        val_roc[epochi] = batch_roc
        val_loss[epochi] = np.mean(val_batch_loss)

        # Print loss, accuracy, and ROC AUC after each epoch
        print(f"Epoch [{epochi+1}/{epochs}], Train Loss: {train_loss[epochi]:.4f}, Val Loss: {val_loss[epochi]:.4f}, Val Acc: {val_acc[epochi]:.4f}, Val ROC AUC: {val_roc[epochi]:.4f}")

    return train_loss, val_loss, val_acc, val_roc, model




def train_epoch_meta(epoch, model, train_loader, optimizer,device):
    model.train()
    num_iter = len(train_loader)
    trainloss = 0
    for i, (images_all, category_label, info_label, idx_info, even_num) in enumerate(train_loader):
        #传入cuda中
        data1, data2, label1, label2, label3 = images_all[0].to(device), images_all[1].to(device), category_label[0].to(device), category_label[1].to(device), info_label.to(device)

        optimizer.zero_grad()

        label_pred1 = model(data1).squeeze()
        label_pred2 = model(data2).squeeze()
        #二分类损伤函数F.binary_cross_entropy_with_logits
        loss1 = F.binary_cross_entropy_with_logits(label_pred1, label1.float())

        loss2 = F.binary_cross_entropy_with_logits(label_pred2, label2.float())

        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        trainloss += loss
    trainloss = trainloss/num_iter
    return trainloss


def val_epoch_meta(model, val_loader, device):
    model.train()
    num_iter = len(val_loader)
    valloss = 0
    all_labels1 = []
    all_preds1 = []
    all_pred_proba1 = []
    all_labels1_int = []
    
    all_labels2 = []
    all_preds2 = []
    all_pred_proba2 = []
    all_labels2_int = []


    model.eval()
    for i, (images_all, category_label, info_label, idx_info, even_num) in enumerate(val_loader):
        # 传入cuda中
        data1, data2, label1, label2, label3 = images_all[0].to(device), images_all[1].to(device), category_label[0].to(device), category_label[1].to(device), info_label.to(device)
        with torch.no_grad():
            label_pred1 = model(data1).squeeze()
            label_pred2 = model(data2).squeeze()
            #二分类损伤函数F.binary_cross_entropy_with_logits
            loss1 = F.binary_cross_entropy_with_logits(label_pred1, label1.float())

            loss2 = F.binary_cross_entropy_with_logits(label_pred2, label2.float())

            loss = loss1 + loss2
            #acc
            valloss += loss
            #acc
            label1_int = classify_sigmoid_output(label1)
            preds_prob1 = torch.sigmoid(label_pred1)
            preds1 = classify_sigmoid_output(preds_prob1)
            all_labels1.extend(label1.cpu().numpy())
            all_preds1.extend(preds1.cpu().numpy())
            all_pred_proba1.extend(preds_prob1.cpu().numpy())
            all_labels1_int.extend(label1_int.cpu().numpy())

            label2_int = classify_sigmoid_output(label2)
            preds_prob2 = torch.sigmoid(label_pred2)
            preds2 = classify_sigmoid_output(preds_prob2)
            all_labels2.extend(label2.cpu().numpy())
            all_preds2.extend(preds2.cpu().numpy())
            all_pred_proba2.extend(preds_prob2.cpu().numpy())
            all_labels2_int.extend(label2_int.cpu().numpy())


    acc1 = accuracy_score(all_labels1_int, all_preds1)
    label1_oh = dense_to_one_hot(np.array(all_labels1_int), 3)
    pred1_oh = dense_to_one_hot(np.array(all_preds1),3)
    roc1  = roc_auc_score(label1_oh, pred1_oh)
    acc2 = accuracy_score(all_labels2_int, all_preds2)
    label2_oh = dense_to_one_hot(np.array(all_labels2_int),3)
    pred2_oh = dense_to_one_hot(np.array(all_preds2),3)
    roc2  = roc_auc_score(label2_oh, pred2_oh)
    valloss = valloss / num_iter
    return valloss, acc1, roc1, acc2, roc2