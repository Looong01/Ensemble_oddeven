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




def train_epoch_meta(epoch, model1, model2, train_loader, optimizer, device):
    model1.train()
    model2.train()
    num_iter = len(train_loader)
    trainloss = 0
    for i, (images_all, category_label, info_label, idx_info, even_num) in enumerate(train_loader):
        #传入cuda中
        data1, data2, label1, label2, label3 = images_all[0].to(device), images_all[1].to(device), category_label[0].to(device), category_label[1].to(device), info_label.to(device)

        optimizer.zero_grad()

        label_pred1 = model1(data1)
        label_pred2 = model1(data2)
        combined_output = torch.cat((label_pred1, label_pred2), dim=1)
        label_pred3 = model2(combined_output).squeeze()
        #二分类损伤函数F.binary_cross_entropy_with_logits
        loss1 = F.cross_entropy(label_pred1, label1.long())

        loss2 = F.cross_entropy(label_pred2, label2.long())

        loss3 = F.binary_cross_entropy_with_logits(label_pred3, label3.float())

        img_loss = (loss1 + loss2)/2.0
        loss = img_loss + loss3
        loss.backward()
        optimizer.step()
        trainloss += loss
    trainloss = trainloss/num_iter
    return trainloss


def val_epoch_meta(model1, model2, val_loader, device):
    model1.train()
    model2.train()
    num_iter = len(val_loader)
    valloss = 0

    all_labels1 = []
    all_preds1 = []
    
    all_labels2 = []
    all_preds2 = []

    all_labels3 = []
    all_preds3 = []
    all_preds3_sig = []



    model1.eval()
    model2.eval()
    for i, (images_all, category_label, info_label, idx_info, even_num) in enumerate(val_loader):
        # 传入cuda中
        data1, data2, label1, label2, label3 = images_all[0].to(device), images_all[1].to(device), category_label[0].to(device), category_label[1].to(device), info_label.to(device)
        with torch.no_grad():
            label_pred1 = model1(data1)
            label_pred2 = model1(data2)
            combined_output = torch.cat((label_pred1, label_pred2), dim=1)
            label_pred3 = model2(combined_output).squeeze()
            label_pred3_sig = torch.sigmoid(label_pred3)


            #二分类损伤函数F.binary_cross_entropy_with_logits
            loss1 = F.cross_entropy(label_pred1, label1.long())

            loss2 = F.cross_entropy(label_pred2, label2.long())

            loss3 = F.binary_cross_entropy_with_logits(label_pred3, label3.float())

            img_loss = (loss1 + loss2)/2.0
            loss = img_loss + loss3
            #acc
            valloss += loss
            #acc
            
            all_labels1.extend(label1.cpu().numpy())#
            all_preds1.extend(label_pred1.cpu().numpy())


            all_labels2.extend(label2.cpu().numpy())
            all_preds2.extend(label_pred2.cpu().numpy())

            all_labels3.extend(label3.cpu().numpy())
            all_preds3.extend(label_pred3.cpu().numpy())
            all_preds3_sig.extend(label_pred3_sig.cpu().numpy())


    all_batch_labels1 = np.array(all_labels1)
    all_batch_labels2 = np.array(all_labels2)
    all_batch_labels3 = np.array(all_labels3)

    all_batch_preds1 = np.array(all_preds1)
    all_batch_preds2 = np.array(all_preds2)
    all_preds3_sig = np.array(all_preds3_sig)

    val_preds_idx1 = np.argmax(all_batch_preds1, axis=1)
    val_preds_idx2 = np.argmax(all_batch_preds2, axis=1)
    val_preds_idx3 = (all_preds3_sig > 0.5).astype(float)

    val_label_onehot1 = dense_to_one_hot(all_batch_labels1, 3)
    val_label_onehot2 = dense_to_one_hot(all_batch_labels2, 3)
    
    acc1 = accuracy_score(all_batch_labels1, val_preds_idx1)
    roc1  = roc_auc_score(val_label_onehot1, all_batch_preds1, multi_class="ovr", average="macro")
   
    acc2 = accuracy_score(all_batch_labels2, val_preds_idx2)
    roc2  = roc_auc_score(val_label_onehot2, all_batch_preds2, multi_class="ovr", average="macro")

    acc3 = accuracy_score(all_batch_labels3, val_preds_idx3)
    roc3  = roc_auc_score(all_batch_labels3, val_preds_idx3)
    valloss = valloss / num_iter
    return valloss, acc1, roc1, acc2, roc2, acc3, roc3