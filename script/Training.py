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

def model_training(epochs, model, train_loader, val_loader, optimizer, device, loss_fun):
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
            label = torch.argmax(label, dim=1)

            loss = loss_fun(label_pred, label)  # 不再对标签进行 float() 转换
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
                label_target_pred = torch.argmax(label_pred, dim=1)

                target_label = torch.argmax(label, dim=1)


                loss = loss_fun(label_pred, target_label)  # 不再对标签进行 float() 转换
                val_batch_loss.append(loss.item())

                val_batch_labels.extend(label.cpu().numpy())
                val_batch_preds.extend(label_pred.cpu().numpy())
                val_batch_target_labels.extend(target_label.cpu().numpy())
                val_batch_target_preds.extend(label_target_pred.cpu().numpy())

                # Update tqdm progress bar
                val_progress.set_postfix(val_loss=np.mean(val_batch_loss))

        val_progress.close()

        batch_acc = accuracy_score(val_batch_target_labels, val_batch_target_preds)
        batch_roc = roc_auc_score(val_batch_labels, val_batch_preds, multi_class="ovr", average="macro")

        val_acc[epochi] = batch_acc
        val_roc[epochi] = batch_roc
        val_loss[epochi] = np.mean(val_batch_loss)

        # Print loss, accuracy, and ROC AUC after each epoch
        print(f"Epoch [{epochi+1}/{epochs}], Train Loss: {train_loss[epochi]:.4f}, Val Loss: {val_loss[epochi]:.4f}, Val Acc: {val_acc[epochi]:.4f}, Val ROC AUC: {val_roc[epochi]:.4f}")

    return train_loss, val_loss, val_acc, val_roc, model
