import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from deeplob.DeepLOBModel import deepLob, Dataset
import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
import numpy as np
from visdom import Visdom

batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):
    '''

    :param model:
    :param criterion:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param epochs:
    :return:
    '''
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0

    # viz = Visdom()
    # viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
    # viz.line([0.], [0.], win='test_loss', opts=dict(title='tess loss'))


    for it in tqdm(range(epochs)):
        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            # move to GPU
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # backward and optimize
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            # viz.line([loss.item()], [global_step], win='train_loss', update='append')
        train_loss = np.mean(train_loss)

        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)
        # viz.line([test_loss], [global_step], win='test', update='append')

        # save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        if test_loss < best_test_loss:
            torch.save(model, '/home/yuan/jam/src/model/best_val_model_pytorch.pt')
            best_test_loss = test_loss
            best_test_epoch = it
            print("model saved")

        dt = datetime.now() - t0
        print(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, \
                  Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')

    return train_losses, test_losses

if __name__ == '__main__':



#     data
    dec_data = np.loadtxt('/home/yuan/jam/data/FI-2010/Train_Dst_NoAuction_DecPre_CF_7.txt')
    dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.8))]
    dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)):]

    dec_test1 = np.loadtxt('/home/yuan/jam/data/FI-2010/Test_Dst_NoAuction_DecPre_CF_7.txt')
    dec_test2 = np.loadtxt('/home/yuan/jam/data/FI-2010/Test_Dst_NoAuction_DecPre_CF_8.txt')
    dec_test3 = np.loadtxt('/home/yuan/jam/data/FI-2010/Test_Dst_NoAuction_DecPre_CF_9.txt')
    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))
    # print(dec_train.shape, dec_val.shape, dec_test.shape)

    dataset_train = Dataset(data=dec_train, k=4, num_classes=3, T=100)
    dataset_val = Dataset(data=dec_val, k=4, num_classes=3, T=100)
    dataset_test = Dataset(data=dec_test, k=4, num_classes=3, T=100)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    # print(dataset_train.x.shape, dataset_train.y.shape)

    # train model
    model = deepLob(yLen=dataset_train.num_classes, device=device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_losses, val_losses = batch_gd(model, criterion, optimizer, train_loader, val_loader, epochs=50)


    # model test

    model = torch.load('/home/yuan/jam/src/model/best_val_model_pytorch.pt')
    n_correct = 0.
    n_total = 0.
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

        outputs = model(inputs)

        _, predictions = torch.max(outputs, 1)

        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]
    test_acc = n_correct / n_total
    print(f"Test acc: {test_acc:.4f}")

    all_targets = []
    all_predictions = []

    for inputs, targets in test_loader:
        # Move to GPU
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

        # Forward pass
        outputs = model(inputs)

        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    # %%

    print('accuracy_score:', accuracy_score(all_targets, all_predictions))
    print(classification_report(all_targets, all_predictions, digits=4))
