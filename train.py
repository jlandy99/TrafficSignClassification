import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset

from helper import train, test, cal_accuracy
from config import NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, CRITERION

def train_model(net, train_loader, val_loader):
    optimizer = optim.Adam(net.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print('\nStart training')

    train_loss_history = []
    val_loss_history = []

    train_acc_history = []
    val_acc_history = []

    for epoch in range(NUM_EPOCHS):
        print('-----------------Epoch = %d-----------------' % (epoch+1))

        # train_loss = train(train_loader, net, criterion, optimizer, device, epoch+1)
        train_loss = train(train_loader, net, optimizer, epoch+1)

        # train_acc = cal_accuracy(train_loader, net, criterion, device)
        train_acc = cal_accuracy(train_loader, net)

        print('Train accuracy: ' + str(train_acc))

        print('Validation loss: ')

        # val_loss = test(val_loader, net, criterion, device)
        val_loss = test(val_loader, net)

        # val_acc = cal_accuracy(train_loader, net, criterion, device)
        val_acc = cal_accuracy(val_loader, net)

        print('Validation accuracy: ' + str(val_acc))

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

    return train_loss_history, val_loss_history, train_acc_history, val_acc_history

