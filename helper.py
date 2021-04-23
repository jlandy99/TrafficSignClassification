import torch
import numpy as np
from tqdm import tqdm

from config import CRITERION, N_CLASS

# X_train and Y_train are the training portion (not including validaiton)
# validation should have been already split off
# def train(train_loader, net, criterion, optimizer, device, epoch):
def train(train_loader, net, optimizer, epoch):
    #   start = time.time()
    running_loss = 0.0
    cnt = 0
    net = net.train()

    # tqdm just prints a dynamically updating progress bar to the console
    for images, labels in tqdm(train_loader):

        # images = images.to(device)
        # labels = labels.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = CRITERION(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        cnt += 1

    #   end = time.time()
    running_loss /= cnt
    #   print('\n [epoch %d] loss: %.3f elapsed time %.3f' %
    #         (epoch, running_loss, end-start))

    return running_loss


# def test(test_loader, net, criterion, device):
def test(test_loader, net):

    losses = 0.
    cnt = 0

    with torch.no_grad():

        net = net.eval()

        for images, labels in tqdm(test_loader):

            #   images = images.to(device)
            #   labels = labels.to(device)
            output = net(images)
            loss = CRITERION(output, labels)
            losses += loss.item()
            cnt += 1

    return (losses/cnt)


# def cal_accuracy(test_loader, net, criterion, device):
# When test set is true we know we are not using this function for test set so we can then
# keep track of correct/incorrect for each class for plotting
def cal_accuracy(test_loader, net, test_set=False):

    count = 0.0
    correct = 0.0

    # only used for test set
    correct_from_class = np.zeros(N_CLASS)
    total_from_class = np.zeros(N_CLASS)

    with torch.no_grad():

        net = net.eval()

        for images, labels in tqdm(test_loader):

            # images = images.to(device)
            # labels = labels.to(device).cpu().numpy().reshape(-1, 1)
            labels = labels.cpu().numpy().reshape(-1, 1)

            output = net(images).cpu().numpy()
            y_pred = np.argmax(output, axis=1).reshape(-1, 1)

            # add total number of samples tested (batch size)
            count += output.shape[0]

            # add correctly identified samples
            correct += np.sum(y_pred == labels)

            #  test set batch size is 1 so this should work ebcause each output only has
            #  one sample
            if test_set:
                correct_from_class[labels] += (y_pred == labels)
                total_from_class[labels] += 1

        print(correct_from_class)
        print(total_from_class)

        return correct / count
