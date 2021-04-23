import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_class_dist_and_stats(y, n_class, filename):
    fig, ax = plt.subplots()

    ax.hist(y, n_class)
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Distribution of Samples by Class')
    plt.savefig(filename)

    bincount = np.bincount(y.astype(np.uint8))
    print('\nMedian:\t', np.median(bincount))
    print('Mean:\t', np.average(bincount))
    print('Stddev:\t', np.std(bincount))
    print('Min:\t', np.amin(bincount))
    print('Max:\t', np.amax(bincount))


# X_train and Y_train are the training portion (not including validaiton)
# validation should have been already split off
# def train(train_loader, net, criterion, optimizer, device, epoch):
def train(train_loader, net, criterion, optimizer, epoch):
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
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    cnt += 1

    #   end = time.time()
    running_loss /= cnt
    #   print('\n [epoch %d] loss: %.3f elapsed time %.3f' %
    #         (epoch, running_loss, end-start))

    print('\n [epoch %d] loss: %.3f' %(epoch, running_loss))

    return running_loss


# def test(test_loader, net, criterion, device):
def test(test_loader, net, criterion):
    losses = 0.
    cnt = 0
    with torch.no_grad():

    net = net.eval()

    for images, labels in tqdm(test_loader):

    #   images = images.to(device)
    #   labels = labels.to(device)
      output = net(images)
      loss = criterion(output, labels)
      losses += loss.item()
      cnt += 1
      
    print('\n',losses / cnt)
    return (losses/cnt)


# def cal_accuracy(test_loader, net, criterion, device):
def cal_accuracy(test_loader, net, criterion):

    count = 0.0
    correct = 0.0

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
      
    return correct / count


def plot_history(train_history, val_history, filename, loss=False):

    plt.figure()

    x_axis = np.arange(len(train_history))
    plt.plot(x_axis, train_history, label="Training")
    plt.plot(x_axis, val_history, label="Validation")

    plt.xticks(x_axis)
    plt.xlabel('Epoch')

    if loss:
    plt.ylabel('Loss')
    else:
    plt.ylabel('Accuracy')

    plt.legend()

    plt.savefig(filename)
