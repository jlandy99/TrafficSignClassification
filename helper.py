import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    # true positives for each class
    correct_from_class = np.zeros(N_CLASS)

    # false negatives from each class
    incorrect_from_class = np.zeros(N_CLASS)

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
                incorrect_from_class[labels] += (y_pred != labels)


        if test_set:
            # stacked bar chart for totals
            fig, ax = plt.subplots()

            x_axis = np.arange(N_CLASS)

            ax.bar(x_axis, correct_from_class, label='Correct')
            ax.bar(x_axis, incorrect_from_class, bottom=correct_from_class, label='Incorrect')

            ax.set_ylabel('Number of Samples')
            ax.set_xlabel('Class ID')
            ax.set_title('Model Performance on Test Set by Class')
            ax.legend()
            plt.savefig('plots/test_set_performance_totals.png')

            # calculate accuracy and replot
            plt.clf()
            fig, ax = plt.subplots()

            accuracy_per_class = correct_from_class / (correct_from_class + incorrect_from_class)
            ax.bar(x_axis, accuracy_per_class)
            ax.set_ylabel('Accuracy (%)')
            ax.set_xlabel('Class ID')
            ax.set_title('Model Accuracy on Test Set by Class')
            ax.legend()

            plt.ylim(bottom=0.50)
            plt.savefig('plots/test_set_accuracy.png')

            # calculate statistcs
            print('\nTest Set Accuracy Statistics')
            print('\nMedian Class Accuracy:\t', "%.4f" % np.median(accuracy_per_class))
            print('Mean Class Accuracy:\t', "%.4f" % np.average(accuracy_per_class))
            print('Stddev Class Accuracy:\t', "%.4f" % np.std(accuracy_per_class))
            print('Min Class Accuracy:\t', "%.4f" % np.amin(accuracy_per_class))
            print('Max Class Accuracy:\t', "%.4f" % np.amax(accuracy_per_class))


        return correct / count
