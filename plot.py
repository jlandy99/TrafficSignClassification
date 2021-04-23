from helper import cal_accuracy
import matplotlib.pyplot as plt
import numpy as np

def plot_history(train_history, val_history, filepath, loss=False):

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

    plt.savefig(filepath)


def plot(net, train_loss_history, val_loss_history, train_acc_history, val_acc_history):
    plot_history(train_loss_history, val_loss_history, 'plots/train_loss.png', loss=True)
    plot_history(train_acc_history, val_acc_history, 'plots/train_acc.png')


    print('\nFinished Training, Testing on test set')


def plot_class_dist_and_stats(y, n_class, filepath):
    fig, ax = plt.subplots()

    ax.hist(y, n_class)
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Distribution of Samples by Class')
    plt.savefig(filepath)

    bincount = np.bincount(y.astype(np.uint8))
    print('\nMedian:\t', np.median(bincount))
    print('Mean:\t', np.average(bincount))
    print('Stddev:\t', np.std(bincount))
    print('Min:\t', np.amin(bincount))
    print('Max:\t', np.amax(bincount))

