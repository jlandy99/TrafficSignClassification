from helper import plot_history, cal_accuracy


def plot(net, test_loader, train_loss_history, val_loss_history, train_acc_history, val_acc_history):
    plot_history(train_loss_history, val_loss_history, 'train_loss.png', loss=True)
    plot_history(train_acc_history, val_acc_history, 'train_acc.png')


    print('\nFinished Training, Testing on test set')

    # print('\nFinal Test Set Accuracy:', cal_accuracy(test_loader, net, criterion, device))
    print('\nFinal Test Set Accuracy:', cal_accuracy(test_loader, net))


