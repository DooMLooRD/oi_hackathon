import torch
import torchvision
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def extract_conf_matrix(conf_matrix, n_classes, classes):
    TP = conf_matrix.diag()
    for c in range(n_classes):
        idx = torch.ones(n_classes, dtype=bool)
        idx[c] = 0
        TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum()
        FP = conf_matrix[c, idx].sum()
        FN = conf_matrix[idx, c].sum()

        sensitivity = (TP[c] / (TP[c]+FN))
        specificity = (TN / (TN+FP))

        print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
            classes[c], TP[c], TN, FP, FN))
        print('Sensitivity = {}'.format(sensitivity))
        print('Specificity = {}'.format(specificity))


def save_plots(train_history, val_history, num_epochs, ticks, filedirectory):
    if(ticks == 0):
        ticks = 1
    display_plot_acc(train_history, val_history,
                     num_epochs, ticks, filedirectory)
    display_plot_loss(train_history, val_history,
                      num_epochs, ticks, filedirectory)

    display_val_plot_acc(val_history, num_epochs, ticks, filedirectory)
    display_val_plot_loss(val_history, num_epochs, ticks, filedirectory)

    display_train_plot_acc(train_history, num_epochs,
                           ticks, filedirectory, "epoch")
    display_train_plot_loss(
        train_history, num_epochs, ticks, filedirectory, "epoch")


def display_plot_acc(train_history, val_history, num_epochs, ticks, filedirectory):
    ohist = list(map(lambda x: x.acc, train_history[::int(ticks)]))
    shist = list(map(lambda x: x.acc, val_history[::int(ticks)]))

    plt.title("Accuracy vs. Number of Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(arange(1, num_epochs+1, ticks), ohist, label="Train")
    plt.plot(arange(1, num_epochs+1, ticks), shist, label="Val")
    plt.ylim(top=1.)
    plt.xticks(np.arange(1, num_epochs+1, int(num_epochs/10)))
    plt.legend()
    plt.savefig(f'{filedirectory}/plot_acc_{ticks}.png')
    plt.close()


def display_plot_loss(train_history, val_history, num_epochs, ticks, filedirectory):
    ohist = list(map(lambda x: x.loss, train_history[::int(ticks)]))
    shist = list(map(lambda x: x.loss, val_history[::int(ticks)]))

    plt.title("Loss vs. Number of Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(arange(1, num_epochs+1, ticks), ohist, label="Train")
    plt.plot(arange(1, num_epochs+1, ticks), shist, label="Val")
    plt.xticks(np.arange(1, num_epochs+1, int(num_epochs/10)))
    plt.legend()
    plt.savefig(f'{filedirectory}/plot_loss_{ticks}.png')
    plt.close()


def display_train_plot_acc(train_history, num_epochs, ticks, filedirectory, label):
    ohist = list(map(lambda x: x.acc, train_history[::int(ticks)]))

    plt.title("Accuracy vs. Number of Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(arange(1, num_epochs+1, ticks), ohist, label="Train")
    plt.ylim(top=1.)
    plt.xticks(np.arange(1, num_epochs+1, int(num_epochs/10)))
    plt.legend()
    plt.savefig(f'{filedirectory}/plot_train_acc_{ticks}_{label}.png')
    plt.close()


def display_train_plot_loss(train_history, num_epochs, ticks, filedirectory, label):
    ohist = list(map(lambda x: x.loss, train_history[::int(ticks)]))

    plt.title("Loss vs. Number of Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(arange(1, num_epochs+1, ticks), ohist, label="Train")
    plt.xticks(np.arange(1, num_epochs+1, int(num_epochs/10)))
    plt.legend()
    plt.savefig(f'{filedirectory}/plot_train_loss_{ticks}_{label}.png')
    plt.close()


def display_val_plot_acc(val_history, num_epochs, ticks, filedirectory):
    ohist = list(map(lambda x: x.acc, val_history[::int(ticks)]))

    plt.title("Accuracy vs. Number of Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(arange(1, num_epochs+1, ticks), ohist, label="Val")
    plt.ylim(top=1.)
    plt.xticks(np.arange(1, num_epochs+1, int(num_epochs/10)))
    plt.legend()
    plt.savefig(f'{filedirectory}/plot_val_acc_{ticks}.png')
    plt.close()


def display_val_plot_loss(val_history, num_epochs, ticks, filedirectory):
    ohist = list(map(lambda x: x.loss, val_history[::int(ticks)]))

    plt.title("Loss vs. Number of Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(arange(1, num_epochs+1, ticks), ohist, label="Val")
    plt.xticks(np.arange(1, num_epochs+1, int(num_epochs/10)))
    plt.legend()
    plt.savefig(f'{filedirectory}/plot_val_loss_{ticks}.png')
    plt.close()
